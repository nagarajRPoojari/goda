from goda.tokenizer import Tokenizer
from typing import Any
from abc import ABC, abstractmethod

from goda.device import Device
from goda.config import Config
from goda.logger import logger
import json
import random
from pathlib import Path
import math
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import yaml

from enum import Enum

class Task(Enum):
    MC = "multiple_choice"
    SCHEMA = "schema"
    LM = "language_modeling"


class Evaluator(ABC):
    
    def __init__(self, model: nn.Module, config: Config, device: Device):
        self.model = model
        self.config = config
        self.device = device
        self.process_info = device.process_info()
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, Any]:
        pass


class BPBEvaluator(Evaluator):
    
    def __init__(self, model: nn.Module, config: Config, device: Device, dataloader: Any):
        super().__init__(model, config, device)
        self.dataloader = dataloader
    
    def evaluate(self, num_steps: int = 10, step: int | None = None) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        eval_start_time = time.perf_counter()
        total_tokens = 0

        with torch.no_grad():
            for eval_step, (inputs, targets) in enumerate(self.dataloader.batch_loader(split="val")):
                if eval_step >= num_steps:
                    break

                with self.device.autocast():
                    logits = self.model(inputs)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )

                total_loss += loss.item()
                total_tokens += targets.numel()

        self.device.synchronize()
        eval_time = time.perf_counter() - eval_start_time
        avg_loss = total_loss / num_steps
        
        # Reduce validation loss across all GPUs to get global average
        if self.process_info["distributed"]:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        # Calculate bits per byte (BPB) metric
        # BPB = loss / ln(2), converting from nats to bits
        bpb: float = avg_loss / math.log(2)
        # TODO: divide it by bytes per token ratio
        
        tokens_per_second = (total_tokens * self.process_info["world_size"]) / eval_time if eval_time > 0 else 0.0

        return {
            "loss": avg_loss,
            "bpb": bpb,
            "eval_time_sec": eval_time,
            "tokens_per_second": tokens_per_second,
        }


class CoreEvaluator(Evaluator):

    def __init__(self, model: nn.Module, config: Config, tokenizer: Tokenizer, device: Device, bundle_dir: str = "data/eval_bundle", config_path: str = "data/eval_bundle/core.yaml", seed: int = 42):
        super().__init__(model, config, device)
        self.tokenizer = tokenizer
        self.bundle_dir = Path(bundle_dir)
        self.config_path = Path(config_path)
        self.seed = seed
        self.pad_token_id = getattr(tokenizer, "pad_token", None)
        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token is required")

        with open(self.config_path, "r") as f:
            raw = yaml.safe_load(f)

        self.tasks = raw.get("icl_tasks", [])

    def evaluate(self, task_labels: list[str] | None = None, limit: int | None = None):
        self.model.eval()
        eval_start_time = time.perf_counter()
        results = {}
        selected_tasks = self.tasks
        if task_labels is not None:
            wanted = set(task_labels)
            selected_tasks = [task for task in self.tasks if task["label"] in wanted]

        with torch.no_grad():
            for task in selected_tasks:
                results[task["label"]] = self.evaluate_task(task, limit=limit)

        self.device.synchronize()
        eval_time = time.perf_counter() - eval_start_time
        total_examples = sum(task_result["num_examples"] for task_result in results.values())

        if not results:
            return {
                "tasks": {},
                "core": 0.0,
                "eval_time_sec": eval_time,
                "tokens_per_second": 0.0,
            }

        core = sum(task_result["accuracy"] for task_result in results.values()) / len(results)
        tokens_per_second = (total_examples * self.process_info["world_size"]) / eval_time if eval_time > 0 else 0.0
        return {
            "tasks": results,
            "core": core,
            "eval_time_sec": eval_time,
            "tokens_per_second": tokens_per_second,
        }

    def evaluate_task(self, task_meta: dict, limit: int | None = None):
        data = self._load_dataset(task_meta["dataset_uri"])
        if limit is not None:
            data = data[:limit]

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        correct = torch.zeros(len(data), dtype=torch.float32, device=self.device.device)

        for idx in range(rank, len(data), world_size):
            correct[idx] = float(self.evaluate_example(idx, data, task_meta))

        if world_size > 1:
            dist.barrier()
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)

        accuracy = correct.mean().item() if len(data) > 0 else 0.0
        return {
            "accuracy": accuracy,
            "num_examples": len(data),
        }

    def evaluate_example(self, idx: int, data: list[dict], task_meta: dict):
        item = data[idx]
        fewshot_examples = self._sample_fewshot(idx, data, task_meta)

        task_type: Task = Task(task_meta["icl_task_type"])
        continuation_delimiter = task_meta.get("continuation_delimiter", " ")

        if task_type == Task.MC:
            prompts = self._render_multiple_choice(item, fewshot_examples, continuation_delimiter)
            token_lists = self._encode_prompts(prompts)
            prompt_token_lists = self._encode_prompts(
                self._render_multiple_choice_prefixes(item, fewshot_examples, continuation_delimiter)
            )
            losses = []
            for full_tokens, prompt_tokens in zip(token_lists, prompt_token_lists):
                losses.append(self._continuation_mean_loss(full_tokens, len(prompt_tokens)))
            pred = min(range(len(losses)), key=lambda i: losses[i])
            return pred == item["gold"]

        if task_type == Task.SCHEMA:
            prompts = self._render_schema(item, fewshot_examples, continuation_delimiter)
            token_lists = self._encode_prompts(prompts)
            prompt_token_lists = self._encode_prompts(
                self._render_schema_prefixes(item, fewshot_examples, continuation_delimiter)
            )
            losses = []
            for full_tokens, prompt_tokens in zip(token_lists, prompt_token_lists):
                losses.append(self._continuation_mean_loss(full_tokens, len(prompt_tokens)))
            pred = min(range(len(losses)), key=lambda i: losses[i])
            return pred == item["gold"]

        if task_type == Task.LM:
            prompt, full = self._render_language_modeling(item, fewshot_examples, continuation_delimiter)
            prompt_tokens = self._encode_single(prompt)
            full_tokens = self._encode_single(full)
            return self._language_model_exact_match(full_tokens, len(prompt_tokens))

    def _load_dataset(self, dataset_uri: str):
        path = self.bundle_dir / "eval_data" / dataset_uri
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _sample_fewshot(self, idx: int, data: list[dict], task_meta: dict):
        num_fewshot_values = task_meta.get("num_fewshot", [0])
        num_fewshot = num_fewshot_values[0] if num_fewshot_values else 0
        if num_fewshot <= 0:
            return []

        if task_meta.get("has_categories") and "category" in data[idx]:
            category = data[idx]["category"]
            candidate_indices = [
                i for i, row in enumerate(data)
                if i != idx and row.get("category") == category
            ]
        else:
            candidate_indices = [i for i in range(len(data)) if i != idx]

        if not candidate_indices:
            return []

        count = min(num_fewshot, len(candidate_indices))
        rng = random.Random(self.seed + idx)
        chosen = rng.sample(candidate_indices, count)
        return [data[i] for i in chosen]

    def _render_multiple_choice(self, item: dict, fewshot_examples: list[dict], delimiter: str):
        prompt_prefix = self._render_multiple_choice_shared_prefix(item, fewshot_examples, delimiter)
        return [prompt_prefix + choice for choice in item["choices"]]

    def _render_multiple_choice_prefixes(self, item: dict, fewshot_examples: list[dict], delimiter: str):
        prompt_prefix = self._render_multiple_choice_shared_prefix(item, fewshot_examples, delimiter)
        return [prompt_prefix for _ in item["choices"]]

    def _render_multiple_choice_shared_prefix(self, item: dict, fewshot_examples: list[dict], delimiter: str):
        parts = []
        for example in fewshot_examples:
            parts.append(f'{example["query"]}{delimiter}{example["choices"][example["gold"]]}')
        parts.append(f'{item["query"]}{delimiter}')
        return "\n\n".join(parts)

    def _render_schema(self, item: dict, fewshot_examples: list[dict], delimiter: str):
        shared_suffix = f'{delimiter}{item["continuation"]}'
        prompts = []
        prefix = self._render_schema_fewshot_prefix(fewshot_examples, delimiter)
        for option in item["context_options"]:
            if prefix:
                prompts.append(f"{prefix}\n\n{option}{shared_suffix}")
            else:
                prompts.append(f"{option}{shared_suffix}")
        return prompts

    def _render_schema_prefixes(self, item: dict, fewshot_examples: list[dict], delimiter: str):
        prefix = self._render_schema_fewshot_prefix(fewshot_examples, delimiter)
        prefixes = []
        for option in item["context_options"]:
            if prefix:
                prefixes.append(f"{prefix}\n\n{option}{delimiter}")
            else:
                prefixes.append(f"{option}{delimiter}")
        return prefixes

    def _render_schema_fewshot_prefix(self, fewshot_examples: list[dict], delimiter: str):
        parts = []
        for example in fewshot_examples:
            parts.append(
                f'{example["context_options"][example["gold"]]}{delimiter}{example["continuation"]}'
            )
        return "\n\n".join(parts)

    def _render_language_modeling(self, item: dict, fewshot_examples: list[dict], delimiter: str):
        parts = []
        for example in fewshot_examples:
            parts.append(
                f'{example["context"].rstrip()}{delimiter}{example["continuation"]}'
            )
        prefix = f'{item["context"].rstrip()}{delimiter}'
        if parts:
            prompt = "\n\n".join(parts + [prefix])
        else:
            prompt = prefix
        full = prompt + item["continuation"]
        return prompt, full

    def _encode_prompts(self, prompts: list[str]):
        return self.tokenizer.encode_to_list(prompts, add_bos=True, add_eos=False, padding=False)

    def _encode_single(self, text: str):
        return self.tokenizer.encode_to_list([text], add_bos=True, add_eos=False, padding=False)[0]

    def _prepare_input(self, token_ids: list[int]):
        max_len = int(self.config.seq_length)
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
        return torch.tensor([token_ids], dtype=torch.long, device=self.device.device)

    def _forward_losses_and_predictions(self, token_ids: list[int]):
        input_ids = self._prepare_input(token_ids)
        logits = self.model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(1, -1)
        predictions = shift_logits.argmax(dim=-1)
        return input_ids, losses[0], predictions[0]

    def _continuation_mean_loss(self, full_tokens: list[int], prompt_len: int):
        max_len = int(self.config.seq_length)
        if len(full_tokens) > max_len:
            crop = len(full_tokens) - max_len
            full_tokens = full_tokens[crop:]
            prompt_len = max(0, prompt_len - crop)

        if prompt_len >= len(full_tokens):
            return float("inf")

        _, losses, _ = self._forward_losses_and_predictions(full_tokens)
        start = max(prompt_len - 1, 0)
        end = len(full_tokens) - 1
        span = losses[start:end]
        if span.numel() == 0:
            return float("inf")
        return span.mean().item()

    def _language_model_exact_match(self, full_tokens: list[int], prompt_len: int):
        max_len = int(self.config.seq_length)
        if len(full_tokens) > max_len:
            crop = len(full_tokens) - max_len
            full_tokens = full_tokens[crop:]
            prompt_len = max(0, prompt_len - crop)

        if prompt_len >= len(full_tokens):
            return False

        input_ids, _, predictions = self._forward_losses_and_predictions(full_tokens)
        start = max(prompt_len - 1, 0)
        end = len(full_tokens) - 1
        predicted = predictions[start:end]
        actual = input_ids[0, start + 1:end + 1]
        if predicted.numel() == 0:
            return False
        return bool(torch.equal(predicted, actual))
