
import torch
import time
import torch.nn as nn
import torch.distributed as dist
from typing import Any

from mint.eval.base import Evaluator
from goda.config import Config
from mint.tokenizer import Tokenizer
from mint.utils.device import Device
from mint.data.datasets.base import SFTEvalDataset


class ChatCoreEvaluator(Evaluator):
    def __init__(self, model: nn.Module, config: Config, tokenizer: Tokenizer, device: Device, datasets: list[SFTEvalDataset]):
        super().__init__(model, config, device)
        self.tokenizer = tokenizer
        self.datasets = datasets
    
    def evaluate(self, num_examples: int | None = None) -> dict[str, Any]:
        self.model.eval()
        eval_start_time = time.perf_counter()
        results = {}
        
        with torch.no_grad():
            for dataset in self.datasets:
                results[dataset.__class__.__name__] = self._evaluate_dataset(dataset, num_examples)
        
        self.device.synchronize()
        eval_time = time.perf_counter() - eval_start_time
        
        if not results:
            return {"tasks": {}, "accuracy": 0.0, "eval_time_sec": eval_time}
        
        avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
        return {
            "tasks": results,
            "accuracy": avg_accuracy,
            "eval_time_sec": eval_time,
        }
    
    def _evaluate_dataset(self, dataset: SFTEvalDataset, num_examples: int | None = None):
        total = len(dataset) if num_examples is None else min(num_examples, len(dataset))
        
        rank = self.process_info["rank"]
        world_size = self.process_info["world_size"]
        
        correct = 0
        count = 0
        
        for idx in range(rank, total, world_size):
            conversation = dataset[idx]
            completion = self._generate_completion(conversation)
            if dataset.evaluate(conversation, completion):
                correct += 1
            count += 1
        
        if self.process_info["distributed"]:
            correct_tensor = torch.tensor(correct, dtype=torch.float32, device=self.device.device)
            count_tensor = torch.tensor(count, dtype=torch.float32, device=self.device.device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            correct = correct_tensor.item()
            count = count_tensor.item()
        
        accuracy = correct / count if count > 0 else 0.0
        return {"accuracy": accuracy, "num_examples": int(count)}
    
    def _generate_completion(self, conversation: dict) -> str:
        import copy
        
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        messages.pop()
        
        # Reserve 1 token for assistant_start to stay within seq_length
        ids, _ = self.tokenizer.render_conversation(conversation, self.config.seq_length - 1)
        
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            next_token = logits[0, -1, :].argmax().item()
        
        return self.tokenizer.decode(torch.tensor([next_token]))[0]