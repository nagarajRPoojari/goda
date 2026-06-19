import math
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from mint.eval.base import EvalConfig, Evaluator
from mint.utils.device import Device


class BPBEvaluator(Evaluator):
    def __init__(self, model: nn.Module, config: EvalConfig, device: Device, dataloader: Any):
        super().__init__(model, config, device)
        self.dataloader = dataloader

    def evaluate(self, num_steps: int = 10, step: int | None = None) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        eval_start_time = time.perf_counter()
        total_tokens = 0

        with torch.no_grad():
            for eval_step, (inputs, targets) in enumerate(
                self.dataloader.batch_loader(split="val")
            ):
                if eval_step >= num_steps:
                    break

                with self.device.autocast():
                    logits = self.model(inputs)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
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

        tokens_per_second = (
            (total_tokens * self.process_info["world_size"]) / eval_time if eval_time > 0 else 0.0
        )

        return {
            "loss": avg_loss,
            "bpb": bpb,
            "eval_time_sec": eval_time,
            "tokens_per_second": tokens_per_second,
        }
