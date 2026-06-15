import math
from dataclasses import dataclass

from mint.config.base import Config


@dataclass
class SchedulerConfig(Config):
    weight_decay: float = 0.0  # initial place holder, only used in step 0
    weight_decay_lm_head: float = 0.01
    weight_decay_embedding: float = 0.001
    weight_decay_scalar: float = 0.05

    # Scalar LR multiplier
    scalar_lr_multiplier: float = 0.01

    warmup_steps: int = 0
    warmdown_ratio: float = 0.0
    final_lr_frac: float = 0.1
    muon_momentum_warmup_steps: int = 400
    muon_momentum_start: float = 0.85
    muon_momentum_peak: float = 0.97
    muon_momentum_final: float = 0.90

    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5


class Scheduler:
    def __init__(
        self,
        num_iterations: int,
        config: SchedulerConfig,
    ):
        self.num_iterations = num_iterations
        self.warmup_steps = config.warmup_steps
        self.warmdown_iters = round(config.warmdown_ratio * num_iterations)
        self.warmdown_start = num_iterations - self.warmdown_iters
        self.final_lr_frac = config.final_lr_frac

        self.muon_momentum_warmup_steps = config.muon_momentum_warmup_steps
        self.muon_momentum_start = config.muon_momentum_start
        self.muon_momentum_peak = config.muon_momentum_peak
        self.muon_momentum_final = config.muon_momentum_final

        self.weight_decay = config.weight_decay

    def get_lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        elif step <= self.warmdown_start:
            return 1.0
        else:
            progress = (self.num_iterations - step) / self.warmdown_iters
            return progress * 1.0 + (1 - progress) * self.final_lr_frac

    def get_muon_momentum(self, step: int) -> float:
        if step < self.muon_momentum_warmup_steps:
            frac = step / self.muon_momentum_warmup_steps
            return (
                1 - frac
            ) * self.muon_momentum_start + frac * self.muon_momentum_peak
        elif step >= self.warmdown_start:
            progress = (step - self.warmdown_start) / self.warmdown_iters
            return (
                self.muon_momentum_peak * (1 - progress)
                + self.muon_momentum_final * progress
            )
        else:
            return self.muon_momentum_peak

    def get_weight_decay(self, step: int) -> float:
        return (
            self.weight_decay
            * 0.5
            * (1 + math.cos(math.pi * step / self.num_iterations))
        )

    def step(self, optimizer, step: int) -> dict:
        lrm = self.get_lr_multiplier(step)
        muon_momentum = self.get_muon_momentum(step)
        muon_weight_decay = self.get_weight_decay(step)

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group.get("kind") == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

        return {
            "lr_multiplier": lrm,
            "muon_momentum": muon_momentum,
            "muon_weight_decay": muon_weight_decay,
        }
