import importlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from mint.config.base import Config
from mint.data.dataloader import DistributedDataloader
from mint.eval.base import EvalConfig
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.checkpointer import CheckpointerConfig
from mint.utils.device import Device
from mint.utils.logger import LoggerConfig, logger


@dataclass
class BasetrainConfig(Config):
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    use_meta_device: bool = True
    compile_model: bool = True

    train_num_steps: int = 1000
    grad_clip: float = 1.0
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    eval_num_steps: int = 10
    core_eval_every_n_step: int = 500
    gradient_accumulation_steps: int = 1

    ckpt: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    lg: LoggerConfig = field(default_factory=LoggerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: DistributedDataloader,
        device: Device,
        config: BasetrainConfig,
        tokenizer: Any = None,  # noqa: ANN401
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
        self.process_info = self.device.process_info()
        self.is_main_process = self.process_info["is_main"]
        self.wandb_run = self._init_wandb()

    def _init_wandb(self) -> Any | None:  # noqa: ANN401
        if not self.config.lg.wandb_enabled or not self.is_main_process:
            return None

        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is enabled in config but the package is not installed.")

        wandb = __import__("wandb")

        run = wandb.init(
            project=self.config.lg.wandb_project,
            name=self.config.lg.wandb_run_name,
            entity=self.config.lg.wandb_entity,
            config={
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in self.config.__dict__.items()
            },
        )
        logger.info(f"W&B initialized | project={self.config.lg.wandb_project} | run={run.name}")
        return run
