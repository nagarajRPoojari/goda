import importlib.util
from typing import Any
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from goda.checkpointer import Checkpointer
from goda.config import Config
from goda.dataloader import DistributedDataloader
from goda.device import Device
from goda.logger import logger
from goda.scheduler import Scheduler
from torch.optim import Optimizer


class Trainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, dataloader: DistributedDataloader, device: Device, config: Config) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.process_info = self.device.process_info()
        self.is_main_process = self.process_info["is_main"]
        self.wandb_run = self._init_wandb()
        
        self.checkpointer = Checkpointer(
            checkpoint_dir=config.checkpoint_dir,
            save_every_n_steps=config.save_checkpoint_every_n_steps,
            keep_last_n=config.keep_last_n_checkpoints,
            is_main_process=self.is_main_process,
        )
        
        self.scheduler = Scheduler(
            num_iterations=config.train_num_steps,
            warmup_steps=config.warmup_steps,
            warmdown_ratio=config.warmdown_ratio,
            final_lr_frac=config.final_lr_frac,
            muon_momentum_warmup_steps=config.muon_momentum_warmup_steps,
            muon_momentum_start=config.muon_momentum_start,
            muon_momentum_peak=config.muon_momentum_peak,
            muon_momentum_final=config.muon_momentum_final,
            weight_decay=config.weight_decay,
        )
        
        # Resume from checkpoint if specified
        self.start_step = 0
        if config.resume_from_checkpoint is not None:
            self._resume_from_checkpoint()

    def _init_wandb(self) -> Any | None:
        if not self.config.wandb_enabled or not self.is_main_process:
            return None

        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is enabled in config but the package is not installed.")

        wandb = __import__("wandb")

        run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            entity=self.config.wandb_entity,
            config={
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in self.config.__dict__.items()
            },
        )
        logger.info(f"W&B initialized | project={self.config.wandb_project} | run={run.name}")
        return run
    
    def _resume_from_checkpoint(self):
        checkpoint_info = self.checkpointer.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=self.config.resume_from_checkpoint,
            load_best=self.config.load_best_checkpoint,
        )
        
        self.start_step = checkpoint_info['step']
        self.dataloader_state = self._extract_rank_dataloader_state(
            checkpoint_info.get('dataloader_state', {})
        )
        
        if self.dataloader_state and hasattr(self.dataloader, 'set_state'):
            self.dataloader.set_state(self.dataloader_state)
            logger.info(
                f"Restored dataloader state: "
                f"shard={self.dataloader_state.get('shard_idx', 0)}, "
                f"rg={self.dataloader_state.get('rg_idx', 0)}, "
                f"batches={self.dataloader_state.get('batches_consumed', 0)}"
            )
        
        logger.info(f"Resuming training from step {self.start_step}")
    
    def _local_dataloader_state(self) -> dict:
        if hasattr(self.dataloader, 'get_state'):
            return self.dataloader.get_state()
        return {}
    

    # executed by main process only
    # collect all local dataloader state & save full checkpoint
    def _collect_dataloader_state(self) -> dict:
        local_state = self._local_dataloader_state()
        if not self.process_info["distributed"]:
            return local_state
        
        if self.is_main_process:
            gathered_states: list[dict] = [{} for _ in range(self.process_info["world_size"])]
            dist.gather_object(local_state, gathered_states, dst=0)
            return {
                "per_rank": {
                    rank: state for rank, state in enumerate(gathered_states)
                }
            }
        
        dist.gather_object(local_state, None, dst=0)
        return {}
    
    def _extract_rank_dataloader_state(self, checkpoint_dataloader_state: dict) -> dict:
        if not checkpoint_dataloader_state:
            return {}
        
        if not self.process_info["distributed"]:
            return checkpoint_dataloader_state
        
        per_rank_state = checkpoint_dataloader_state.get("per_rank", {})
        return per_rank_state.get(self.process_info["rank"], {})

    def _log_wandb(self, metrics: dict, step: int):
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def train(self):
        self.model.train()
        train_start_time = time.perf_counter()
        accumulated_loss = 0.0
        micro_step = 0
        step = self.start_step
        
        resume_state = getattr(self, 'dataloader_state', None)

        for step, (inputs, targets) in enumerate(
            self.dataloader.batch_loader(split="train", resume_state=resume_state), start=self.start_step
        ):

            if self.checkpointer.interrupt_requested:
                if self.is_main_process:
                    logger.warning("Saving checkpoint due to keyboard interrupt...")
                    self.checkpointer.save_checkpoint(
                        step=step,
                        model=self.model,
                        optimizer=self.optimizer,
                        dataloader_state=self._collect_dataloader_state(),
                        force=True,
                    )
                    logger.info("Checkpoint saved. Exiting gracefully.")
                break
            
            if step >= self.config.train_num_steps:
                break

            step_start_time = time.perf_counter()
            is_accumulating = (micro_step + 1) % self.config.gradient_accumulation_steps != 0

            with self.device.autocast():
                logits = self.model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss = loss / self.config.gradient_accumulation_steps

            if micro_step % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            self.device.backward(loss)
            accumulated_loss += loss.item()
            micro_step += 1

            if not is_accumulating:
                scheduler_metrics = self.scheduler.step(self.optimizer, step)
                self.device.optimizer_step(
                    self.optimizer,
                    grad_clip=self.config.grad_clip,
                    params=self.model.parameters()
                )
                self.device.synchronize()

                step_time = time.perf_counter() - step_start_time
                tokens_per_step = targets.numel() * self.process_info["world_size"] * self.config.gradient_accumulation_steps
                tokens_per_second = tokens_per_step / step_time if step_time > 0 else 0.0
                elapsed_time = time.perf_counter() - train_start_time

                metrics = {
                    "train/loss": accumulated_loss,
                    "train/step_time_sec": step_time,
                    "train/tokens_per_step": tokens_per_step,
                    "train/tokens_per_second": tokens_per_second,
                    "train/elapsed_time_sec": elapsed_time,
                    "scheduler/lr_multiplier": scheduler_metrics["lr_multiplier"],
                    "scheduler/muon_momentum": scheduler_metrics["muon_momentum"],
                    "scheduler/muon_weight_decay": scheduler_metrics["muon_weight_decay"],
                }

                if step % self.config.log_every_n_steps == 0:
                    logger.info(
                        f"Step {step:4d} | "
                        f"Loss: {metrics['train/loss']:.4f} | "
                        f"Step Time: {metrics['train/step_time_sec']:.4f}s | "
                        f"Tokens/s: {metrics['train/tokens_per_second']:.2f} | "
                        f"Memory: {self.device.memory()}"
                    )
                    self._log_wandb({**metrics, "train/memory": self.device.memory()}, step=step)

                if step % self.config.eval_every_n_steps == 0 and step > 0:
                    val_loss = self.evaluate(num_steps=self.config.eval_num_steps, step=step)
                    self.model.train()
                    metrics["val/loss"] = val_loss
                    
                    if self.is_main_process:
                        is_best = self.checkpointer.should_checkpoint_on_eval(val_loss)
                        self.checkpointer.save_checkpoint(
                            step=step,
                            model=self.model,
                            optimizer=self.optimizer,
                            dataloader_state=self._collect_dataloader_state(),
                            val_loss=val_loss,
                            is_best=is_best,
                        )
                
                if self.is_main_process and self.config.save_checkpoint_every_n_steps is not None:
                    if step % self.config.save_checkpoint_every_n_steps == 0 and step > 0:
                        self.checkpointer.save_checkpoint(
                            step=step,
                            model=self.model,
                            optimizer=self.optimizer,
                            dataloader_state=self._collect_dataloader_state(),
                        )
                accumulated_loss = 0.0

        if self.is_main_process:
            logger.info("Saving final checkpoint")
            self.checkpointer.save_checkpoint(
                step=step,
                model=self.model,
                optimizer=self.optimizer,
                dataloader_state=self._collect_dataloader_state(),
                force=True,
            )
        
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def evaluate(self, num_steps=10, step: int | None = None):
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
        tokens_per_second = (total_tokens * self.process_info["world_size"]) / eval_time if eval_time > 0 else 0.0

        logger.info(f"\n{'='*50}")
        logger.info(
            f"Validation Loss: {avg_loss:.4f} | "
            f"Eval Time: {eval_time:.4f}s | "
            f"Tokens/s: {tokens_per_second:.2f}"
        )
        logger.info(f"{'='*50}\n")

        if step is not None:
            self._log_wandb(
                {
                    "val/loss": avg_loss,
                    "val/eval_time_sec": eval_time,
                    "val/tokens_per_second": tokens_per_second,
                },
                step=step,
            )

        return avg_loss