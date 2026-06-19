import importlib.util
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from goda.checkpointer import Checkpointer
from goda.config import Config
from goda.dataloader import DistributedDataloader
from goda.device import Device
from goda.eval import BPBEvaluator, CoreEvaluator, Evaluator
from goda.logger import logger
from goda.scheduler import Scheduler


class PreTrainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, dataloader: DistributedDataloader, device: Device, config: Config, tokenizer: Any = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
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

        self.core_evaluator = CoreEvaluator(
            model=model,
            config=config,
            tokenizer=tokenizer,
            device=device,
        )

        self.bpb_evaluator = BPBEvaluator(
            model=model,
            config=config,
            device=device,
            dataloader=dataloader,
        )

        self.start_step = 0
        if config.resume_from_checkpoint:
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

        self.start_step = checkpoint_info["step"]
        self.dataloader_state = self._extract_rank_dataloader_state(
            checkpoint_info.get("dataloader_state", {})
        )

        if self.dataloader_state and hasattr(self.dataloader, "set_state"):
            self.dataloader.set_state(self.dataloader_state)
            logger.info(
                f"Restored dataloader state: "
                f"shard={self.dataloader_state.get('shard_idx', 0)}, "
                f"rg={self.dataloader_state.get('rg_idx', 0)}, "
                f"batches={self.dataloader_state.get('batches_consumed', 0)}"
            )

        logger.info(f"Resuming training from step {self.start_step}")

    def _local_dataloader_state(self) -> dict:
        if hasattr(self.dataloader, "get_state"):
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

    def _run_evaluation( self, evaluator: Evaluator, step: int, metric_prefix: str, save_checkpoint: bool = False, **eval_kwargs) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            results = evaluator.evaluate(**eval_kwargs)
        self.model.train()

        metrics = {}

        def flatten(prefix: str, value: Any):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flatten(f"{prefix}/{sub_key}", sub_value)
            else:
                metrics[prefix] = value

        for key, value in results.items():
            flatten(f"{metric_prefix}/{key}", value)

        if self.is_main_process:
            log_items = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            logger.info(f"Eval complete | step={step} | {log_items}")

        self._log_wandb(metrics, step=step)

        if self.is_main_process and results and save_checkpoint:
            val_loss = results.get("loss")
            if val_loss is not None:
                is_best = self.checkpointer.should_checkpoint_on_eval(val_loss)
                self.checkpointer.save_checkpoint(
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=self._collect_dataloader_state(),
                    val_loss=val_loss,
                    is_best=is_best,
                )

        return results

    def train(self):
        self.model.train()
        train_start_time = time.perf_counter()
        accumulated_loss = 0.0
        micro_step = 0

        resume_state = getattr(self, "dataloader_state", None)

        # Don't use enumerate with start parameter - it causes step/micro_step mismatch
        # Instead, manually track the step counter
        step = self.start_step
        batch_iterator = self.dataloader.batch_loader(split="train", resume_state=resume_state)

        for inputs, targets in batch_iterator:
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

                # Clear cache periodically to reduce fragmentation
                if step % 10 == 0:
                    self.device.empty_cache()

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

                # Increment step counter after completing gradient accumulation
                step += 1

                if step % self.config.eval_every_n_steps == 0 and step > 0:
                    self._run_evaluation(
                        evaluator=self.bpb_evaluator,
                        step=step,
                        metric_prefix="val",
                        save_checkpoint=True,
                        num_steps=self.config.eval_num_steps,
                    )

                    if self.is_main_process:
                        samples = self.dataloader.sample(num_samples=3)
                        for i, sample in enumerate(samples, 1):
                            with torch.no_grad():
                                input_tensor = sample["input_tokens"].unsqueeze(0).to(self.device.device)
                                logits = self.model(input_tensor)
                                pred_tokens = logits.argmax(dim=-1).squeeze(0)
                                pred_str = self.tokenizer.decode(pred_tokens.unsqueeze(0))[0]

                            logger.info(f"Sample {i}:")
                            logger.info(f"Input:  ...{sample['input_str'][-100:]}")
                            logger.info(f"Target: ...{sample['target_str'][-100:]}")
                            logger.info(f"Pred:   ...{pred_str[-100:]}")

                if step % self.config.core_eval_every_n_step == 0 and step > 0:
                    self._run_evaluation(
                        evaluator=self.core_evaluator,
                        step=step,
                        metric_prefix="core",
                        task_labels=["hellaswag_zeroshot"],
                        limit=1,
                    )

                if self.is_main_process and self.config.save_checkpoint_every_n_steps is not None:
                    if step % self.config.save_checkpoint_every_n_steps == 0 and step > 0:
                        self.checkpointer.save_checkpoint(
                            step=step,
                            model=self.model,
                            optimizer=self.optimizer,
                            dataloader_state=self._collect_dataloader_state(),
                        )

                if step % self.config.log_every_n_steps == 0:
                    logger.info(
                        f"Step {step:4d} | "
                        f"Loss: {metrics['train/loss']:.4f} | "
                        f"Step Time: {metrics['train/step_time_sec']:.4f}s | "
                        f"Tokens/s: {metrics['train/tokens_per_second']:.2f} | "
                        f"Memory: {self.device.memory()}"
                    )
                    self._log_wandb({**metrics, "train/memory": self.device.memory()}, step=step)

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
