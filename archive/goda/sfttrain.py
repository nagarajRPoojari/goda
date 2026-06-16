import importlib.util
from typing import Any
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from goda.checkpointer import Checkpointer
from goda.config import Config
from goda.dataloader import DistributedSFTDataloader
from goda.device import Device
from goda.logger import logger
from goda.scheduler import Scheduler
from goda.eval import ChatCoreEvaluator
from torch.optim import Optimizer


class SFTTrainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, dataloader: DistributedSFTDataloader,
                 device: Device, config: Config, tokenizer: Any = None, eval_datasets: list | None = None) -> None:
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
        
        self.evaluator = None
        if eval_datasets:
            self.evaluator = ChatCoreEvaluator(
                model=model,
                config=config,
                tokenizer=tokenizer,
                device=device,
                datasets=eval_datasets,
            )
            
            if config.resume_from_checkpoint:
                self._load_pretrained_checkpoint()
        
    def _load_pretrained_checkpoint(self):
        checkpoint_info = self.checkpointer.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=self.config.resume_from_checkpoint,
            load_best=self.config.load_best_checkpoint,
        )
            
        logger.info(f"Loaded pretrained checkpoint from step {checkpoint_info.get('step', 'unknown')}")
        
        # Check for NaN/Inf in model parameters
        nan_params = []
        inf_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
        
        if nan_params:
            logger.error(f"NaN detected in model parameters after loading checkpoint: {nan_params}")
        if inf_params:
            logger.error(f"Inf detected in model parameters after loading checkpoint: {inf_params}")
        
        logger.info("Starting SFT training from step 0")
    
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

    def _log_wandb(self, metrics: dict, step: int):
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
    
    def _run_evaluation(self, step: int, num_examples: int | None = None):
        if self.evaluator is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            results = self.evaluator.evaluate(num_examples=num_examples)
        self.model.train()
        
        metrics = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    metrics[f"eval/{key}/{sub_key}"] = sub_value
            else:
                metrics[f"eval/{key}"] = value
        
        if self.is_main_process:
            log_items = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, (int, float)))
            logger.info(f"Eval complete | step={step} | {log_items}")
        
        self._log_wandb(metrics, step=step)

    def train(self):
        self.model.train()
        train_start_time = time.perf_counter()
        accumulated_loss = 0.0
        micro_step = 0
        step = 0
        
        for step, (inputs, targets, mask) in enumerate(self.dataloader.batch_loader(split="train"), start=0):
            if self.checkpointer.interrupt_requested:
                if self.is_main_process:
                    logger.warning("Saving checkpoint due to keyboard interrupt...")
                    self.checkpointer.save_checkpoint(
                        step=step,
                        model=self.model,
                        optimizer=self.optimizer,
                        dataloader_state={},
                        force=True,
                    )
                    logger.info("Checkpoint saved. Exiting gracefully.")
                break
            
            if step >= self.config.train_num_steps:
                break

            step_start_time = time.perf_counter()
            is_accumulating = (micro_step + 1) % self.config.gradient_accumulation_steps != 0

            # Check for NaN/Inf in inputs before forward pass
            if torch.isnan(inputs).any():
                logger.error(f"NaN detected in inputs at step {step}")
            if torch.isinf(inputs).any():
                logger.error(f"Inf detected in inputs at step {step}")
            
            with self.device.autocast():
                logits = self.model(inputs)
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    logger.warning(f"NaN detected in logits at step {step}")
                    # Log some statistics about the inputs
                    logger.warning(f"Input shape: {inputs.shape}, min: {inputs.min()}, max: {inputs.max()}, mean: {inputs.float().mean()}")
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='none'
                )
                
                mask_sum = mask.sum()
                if mask_sum > 0:
                    masked_loss = (loss * mask.view(-1).float()).sum()
                    loss = masked_loss / mask_sum.float()
                else:
                    logger.warning(f"Zero mask sum at step {step}, skipping batch")
                    loss = torch.tensor(0.0, device=self.device.device, dtype=torch.float32)
                
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
                tokens_per_step = mask.sum().item() * self.process_info["world_size"] * self.config.gradient_accumulation_steps
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

                if (step + 1) % self.config.eval_every_n_steps == 0 and step > 0:
                    self._run_evaluation(step=step, num_examples=100)
                    
                    if self.is_main_process and self.tokenizer:
                        samples = self.dataloader.sample(num_samples=3)
                        for i, sample in enumerate(samples, 1):
                            with torch.no_grad():
                                input_tensor = sample['input_tokens'].unsqueeze(0).to(self.device.device)
                                logits = self.model(input_tensor)
                                pred_tokens = logits.argmax(dim=-1).squeeze(0)
                                pred_str = self.tokenizer.decode(pred_tokens.unsqueeze(0))[0]
                            
                            logger.info(f"Sample {i}:")
                            logger.info(f"Input:  ...{sample['input_str'][-100:]}")
                            logger.info(f"Target: ...{sample['target_str'][-100:]}")
                            logger.info(f"Pred:   ...{pred_str[-100:]}")
                
                if self.is_main_process and self.config.save_checkpoint_every_n_steps is not None:
                    if (step + 1) % self.config.save_checkpoint_every_n_steps == 0 and step > 0:
                        self.checkpointer.save_checkpoint(
                            step=step,
                            model=self.model,
                            optimizer=self.optimizer,
                            dataloader_state={},
                        )

                if (step + 1) % self.config.log_every_n_steps == 0:
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
                dataloader_state={},
                force=True,
            )
        
        if self.wandb_run is not None:
            self.wandb_run.finish()
