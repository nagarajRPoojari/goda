from dataclasses import dataclass
import torch
import yaml
from pathlib import Path
from typing import Any, Dict
from goda.tokenizer import Tokenizer
tokenizer = Tokenizer()

@dataclass
class Config:
    # model hyperparameters
    embed_dim: int
    hidden_dim: int
    seq_length: int
    vocab_size: int
    n_layers: int
    dtype: torch.dtype = torch.float32
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 128
    rope_base_theta: int = 100000
    tie_weights: bool = True

    mixed_precision: bool = True
    
    # Model initialization and compilation
    use_meta_device: bool = True
    compile_model: bool = True

    # dataloader
    data_dir: str = "data/climbmix"
    batch_size: int = 4
    buffer_size: int = 1000
    tokenizer_batch_size: int = 128

    # trainer
    train_num_steps: int = 1000
    grad_clip: float = 1.0
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    eval_num_steps: int = 10
    core_eval_every_n_step: int = 500
    gradient_accumulation_steps: int = 1
    
    # checkpointing
    checkpoint_dir: str = "checkpoints"
    save_checkpoint_every_n_steps: int | None = 200
    keep_last_n_checkpoints: int = 1
    resume_from_checkpoint: str | None = "checkpoints/checkpoint_step_33.pt"
    load_best_checkpoint: bool = True 

    # wandb
    wandb_enabled: bool = True
    wandb_project: str = "goda"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    # Optimizer hyperparameters
    # Learning rates
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    
    # Muon hyperparameters (for matrix params)
    muon_momentum: float = 0.95
    muon_beta2: float = 0.9
    muon_ns_steps: int = 5
    
    # AdamW hyperparameters (for embeddings, lm_head, scalars)
    adamw_beta1: float = 0.8
    adamw_beta2_lm_head: float = 0.96
    adamw_beta2_embedding: float = 0.995
    adamw_beta2_scalar: float = 0.95
    adamw_eps: float = 1e-10
    
    # Weight decay
    weight_decay: float = 0.0 # initial place holder, only used in step 0
    weight_decay_lm_head: float = 0.01
    weight_decay_embedding: float = 0.001
    weight_decay_scalar: float = 0.05
    
    # Scalar LR multiplier
    scalar_lr_multiplier: float = 0.01
    
    # Scheduler hyperparameters
    warmup_steps: int = 0
    warmdown_ratio: float = 0.0
    final_lr_frac: float = 0.1
    muon_momentum_warmup_steps: int = 400
    muon_momentum_start: float = 0.85
    muon_momentum_peak: float = 0.97
    muon_momentum_final: float = 0.90

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if value == 'null' or value == 'None':
                config_dict[key] = None
        
        return cls(**config_dict)


DEFAULT_CONFIG  = Config(
    embed_dim=768,
    hidden_dim=3072,
    seq_length=512,
    vocab_size=tokenizer.vocab_size,
    n_layers=12
)