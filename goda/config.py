from dataclasses import dataclass
import torch
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
    use_meta_device: bool = False
    compile_model: bool = False

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
    gradient_accumulation_steps: int = 1
    
    # checkpointing
    checkpoint_dir: str = "checkpoints"
    save_checkpoint_every_n_steps: int | None = 20
    keep_last_n_checkpoints: int = 1
    resume_from_checkpoint: str | None = "checkpoints/checkpoint_step_33.pt"
    load_best_checkpoint: bool = True 

    # wandb
    wandb_enabled: bool = True
    wandb_project: str = "goda"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    # Optimizer hyperparameters
    muon_lr: float = 3e-4
    adamw_lr: float = 3e-4
    muon_momentum: float = 0.95
    muon_beta2: float = 0.95
    muon_ns_steps: int = 5
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    weight_decay: float = 0.1


DEFAULT_CONFIG  = Config(
    embed_dim=768,
    hidden_dim=3072,
    seq_length=512,
    vocab_size=tokenizer.vocab_size,
    n_layers=12
)