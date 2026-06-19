from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from mint.kvcache.base import KVCache
from mint.nn.base import ModelConfig
from mint.nn.blocks import GQAGluBlock
from mint.nn.norm import RMSNorm
from mint.optim.muon_adamw import MuonAdamW, MuonAdamWConfig
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.logger import logger
from torch.utils.checkpoint import checkpoint


@dataclass
class GemmaConfig(ModelConfig):
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


class Gemma(nn.Module):
    def __init__(self, config: GemmaConfig, gradient_checkpointing: bool):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.embed_dim, dtype=config.dtype
        )

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            # Alternate between short-range ('S') and long-range ('L') attention
            attention_type = "S" if i % 2 == 0 else "L"
            print(config)
            self.blocks.append(
                GQAGluBlock(
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    head_dim=config.head_dim,
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    seq_len=config.seq_length,
                    rope_base_theta=config.rope_base_theta,
                    dtype=config.dtype,
                    attention_type=attention_type,
                )
            )

        self.norm = RMSNorm(config.embed_dim)

        self.output = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False, dtype=config.dtype
        )

        # Optionally tie weights between embeddings and output
        if config.tie_weights:
            self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[KVCache]] = None,
        start_pos: int = 0,
        doc_ids: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        h = self.tok_embeddings(input_ids)

        for i, block in enumerate(self.blocks):
            kv_cache: KVCache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and kv_cache is None:
                # Use gradient checkpointing during training (not during generation with kv_cache)
                h = checkpoint(
                    block,
                    x=h,
                    kv_cache=kv_cache,
                    start_pos=start_pos,
                    base_attn_mask=attn_mask,
                    doc_ids=doc_ids,
                    use_reentrant=False,
                )
            else:
                h = block(
                    x=h,
                    kv_cache=kv_cache,
                    start_pos=start_pos,
                    base_attn_mask=attn_mask,
                    doc_ids=doc_ids,
                )

        h = self.norm(h)
        logits = self.output(h)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        kv_caches: Optional[list[KVCache]] = None,
    ) -> torch.Tensor:

        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if kv_caches is not None:
                    # Incremental decoding: only process the last token
                    start_pos = input_ids.shape[1] - 1
                    logits = self.forward(
                        input_ids[:, -1:], kv_caches=kv_caches, start_pos=start_pos
                    )
                else:
                    # prefill
                    logits = self.forward(input_ids)

                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("Inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params


def configure_optimizer(
    model: nn.Module,
    model_cfg: GemmaConfig,
    sched_cfg: SchedulerConfig,
    optim_cfg: MuonAdamWConfig,
) -> MuonAdamW:
    model_dim = model_cfg.embed_dim

    matrix_params = []  # 2D parameters only (for Muon)
    embedding_params = []
    lm_head_params = []
    scalar_params = []  # 1D parameters (norms, biases, etc.)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "tok_embeddings" in name:
            embedding_params.append(param)
            logger.info(f"Embedding: {name} - shape {param.shape}")
        elif "output" in name:
            lm_head_params.append(param)
            logger.info(f"LM Head: {name} - shape {param.shape}")
        elif param.ndim == 2:
            matrix_params.append(param)
            logger.info(f"Matrix: {name} - shape {param.shape}")
        else:
            scalar_params.append(param)
            logger.info(f"Scalar: {name} - shape {param.shape}")

    logger.info(f"Total embedding parameters: {len(embedding_params)}")
    logger.info(f"Total lm_head parameters: {len(lm_head_params)}")
    logger.info(f"Total matrix parameters: {len(matrix_params)}")
    logger.info(f"Total scalar parameters: {len(scalar_params)}")

    # Scale the LR for the AdamW parameters by 1/sqrt(dmodel) (tuned for 768 dim model)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    logger.info(
        f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
    )

    param_groups = []

    if lm_head_params:
        param_groups.append(
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": sched_cfg.unembedding_lr * dmodel_lr_scale,
                "betas": (optim_cfg.adamw_beta1, optim_cfg.adamw_beta2_lm_head),
                "eps": optim_cfg.adamw_eps,
                "weight_decay": sched_cfg.weight_decay_lm_head,
            }
        )

    if embedding_params:
        param_groups.append(
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": sched_cfg.embedding_lr * dmodel_lr_scale,
                "betas": (optim_cfg.adamw_beta1, optim_cfg.adamw_beta2_embedding),
                "eps": optim_cfg.adamw_eps,
                "weight_decay": sched_cfg.weight_decay_embedding,
            }
        )

    if scalar_params:
        param_groups.append(
            {
                "kind": "adamw",
                "params": scalar_params,
                "lr": sched_cfg.scalar_lr * sched_cfg.scalar_lr_multiplier,
                "betas": (optim_cfg.adamw_beta1, optim_cfg.adamw_beta2_scalar),
                "eps": optim_cfg.adamw_eps,
                "weight_decay": sched_cfg.weight_decay_scalar,
            }
        )

    # Muon groups (matrix params only, grouped by shape for stacking)
    shape_groups = {}
    for param in matrix_params:
        shape = param.shape
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append(param)

    for shape in sorted(shape_groups.keys()):
        group_params = shape_groups[shape]
        logger.info(
            f"Matrix params routed to AdamW for stability: shape {shape} with {len(group_params)} parameters"
        )
        param_groups.append(
            {
                "kind": "adamw",
                "params": group_params,
                "lr": sched_cfg.matrix_lr,
                "betas": (optim_cfg.adamw_beta1, optim_cfg.adamw_beta2_scalar),
                "eps": optim_cfg.adamw_eps,
                "weight_decay": sched_cfg.weight_decay,
            }
        )

    optimizer = MuonAdamW(param_groups)

    # Set initial_lr for each group (useful for learning rate schedulers)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer
