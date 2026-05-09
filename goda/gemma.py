import torch
import torch.nn as nn
from typing import Optional

from goda.config import Config
from goda.attention import FlashAttention
from goda.kvcache import KVCache
from goda.optim import MuonAdamW
from goda.logger import logger

class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.embed_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.embed_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.embed_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
         
        if self.shift is not None:
            out = out + self.shift.float()
         
        return out.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: Config, window_size: int = -1):
        super().__init__()

        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = window_size
        self.attention = FlashAttention()
        
        self.wq = nn.Linear(cfg.embed_dim, self.n_heads * self.head_dim, bias=False, dtype=cfg.dtype)
        self.wk = nn.Linear(cfg.embed_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=cfg.dtype)
        self.wv = nn.Linear(cfg.embed_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=cfg.dtype)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.embed_dim, bias=False, dtype=cfg.dtype)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        cos, sin = self._precompute_rotary_embeddings(
            cfg.seq_length, self.head_dim, cfg.rope_base_theta
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        
    def forward(self, x, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        cos = self.cos[:, start_pos:start_pos + seqlen, :, :].to(x.dtype)
        sin = self.sin[:, start_pos:start_pos + seqlen, :, :].to(x.dtype)
        q = self.apply_rotary_embs(q, cos, sin)
        k = self.apply_rotary_embs(k, cos, sin)
        
        window_size = (self.window_size, -1) if self.window_size > 0 else (-1, -1)
        out = self.attention(q, k, v, causal=True, window_size=window_size, kv_cache=kv_cache)
        
        out = out.reshape(bsz, seqlen, -1)
        return self.wo(out)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None, dtype: torch.dtype = torch.float32):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)

        freqs = torch.outer(t, inv_freq)

        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype), sin.to(dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def apply_rotary_embs(self, x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1)

class Block(nn.Module):
    def __init__(self, config: Config, attention_type: str = 'L') -> None:
        super().__init__()
        self.config = config
        self.attention_type = attention_type  # 'S' for short, 'L' for long

        self.pre_attn_norm = RMSNorm(config.embed_dim)
        self.post_attn_norm = RMSNorm(config.embed_dim)
        self.pre_ffn_norm = RMSNorm(config.embed_dim)
        self.post_ffn_norm = RMSNorm(config.embed_dim)

        window_size = config.seq_length // 4 if attention_type == 'S' else -1
        self.gqa = GroupedQueryAttention(config, window_size=window_size)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, start_pos: int = 0) -> torch.Tensor:
        h = self.pre_attn_norm(x)
        h = x + self.post_attn_norm(self.gqa(h, kv_cache=kv_cache, start_pos=start_pos))
        h_ffn = self.pre_ffn_norm(h)
        h = h + self.post_ffn_norm(self.ffn(h_ffn))
        return h


class Gemma(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            dtype=config.dtype
        )
        
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            # Alternate between short-range ('S') and long-range ('L') attention
            attention_type = 'S' if i % 2 == 0 else 'L'
            self.blocks.append(Block(config, attention_type=attention_type))
        
        self.norm = RMSNorm(config.embed_dim)
        
        self.output = nn.Linear(
            config.embed_dim,
            config.vocab_size,
            bias=False,
            dtype=config.dtype
        )
        
        # Optionally tie weights between embeddings and output
        if config.tie_weights:
            self.output.weight = self.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, kv_caches: Optional[list[KVCache]] = None, start_pos: int = 0) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            h = block(h, kv_cache=kv_cache, start_pos=start_pos)
        
        h = self.norm(h)
        logits = self.output(h)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: 
                Optional[int] = None, top_p: Optional[float] = None, kv_caches: Optional[list[KVCache]] = None) -> torch.Tensor:

        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if kv_caches is not None:
                    # Incremental decoding: only process the last token
                    start_pos = input_ids.shape[1] - 1
                    logits = self.forward(
                        input_ids[:, -1:],
                        kv_caches=kv_caches,
                        start_pos=start_pos
                    )
                else:
                    # prefill
                    logits = self.forward(input_ids)
                
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params


def configure_optimizer(model: nn.Module, config: Config) -> MuonAdamW:
    model_dim = config.embed_dim
    
    matrix_params = []  # 2D parameters only (for Muon)
    embedding_params = []
    lm_head_params = []
    scalar_params = []  # 1D parameters (norms, biases, etc.)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'tok_embeddings' in name:
            embedding_params.append(param)
            logger.info(f"Embedding: {name} - shape {param.shape}")
        elif 'output' in name:
            lm_head_params.append(param)
            logger.info(f"LM Head: {name} - shape {param.shape}")
        elif param.ndim == 2:
            matrix_params.append(param)
            logger.info(f"Matrix: {name} - shape {param.shape}")
        else:
            scalar_params.append(param)
            logger.info(f"Scalar: {name} - shape {param.shape}")
    
    logger.info(f"\nTotal embedding parameters: {len(embedding_params)}")
    logger.info(f"Total lm_head parameters: {len(lm_head_params)}")
    logger.info(f"Total matrix parameters: {len(matrix_params)}")
    logger.info(f"Total scalar parameters: {len(scalar_params)}")
    
    # Scale the LR for the AdamW parameters by 1/sqrt(dmodel) (tuned for 768 dim model)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    logger.info(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
    
    param_groups = []
    
    if lm_head_params:
        param_groups.append({
            'kind': 'adamw',
            'params': lm_head_params,
            'lr': config.unembedding_lr * dmodel_lr_scale,
            'betas': (config.adamw_beta1, config.adamw_beta2_lm_head),
            'eps': config.adamw_eps,
            'weight_decay': config.weight_decay_lm_head,
        })
    
    if embedding_params:
        param_groups.append({
            'kind': 'adamw',
            'params': embedding_params,
            'lr': config.embedding_lr * dmodel_lr_scale,
            'betas': (config.adamw_beta1, config.adamw_beta2_embedding),
            'eps': config.adamw_eps,
            'weight_decay': config.weight_decay_embedding,
        })
    
    if scalar_params:
        param_groups.append({
            'kind': 'adamw',
            'params': scalar_params,
            'lr': config.scalar_lr * config.scalar_lr_multiplier,
            'betas': (config.adamw_beta1, config.adamw_beta2_scalar),
            'eps': config.adamw_eps,
            'weight_decay': config.weight_decay_scalar,
        })
    
    # Muon groups (matrix params only, grouped by shape for stacking)
    shape_groups = {}
    for param in matrix_params:
        shape = param.shape
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append(param)
    
    for shape in sorted(shape_groups.keys()):
        group_params = shape_groups[shape]
        logger.info(f"Muon group: shape {shape} with {len(group_params)} parameters")
        param_groups.append({
            'kind': 'muon',
            'params': group_params,
            'lr': config.matrix_lr,
            'momentum': config.muon_momentum,
            'ns_steps': config.muon_ns_steps,
            'beta2': config.muon_beta2,
            'weight_decay': config.weight_decay,
        })
    
    optimizer = MuonAdamW(param_groups)
    
    # Set initial_lr for each group (useful for learning rate schedulers)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    
    return optimizer