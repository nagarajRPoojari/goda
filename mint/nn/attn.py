
import torch
import torch.nn as nn
from typing import Optional

from goda.config import Config
from mint.nn.flashattn import FlashAttention
from mint.nn.norm import RMSNorm
from mint.kvcache.base import KVCache

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: Config, window_size: int = -1):
        super().__init__()

        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = window_size
        self.attention = FlashAttention(use_custom_fa=True)
        
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
