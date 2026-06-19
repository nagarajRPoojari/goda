from typing import Optional

import torch
import torch.nn as nn

from mint.kvcache.base import KVCache
from mint.nn.flashattn import FlashAttention
from mint.nn.norm import RMSNorm
from mint.nn.pos import PosEmbeddings


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        pos_embeddings: PosEmbeddings,
        dtype: torch.dtype,
        window_size: int = -1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.pos_embeddings = pos_embeddings

        self.attention = FlashAttention(use_custom_fa=True)

        self.wq = nn.Linear(
            embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wk = nn.Linear(
            embed_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wv = nn.Linear(
            embed_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, embed_dim, bias=False, dtype=dtype
        )

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, attn_bias = self.pos_embeddings(q, k, start_pos=start_pos)

        assert attn_bias is None, "attention bias is not supported yet"

        window_size = (self.window_size, -1) if self.window_size > 0 else (-1, -1)
        out = self.attention(
            q, k, v, causal=True, window_size=window_size, kv_cache=kv_cache
        )

        out = out.reshape(bsz, seqlen, -1)
        return self.wo(out)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        embed_dim: int,
        pos_embeddings: PosEmbeddings,
        dtype: torch.dtype,
        window_size: int = -1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.pos_embeddings = pos_embeddings

        self.attention = FlashAttention(use_custom_fa=True)

        self.wq = nn.Linear(
            embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wk = nn.Linear(
            embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wv = nn.Linear(
            embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, embed_dim, bias=False, dtype=dtype
        )

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, attn_bias = self.pos_embeddings(q, k, start_pos=start_pos)

        assert attn_bias is None, "attention bias is not supported yet"

        window_size = (self.window_size, -1) if self.window_size > 0 else (-1, -1)
        out = self.attention(
            q, k, v, causal=True, window_size=window_size, kv_cache=kv_cache
        )

        out = out.reshape(bsz, seqlen, -1)
        return self.wo(out)
