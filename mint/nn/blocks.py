

import torch
import torch.nn as nn
from typing import Optional

from goda.config import Config
from mint.nn.norm import RMSNorm
from mint.nn.attn import GroupedQueryAttention
from mint.nn.ffn import GLU
from mint.kvcache.base import KVCache

class GQAGluBlock(nn.Module):
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
        self.ffn = GLU(config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, start_pos: int = 0) -> torch.Tensor:
        h = self.pre_attn_norm(x)
        h = x + self.post_attn_norm(self.gqa(h, kv_cache=kv_cache, start_pos=start_pos))
        h_ffn = self.pre_ffn_norm(h)
        h = h + self.post_ffn_norm(self.ffn(h_ffn))
        return h
