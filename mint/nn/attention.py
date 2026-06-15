import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mint.kvcache.base import KVCache

class FlashAttention(nn.Module):
    def __init__(self, use_custom_fa: bool = True):
        super().__init__()
        self.use_custom_fa = use_custom_fa
        self._custom_fa: Any | None = self._load_custom_flash_attention() if use_custom_fa else None
        self._fa3: Any | None = self._load_flash_attention_3()

    def forward( self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, window_size: tuple[int, int] = (-1, -1), kv_cache: KVCache | None = None) -> torch.Tensor:
        if kv_cache is None:
            return self._attention(q, k, v, causal=causal, window_size=window_size)
        
        return self._attention_with_kvcache( q=q, k=k, v=v, kv_cache=kv_cache, causal=causal, window_size=window_size)

    def _load_custom_flash_attention(self):
        if not torch.cuda.is_available():
            return None
        try:
            from kernels.flash_attn import FlashAttentionKernel  # pyright: ignore[reportMissingImports]
            return FlashAttentionKernel
        except Exception:
            return None

    def _load_flash_attention_3(self):
        if not torch.cuda.is_available():
            return None
        try:
            major, _ = torch.cuda.get_device_capability()
            if major != 9:
                return None
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            from kernels import get_kernel  # pyright: ignore[reportMissingImports]

            return get_kernel("varunneal/flash-attention-3").flash_attn_interface
        except Exception:
            return None
            

    def _use_custom_fa(self, q: torch.Tensor) -> bool:
        return self._custom_fa is not None and q.is_cuda and q.dtype in [torch.float16, torch.bfloat16]

    def _use_fa3(self, q: torch.Tensor) -> bool:
        return self._fa3 is not None and q.is_cuda and q.dtype == torch.bfloat16

    def _attention(self, q: torch.Tensor ,k: torch.Tensor, v: torch.Tensor, causal: bool, window_size: tuple[int, int]) -> torch.Tensor:
        # Try custom flash attention first if enabled
        if self.use_custom_fa and self._use_custom_fa(q) and window_size == (-1, -1):
            custom_fa = self._custom_fa
            assert custom_fa is not None
            # Calculate softmax scale (tau)
            head_dim = q.shape[-1]
            tau = 1.0 / (head_dim ** 0.5)
            return custom_fa.apply(q, k, v, causal, tau)
        
        # Fall back to FA3 if available
        if self._use_fa3(q):
            fa3 = self._fa3
            assert fa3 is not None
            return fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

        # Fall back to SDPA
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
        out = self._sdpa_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            causal=causal,
            window_size=window_size,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)

    def _attention_with_kvcache( self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_cache: KVCache, causal: bool, window_size: tuple[int, int]) -> torch.Tensor:
        k_cache, v_cache, cache_seqlens = kv_cache.get_cache_tensors()

        if self._use_fa3(q):
            fa3 = self._fa3
            assert fa3 is not None
            return fa3.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                causal=causal,
                window_size=window_size,
            )

        k_full, v_full = kv_cache.update(k, v, start_pos=int(cache_seqlens[0].item()))

        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

        out = self._sdpa_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            causal=causal,
            window_size=window_size,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)

    def _sdpa_attention( self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, window_size: tuple[int, int], enable_gqa: bool) -> torch.Tensor:
        if not causal:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)

        q_len = q.size(2)
        k_len = k.size(2)
        left_window = window_size[0]

        if (left_window < 0 or left_window >= q_len) and q_len == k_len:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=enable_gqa,
            )

        if q_len == 1:
            if 0 <= left_window < k_len:
                start = max(0, k_len - (left_window + 1))
                k = k[:, :, start:, :]
                v = v[:, :, start:, :]
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)

        device = q.device
        row_idx = (k_len - q_len) + torch.arange(q_len, device=device).unsqueeze(1)
        col_idx = torch.arange(k_len, device=device).unsqueeze(0)
        mask = col_idx <= row_idx

        if 0 <= left_window < k_len:
            mask = mask & ((row_idx - col_idx) <= left_window)

        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

