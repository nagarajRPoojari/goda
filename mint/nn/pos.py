import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class PosEmbeddings(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...


class SinusoidalPosEmbeddings(PosEmbeddings):
    def __init__(self, seq_len: int, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim

        pe = torch.zeros(seq_len, head_dim)

        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, head_dim, 2, dtype=torch.float32)
            * -(math.log(base) / head_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe[None, :, None, :]

        self.register_buffer("pe", pe, persistent=False)
        self.pe: torch.Tensor

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        seqlen = q.shape[1]

        pe_slice = self.pe[:, start_pos : start_pos + seqlen, :, :].to(q.dtype)

        q_out = q + pe_slice
        k_out = k + pe_slice

        return q_out, k_out, None


class RoPE(PosEmbeddings):
    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        rope_base_theta: int = 100000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        cos, sin = self._precompute_rotary_embeddings(
            seq_len, head_dim, rope_base_theta, dtype=dtype
        )
        # Registering as buffers ensures they are moved to the correct device
        # alongside the model but are not treated as trainable parameters.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cos: torch.Tensor
        self.sin: torch.Tensor

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype), sin.to(dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _apply_rotary_embs(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seqlen = q.shape[1]

        # Slice the precomputed embeddings to match the current sequence length
        cos = self.cos[:, start_pos : start_pos + seqlen, :, :].to(q.dtype)
        sin = self.sin[:, start_pos : start_pos + seqlen, :, :].to(q.dtype)

        q_out = self._apply_rotary_embs(q, cos, sin)
        k_out = self._apply_rotary_embs(k, cos, sin)

        return q_out, k_out, None


class LinearScaledRoPE(RoPE):
    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        scale_factor: float = 4.0,
        rope_base_theta: int = 100000,
        dtype: torch.dtype = torch.float32,
    ):
        self.scale_factor = scale_factor
        super().__init__(seq_len, head_dim, rope_base_theta, dtype)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # scale the absolute pos index by a scaling factor
        # why ?: imagine initially model is trained on seq_len=512 with RoPE, now
        # i want to train for 4k, model struggles to understand fequencies of pos beyond 512.
        # we can scale down those pos to smaller interpolated values
        t = (
            torch.arange(seq_len, dtype=torch.float32, device=device)
            / self.scale_factor
        )
        freqs = torch.outer(t, inv_freq)

        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype), sin.to(dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin


class NoPE(PosEmbeddings):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k, None


class ALiBiPositionalBias(PosEmbeddings):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        slopes = torch.tensor(self._get_slopes(n_heads))

        # Shape: [1, n_heads, 1, 1]
        self.register_buffer("slopes", slopes[None, :, None, None], persistent=False)

    def _get_slopes(self, n_heads: int):
        def get_slopes_power_of_2(n_heads):
            start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n_heads)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        # should handle non-power-of-2 head counts gracefully
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self._get_slopes(2 * closest_power_of_2)[0::2][
                : n_heads - closest_power_of_2
            ]
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        seq_len_q = q.shape[1]

        total_kv_len = start_pos + seq_len_q

        # query positions on the global timeline: [seq_len_q, 1]
        q_idx = torch.arange(start_pos, start_pos + seq_len_q, device=q.device)[:, None]

        # key positions spanning from the beginning of time: [1, total_kv_len]
        k_idx = torch.arange(0, total_kv_len, device=k.device)[None, :]

        # causal relative distance calculation: [seq_len_q, total_kv_len]
        # In causal sequences, keys occur before or at the query index (k_idx <= q_idx).
        # (k_idx - q_idx) naturally yields negative values or zero, perfect for a penalty.
        relative_position = k_idx - q_idx

        # multiply positive head-slopes by negative distances: [1, n_heads, seq_len_q, total_kv_len]
        alibi_bias = self.slopes * relative_position[None, None, :, :]

        return q, k, alibi_bias.to(q.dtype)
