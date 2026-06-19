import torch


class KVCache:
    def __init__(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
        n_kv_heads: int = 4,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.k_cache = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        self.cache_seqlens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

    def update(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_kv_heads, head_dim = k.shape  # B, T, H, D

        self.k_cache[:batch_size, start_pos : start_pos + seq_len, :, :] = k
        self.v_cache[:batch_size, start_pos : start_pos + seq_len, :, :] = v

        self.cache_seqlens[:batch_size] = start_pos + seq_len

        # Return full cache up to current position
        end_pos = start_pos + seq_len
        full_k = self.k_cache[:batch_size, :end_pos, :, :]
        full_v = self.v_cache[:batch_size, :end_pos, :, :]

        return full_k, full_v

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.cache_seqlens[0].item()
        return (self.k_cache[:, :pos, :, :], self.v_cache[:, :pos, :, :])

    def get_cache_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k_cache, self.v_cache, self.cache_seqlens

    def reset(self):
        self.cache_seqlens.zero_()

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_seqlens.zero_()
