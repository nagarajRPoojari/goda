import sys

import torch


use_cuda = (sys.platform in ("win32", "linux", "linux2")) and torch.cuda.is_available()

if use_cuda:
    from .flash_attn_gqa.cuda import FlashAttentionGQAKernel
    from .flash_attn_mqa.cuda import FlashAttentionMQAKernel
else:
    from .flash_attn_gqa.nil import FlashAttentionGQAKernel
    from .flash_attn_mqa.nil import FlashAttentionMQAKernel

__all__ = ["FlashAttentionGQAKernel", "FlashAttentionMQAKernel"]
