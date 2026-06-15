from typing import List, Optional

import torch.nn as nn

from .base import Adapter


class QLoRALinear(nn.Module):
    def __init__(
        self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05
    ): ...

    def forward(self, x): ...


class QLoRA(Adapter):
    def __init__(self):
        super().__init__()

    def apply(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        r: int = 8,
        alpha: int = 16,
    ): ...
