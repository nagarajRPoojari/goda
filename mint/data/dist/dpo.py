from pathlib import Path
from typing import Any

import torch

from mint.data.dataloader import DistributedDataloader


class DistributedDPODataloader(DistributedDataloader):
    def __init__(
        self,
        device: torch.device,
        data_dir: str,
        batch_size: int,
        seq_len: int,
        tokenizer: Any,  # noqa: ANN401
        filename: str,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(device, data_dir, batch_size, seq_len, tokenizer, args, kwargs)
        self.filepath = Path(data_dir) / Path(filename)

        assert Path.exists(self.filepath)

    def batch_loader(self, split="train", resume_state=None): ...

    def get_state(self):
        return super().get_state()

    def set_state(self, state):
        return super().set_state(state)
