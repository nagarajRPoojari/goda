import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple
from mint.utils.device import Device
from mint.tokenizer import Tokenizer
from mint.data.sampler import Sampler

class DistributedDataloader(Sampler, ABC):
    def __init__(self, device: Device, data_dir: str,  batch_size: int, seq_len: int, tokenizer: Tokenizer) -> None:
        self.device = device
        self.data_dir = Path(data_dir)
        self.B = batch_size
        self.T = seq_len
        self.tokenizer = tokenizer

    @abstractmethod
    def batch_loader(self, split: str = "train", resume_state: dict | None = None) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self) -> dict:
        ...
    
    @abstractmethod
    def set_state(self, state: dict) -> None:
        ...


    