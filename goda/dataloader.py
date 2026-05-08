

from abc import ABC
from pathlib import Path
from typing import Iterator, Generator, Tuple, List, Union
from goda.device import Device
from goda.tokenizer import Tokenizer
from goda.config import Config
import pyarrow.parquet as pq
import torch


class DistributedDataloader(ABC):
    def __init__(self, device: Device, data_dir: str,  batch_size: int, seq_len: int, tokenizer: Tokenizer) -> None:
        self.device = device
        self.data_dir = Path(data_dir)
        self.B = batch_size
        self.T = seq_len
        self.tokenizer = tokenizer

    def batch_loader(self, split: str = "train", resume_state: dict | None = None) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        ...
    
    def get_state(self) -> dict:
        return {}
    
    def set_state(self, state: dict) -> None:
        pass


class DistributedPretrainDataloader(DistributedDataloader):
    def __init__(self, device: Device, config: Config, tokenizer: Tokenizer) -> None:
        
        super().__init__(
            device=device,
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seq_len=config.seq_length,
            tokenizer=tokenizer
        )

        self.buffer_size = config.buffer_size
        self.tokenizer_batch_size = config.tokenizer_batch_size
        
        proc_info = device.process_info()
        self.rank = proc_info["rank"]
        self.world_size = proc_info["world_size"]
        
        self.train_shards = sorted((self.data_dir / "train").glob("*.parquet"))
        self.val_shards = sorted((self.data_dir / "val").glob("*.parquet"))
        
        self.row_capacity = self.T + 1
        use_cuda: bool = device.is_cuda
        
        self.row_buffer = torch.empty((self.B, self.row_capacity), dtype=torch.long)
        self.cpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, pin_memory=use_cuda)
        self.gpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, device=device.device)
        
        self.cpu_inputs = self.cpu_buffer[:self.B * self.T].view(self.B, self.T)
        self.cpu_targets = self.cpu_buffer[self.B * self.T:].view(self.B, self.T)

        self.inputs = self.gpu_buffer[:self.B * self.T].view(self.B, self.T)
        self.targets = self.gpu_buffer[self.B * self.T:].view(self.B, self.T)
        
        self.current_shard_idx = 0
        self.current_rg_idx = self.rank
        self.batches_consumed = 0
        

    def _document_batches(self, split: str, start_shard_idx: int = 0, start_rg_idx: int = -1) -> Iterator[list]:
        shards = self.train_shards if split == "train" else self.val_shards
        
        shard_cycle_start = start_shard_idx % len(shards) if shards else 0
        
        while True:
            for shard_offset in range(len(shards)):
                shard_idx = (shard_cycle_start + shard_offset) % len(shards)
                shard_path = shards[shard_idx]
                pf = pq.ParquetFile(shard_path)
                
                # Determine starting row group index
                if shard_offset == 0 and start_rg_idx >= 0:
                    rg_idx = start_rg_idx
                else:
                    rg_idx = self.rank
                
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    
                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        yield batch[i:i + self.tokenizer_batch_size]
                        self.batches_consumed += 1
                    
                    rg_idx += self.world_size
                    self.current_rg_idx = rg_idx
                
                self.current_shard_idx = (shard_idx + 1) % len(shards)
    
    def _refill_buffer(self, doc_buffer: List, batches: Iterator) -> None:
        doc_batch = next(batches)
        token_lists = self.tokenizer.encode_to_list(doc_batch, add_bos=True, add_eos=False, padding=False)
        doc_buffer.extend(token_lists)
    
    def batch_loader(self, split: str = "train", resume_state: dict | None = None) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        # Restore state if resuming
        if resume_state and split == "train":
            start_shard_idx = resume_state.get('shard_idx', 0)
            start_rg_idx = resume_state.get('rg_idx', self.rank)
            self.batches_consumed = resume_state.get('batches_consumed', 0)
        else:
            start_shard_idx = 0
            start_rg_idx = self.rank
            self.batches_consumed = 0
        
        batches = self._document_batches(split, start_shard_idx, start_rg_idx)
        doc_buffer = []
        
        while True:
            for row_idx in range(self.B):
                pos = 0
                
                while pos < self.row_capacity:
                    while len(doc_buffer) < self.buffer_size:
                        self._refill_buffer(doc_buffer, batches)
                    
                    remaining = self.row_capacity - pos
                    
                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len
                    
                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        doc_len = len(doc)
                        self.row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                        pos += doc_len
                    else:
                        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(shortest_idx)
                        self.row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                        pos += remaining
            
            self.cpu_inputs.copy_(self.row_buffer[:, :-1])
            self.cpu_targets.copy_(self.row_buffer[:, 1:])
            
            self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=self.device.is_cuda)
            
            yield self.inputs, self.targets
    
    def get_state(self) -> dict:
        return {
            'shard_idx': self.current_shard_idx,
            'rg_idx': self.current_rg_idx,
            'batches_consumed': self.batches_consumed,
        }
    
    def set_state(self, state: dict) -> None:
        self.current_shard_idx = state.get('shard_idx', 0)
        self.current_rg_idx = state.get('rg_idx', self.rank)
        self.batches_consumed = state.get('batches_consumed', 0)

