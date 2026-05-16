import torch
import time
from abc import ABC
from pathlib import Path
from typing import Iterator, Generator, Tuple, List, Union
from goda.device import Device
from goda.sft.base import SFTDataset, SFTTrainDataset
from goda.tokenizer import Tokenizer
from goda.config import Config
import pyarrow.parquet as pq
from goda.logger import logger

class DistributedDataloader(ABC):
    def __init__(self, device: Device, data_dir: str,  batch_size: int, seq_len: int, tokenizer: Tokenizer) -> None:
        self.device = device
        self.data_dir = Path(data_dir)
        self.B = batch_size
        self.T = seq_len
        self.tokenizer = tokenizer
    def batch_loader(self, split: str = "train", resume_state: dict | None = None) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        ...
    
    def get_state(self) -> dict:
        return {}
    
    def set_state(self, state: dict) -> None:
        pass

class DistributedPretrainDataloader(DistributedDataloader):
    def __init__(self, device: Device, config: Config, tokenizer: Tokenizer,
                 min_shards_required: int = 2, max_shards_to_wait: int = -1,
                 shard_check_interval: float = 5.0) -> None:
        
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
        
        # Dynamic shard discovery
        self.min_shards_required = min_shards_required
        self.max_shards_to_wait = max_shards_to_wait  # -1 means wait indefinitely
        self.shard_check_interval = shard_check_interval
        self.train_shards = []
        self.val_shards = []
        self._refresh_shard_lists()
        
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
        self.last_shard_refresh = time.time()
        

    def _refresh_shard_lists(self) -> None:
        self.train_shards = sorted((self.data_dir / "train").glob("*.parquet"))
        self.val_shards = sorted((self.data_dir / "val").glob("*.parquet"))
    
    def _wait_for_shards(self, split: str, required_count: int) -> None:
        """
        Wait until minimum number of shards are available.
        If max_shards_to_wait is set and reached, stop waiting even if minimum not met.
        """
        shards = self.train_shards if split == "train" else self.val_shards
        
        while len(shards) < required_count:
            # Check if we've hit the upper limit
            if self.max_shards_to_wait > 0 and len(shards) >= self.max_shards_to_wait:
                if self.rank == 0:
                    logger.info(f"✓ Reached max shard limit ({self.max_shards_to_wait}). Proceeding with {len(shards)} shards.")
                return
            
            if self.rank == 0:
                wait_msg = f"Waiting for shards... ({len(shards)}/{required_count} available"
                if self.max_shards_to_wait > 0:
                    wait_msg += f", max: {self.max_shards_to_wait}"
                wait_msg += ")"
                logger.info(wait_msg)
            
            time.sleep(self.shard_check_interval)
            self._refresh_shard_lists()
            shards = self.train_shards if split == "train" else self.val_shards
        
        if self.rank == 0:
            logger.info(f"✓ Sufficient shards available: {len(shards)}")
    
    def _should_refresh_shards(self) -> bool:
        current_time = time.time()
        if current_time - self.last_shard_refresh > self.shard_check_interval:
            self.last_shard_refresh = current_time
            return True
        return False

    def _document_batches(self, split: str, start_shard_idx: int = 0, start_rg_idx: int = -1) -> Iterator[list]:
        self._wait_for_shards(split, self.min_shards_required)
        
        shards = self.train_shards if split == "train" else self.val_shards
        shard_cycle_start = start_shard_idx % len(shards) if shards else 0
        
        while True:
            # Periodically refresh shard list to discover new downloads
            if self._should_refresh_shards():
                old_count = len(shards)
                self._refresh_shard_lists()
                shards = self.train_shards if split == "train" else self.val_shards
                if len(shards) > old_count and self.rank == 0:
                    logger.info(f"Discovered {len(shards) - old_count} new shard(s). Total: {len(shards)}")
            
            # If no shards available, check if we should wait or stop
            if not shards:
                # If max_shards_to_wait is set and we've reached it, stop waiting
                if self.max_shards_to_wait > 0:
                    if self.rank == 0:
                        logger.info(f"No shards available and max limit ({self.max_shards_to_wait}) reached. Stopping.")
                    break
                
                if self.rank == 0:
                    logger.info("No shards available, waiting...")
                time.sleep(self.shard_check_interval)
                self._refresh_shard_lists()
                shards = self.train_shards if split == "train" else self.val_shards
                continue
            
            for shard_offset in range(len(shards)):
                shard_idx = (shard_cycle_start + shard_offset) % len(shards)
                shard_path = shards[shard_idx]
                
                # Check if shard still exists (in case of cleanup)
                if not shard_path.exists():
                    continue
                
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


class DistributedSFTDataloader(DistributedDataloader):
    def __init__(self, device: Device, config: Config, tokenizer: Tokenizer, datasets: list[SFTDataset], shuffle: bool = True) -> None:
        super().__init__(
            device=device,
            data_dir="",
            batch_size=config.batch_size,
            seq_len=config.seq_length,
            tokenizer=tokenizer
        )
        
        self.datasets = datasets
        self.max_tokens = config.seq_length
        self.shuffle = shuffle
        
        proc_info = device.process_info()
        self.rank = proc_info["rank"]
        self.world_size = proc_info["world_size"]
        
        self.dataset_indices = self._build_dataset_indices()
        self.rank_indices = self._partition_for_rank()
        
        use_cuda = device.is_cuda
        self.cpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, pin_memory=use_cuda)
        self.gpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, device=device.device)
        
        self.cpu_inputs = self.cpu_buffer[:self.B * self.T].view(self.B, self.T)
        self.cpu_targets = self.cpu_buffer[self.B * self.T:].view(self.B, self.T)
        self.cpu_mask = torch.zeros(self.B, self.T, dtype=torch.long, pin_memory=use_cuda)
        
        self.inputs = self.gpu_buffer[:self.B * self.T].view(self.B, self.T)
        self.targets = self.gpu_buffer[self.B * self.T:].view(self.B, self.T)
        self.mask = torch.zeros(self.B, self.T, dtype=torch.long, device=device.device)
    
    def _build_dataset_indices(self) -> list:
        import random
        indices = []
        for ds_idx, ds in enumerate(self.datasets):
            indices.extend([(ds_idx, i) for i in range(len(ds))])
        if self.shuffle:
            random.Random(42).shuffle(indices)
        return indices
    
    def _partition_for_rank(self) -> list:
        return [idx for i, idx in enumerate(self.dataset_indices) if i % self.world_size == self.rank]
    
    def batch_loader(self, split: str = "train", resume_state: dict | None = None) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        idx = 0
        total = len(self.rank_indices)
        
        while True:
            for batch_idx in range(self.B):
                ds_idx, example_idx = self.rank_indices[idx % total]
                idx += 1
                
                conversation = self.datasets[ds_idx][example_idx]
                ids, mask = self.tokenizer.render_conversation(conversation, self.max_tokens)
                
                seq_len = len(ids)
                if seq_len <= self.T:
                    ids = ids + [self.tokenizer.pad_token] * (self.T + 1 - seq_len)
                    mask = mask + [0] * (self.T + 1 - seq_len)
                else:
                    ids = ids[:self.T + 1]
                    mask = mask[:self.T + 1]
                
                self.cpu_inputs[batch_idx] = torch.tensor(ids[:-1], dtype=torch.long)
                self.cpu_targets[batch_idx] = torch.tensor(ids[1:], dtype=torch.long)
                self.cpu_mask[batch_idx] = torch.tensor(mask[1:], dtype=torch.long)
            
            self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=self.device.is_cuda)
            self.mask.copy_(self.cpu_mask, non_blocking=self.device.is_cuda)
            
            yield self.inputs, self.targets, self.mask