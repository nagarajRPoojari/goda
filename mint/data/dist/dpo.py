import json
import os
from pathlib import Path

from mint.data.dataloader import DistributedDataloader
class DistributedDPODataloader(DistributedDataloader):
    def __init__(self, 
                 device, 
                 data_dir, 
                 batch_size, 
                 seq_len, 
                 tokenizer, 
                 filename, 
                 *args, 
                 **kwargs):
        super().__init__(device, data_dir, batch_size, seq_len, tokenizer, args, kwargs)
        self.filepath = Path(data_dir) / Path(filename)
        
        assert os.path.exists(self.filepath)

    def batch_loader(self, split = "train", resume_state = None):
        ...

    def get_state(self):
        return super().get_state()

    def set_state(self, state):
        return super().set_state(state)
    