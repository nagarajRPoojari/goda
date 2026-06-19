from mint.data.dataloader import DistributedDataloader


class DistributedDPODataloader(DistributedDataloader):

    def __init__(self, device, data_dir, batch_size, seq_len, tokenizer):
        super().__init__(device, data_dir, batch_size, seq_len, tokenizer)

    def batch_loader(self, split = "train", resume_state = None):
        ...

    def get_state(self):
        return super().get_state()

    def set_state(self, state):
        return super().set_state(state)
    