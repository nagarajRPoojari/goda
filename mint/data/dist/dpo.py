from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel

from mint.data.dataloader import DataloaderConfig, DistributedDataloader
from mint.tokenizer import Tokenizer
from mint.utils.device import Device


class ContentPart(BaseModel):
    type: Literal["text", "python", "python_output"]
    text: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class Row(BaseModel):
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]


class Datapoint(BaseModel):
    chosen_input_ids: torch.Tensor
    chosen_attn_mask: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attn_mask: torch.Tensor
    rejected_labels: torch.Tensor


class DistributedDPODataloader(DistributedDataloader):
    def __init__(
        self,
        device: Device,
        config: DataloaderConfig,
        tokenizer: Tokenizer,
        filename: str,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(
            device,
            config.data_dir,
            config.batch_size,
            config.seq_length,
            tokenizer,
            *args,
            **kwargs,
        )
        self.filepath = Path(config.data_dir) / Path(filename)
        self.buffer_size = config.buffer_size
        self.tokenizer_batch_size = config.tokenizer_batch_size

        proc_info = device.process_info()
        self.rank = proc_info["rank"]
        self.world_size = proc_info["world_size"]

        assert Path(self.filepath).exists()

    def _document_batches(self) -> Iterator[list[Row]]:
        batch = []

        with Path(self.filepath).open("r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                if idx % self.world_size != self.rank:
                    continue

                clean_line = line.strip()
                if not clean_line:
                    continue

                batch.append(Row.model_validate_json(clean_line))

                if len(batch) == self.tokenizer_batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def _refill_buffer(self, doc_buffer: list, batches: Iterator) -> None:
        doc_batch: list[Row] = next(batches)

        def pad_sequence(sequences, pad_value) -> torch.Tensor:  # noqa: ANN001
            return torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=pad_value
            )

        chosen_ids_list = []
        chosen_attn_mask_list = []
        chosen_labels_list = []

        rejected_ids_list = []
        rejected_attn_mask_list = []
        rejected_labels_list = []

        for row in doc_batch:
            chosen_conv = {"messages": [m.model_dump() for m in (row.prompt + row.chosen)]}
            rejected_conv = {"messages": [m.model_dump() for m in (row.prompt + row.rejected)]}

            c_ids, c_mask = self.tokenizer.render_conversation(
                chosen_conv, max_tokens=self.seq_length
            )
            r_ids, r_mask = self.tokenizer.render_conversation(
                rejected_conv, max_tokens=self.seq_length
            )

            c_labels = [tid if m == 1 else -100 for tid, m in zip(c_ids, c_mask, strict=False)]
            r_labels = [tid if m == 1 else -100 for tid, m in zip(r_ids, r_mask, strict=False)]

            chosen_ids_list.append(torch.tensor(c_ids, dtype=torch.long))
            chosen_attn_mask_list.append(torch.tensor([1] * len(c_ids), dtype=torch.long))
            chosen_labels_list.append(torch.tensor(c_labels, dtype=torch.long))

            rejected_ids_list.append(torch.tensor(r_ids, dtype=torch.long))
            rejected_attn_mask_list.append(torch.tensor([1] * len(r_ids), dtype=torch.long))
            rejected_labels_list.append(torch.tensor(r_labels, dtype=torch.long))

        chosen_input_ids = pad_sequence(chosen_ids_list, self.tokenizer.pad_token)
        chosen_attn_mask = pad_sequence(chosen_attn_mask_list, 0)
        chosen_labels = pad_sequence(chosen_labels_list, -100)

        rejected_input_ids = pad_sequence(rejected_ids_list, self.tokenizer.pad_token)
        rejected_attn_mask = pad_sequence(rejected_attn_mask_list, 0)
        rejected_labels = pad_sequence(rejected_labels_list, -100)

        doc_buffer.extend(
            [
                Datapoint(
                    chosen_input_ids=chosen_input_ids[i],
                    chosen_attn_mask=chosen_attn_mask[i],
                    chosen_labels=chosen_labels[i],
                    rejected_input_ids=rejected_input_ids[i],
                    rejected_attn_mask=rejected_attn_mask[i],
                    rejected_labels=rejected_labels[i],
                )
                for i in range(len(doc_batch))
            ]
        )

    def batch_loader(self, split="train", resume_state=None) -> Iterator[list[Datapoint]]:  # noqa: ANN001, ARG002
        doc_buffer = []
        batches = self._document_batches()

        while True:
            while len(doc_buffer) < self.B:
                try:
                    self._refill_buffer(doc_buffer=doc_buffer, batches=batches)
                except StopIteration:
                    break

            if len(doc_buffer) < self.B:
                break

            yield doc_buffer[: self.B]
            doc_buffer = doc_buffer[self.B :]

    def get_state(self):  # noqa: ANN201
        return NotImplementedError()  # TODO: implement resume state for DPO

    def set_state(self, state):  # noqa: ANN001, ANN201, ARG002
        return NotImplementedError()
