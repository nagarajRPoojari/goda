import copy
from typing import Any

import tiktoken
import torch


class Tokenizer:
    def __init__(self, model_name: str = "gpt-2") -> None:
        self.encoding = tiktoken.encoding_for_model(model_name=model_name)
        self.bos_token = self.encoding.max_token_value + 1
        self.eos_token = self.encoding.eot_token
        self.pad_token = self.encoding.max_token_value + 2

        self.special_tokens = {
            "<|user_start|>": self.encoding.max_token_value + 3,
            "<|user_end|>": self.encoding.max_token_value + 4,
            "<|assistant_start|>": self.encoding.max_token_value + 5,
            "<|assistant_end|>": self.encoding.max_token_value + 6,
            "<|python_start|>": self.encoding.max_token_value + 7,
            "<|python_end|>": self.encoding.max_token_value + 8,
            "<|output_start|>": self.encoding.max_token_value + 9,
            "<|output_end|>": self.encoding.max_token_value + 10,
        }

    def encode(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> torch.Tensor:
        return torch.tensor(
            self.encode_to_list(
                batch=batch,
                add_bos=add_bos,
                add_eos=add_eos,
                padding=padding,
                max_length=max_length,
            ),
            dtype=torch.long,
        )

    def encode_to_list(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> list[Any]:
        encoded = self.encoding.encode_batch(text=batch, disallowed_special=())

        processed = []
        for enc in encoded:
            tokens = enc.copy()

            if max_length is not None:
                available = max_length - int(add_bos) - int(add_eos)
                tokens = tokens[:available]

            if add_bos:
                tokens = [self.bos_token, *tokens]
            if add_eos:
                tokens = [*tokens, self.eos_token]

            processed.append(tokens)

        if padding:
            max_len = max(len(seq) for seq in processed)
            return [seq + [self.pad_token] * (max_len - len(seq)) for seq in processed]
        return list(processed)

    def decode(self, batch: torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        batch_list = batch.tolist()

        if skip_special_tokens:
            special = {self.bos_token, self.eos_token, self.pad_token}
            batch_list = [[t for t in seq if t not in special] for seq in batch_list]

        return self.encoding.decode_batch(batch_list)

    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 11

    def encode_special(self, token: str) -> int:
        return self.special_tokens[token]

    def render_conversation(
        self, conversation: dict, max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        ids, mask = [], []

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(self.bos_token, 0)

        for _i, message in enumerate(messages):
            content = message["content"]

            if message["role"] == "user":
                value_ids = self.encoding.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.encoding.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encoding.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
