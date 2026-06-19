from typing import Any, Literal

from datasets import load_dataset

from goda.sft.base import SFTTrainDataset


class SmolTalk(SFTTrainDataset):
    def __init__(self, split: Literal["train", "test"] = "train") -> None:
        super().__init__()
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"messages": self.ds[index]["messages"]}

