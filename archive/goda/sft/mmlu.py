from typing import Any, Literal

from datasets import load_dataset

from goda.sft.base import SFTEvalDataset, SFTTrainDataset, build_mc_prompt


class MMLU(SFTTrainDataset, SFTEvalDataset):
    LETTERS = ("A", "B", "C", "D")

    def __init__(self, subset: Literal["all"] = "all", split: Literal["auxiliary_train", "validation", "dev", "test"] = "test") -> None:
        super().__init__()
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> Literal["categorical", "generative"]:
        return "categorical"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        user_msg = build_mc_prompt(row["question"], self.LETTERS, row["choices"])

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": self.LETTERS[row["answer"]]}
            ],
            "subject": row["subject"],
            "letters": self.LETTERS
        }

    def evaluate(self, conversation: dict[str, Any], completion: str) -> bool:
        return completion == conversation["messages"][-1]["content"]
