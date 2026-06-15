from typing import Literal, Dict, Any
from datasets import load_dataset
from mint.data.datasets.base import SFTTrainDataset, SFTEvalDataset, build_mc_prompt

class ARC(SFTTrainDataset, SFTEvalDataset):
    def __init__(self, subset: Literal["ARC-Easy", "ARC-Challenge"] = "ARC-Challenge", split: Literal["train", "validation", "test"] = "train") -> None:
        super().__init__()
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> Literal["categorical", "generative"]:
        return "categorical"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row: dict[Any, Any] = self.ds[index]
        letters = tuple(row["choices"]["label"])
        user_msg = build_mc_prompt(row["question"], letters, row["choices"]["text"])
        
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": row["answerKey"]}
            ],
            "letters": letters
        }

    def evaluate(self, conversation: Dict[str, Any], completion: str) -> bool:
        return completion == conversation['messages'][-1]['content']
