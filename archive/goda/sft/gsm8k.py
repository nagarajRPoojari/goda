import re
from typing import Literal, Dict, Any
from datasets import load_dataset
from goda.sft.base import SFTTrainDataset, SFTEvalDataset

class GSM8K(SFTTrainDataset, SFTEvalDataset):
    ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    def __init__(self, subset: Literal["main", "socratic"] = "main",
                 split: Literal["train", "test"] = "train") -> None:
        super().__init__()
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> Literal["categorical", "generative"]:
        return "generative"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.ds[index]
        parts = []
        
        for segment in re.split(r'(<<[^>]+>>)', row['answer']):
            if segment.startswith('<<') and segment.endswith('>>'):
                inner = segment[2:-2]
                expr, result = inner.rsplit('=', 1) if '=' in inner else (inner, "")
                parts.extend([
                    {"type": "python", "text": expr},
                    {"type": "python_output", "text": result}
                ])
            else:
                parts.append({"type": "text", "text": segment})
        
        return {
            "messages": [
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": parts}
            ]
        }

    def evaluate(self, conversation: Dict[str, Any], completion: str) -> bool:
        ground_truth_parts = conversation['messages'][-1]['content']
        ground_truth_text = ground_truth_parts[-1]['text']
        
        ref_num = self._extract_answer(ground_truth_text)
        pred_num = self._extract_answer(completion)
        return pred_num == ref_num
    
    def _extract_answer(self, text: str) -> str | None:
        match = self.ANSWER_RE.search(text)
        return match.group(1).strip().replace(",", "") if match else None

