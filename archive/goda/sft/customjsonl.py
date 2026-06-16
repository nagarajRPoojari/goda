import os
import json
from typing import Dict, Any
from goda.sft.base import SFTTrainDataset

class CustomJSON(SFTTrainDataset):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.conversations = []
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.conversations.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"messages": self.conversations[index]}
