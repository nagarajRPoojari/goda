import re
from typing import Dict, Any, Literal
from datasets import load_dataset
from mint.data.datasets.base import SFTTrainDataset, SFTEvalDataset
from mint.utils.exec import execute_code

class HumanEval(SFTTrainDataset, SFTEvalDataset):
    def __init__(self) -> None:
        super().__init__()
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self) -> Literal["categorical", "generative"]:
        return "generative"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.ds[index]
        solution = f"{row['prompt']}\n{row['canonical_solution']}"
        
        return {
            "messages": [
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": solution}
            ],
            "entry_point": row['entry_point'],
            "test": row['test']
        }

    def evaluate(self, conversation: Dict[str, Any], completion: str) -> bool:
        imports = self._extract_imports(conversation['messages'][0]['content'])
        code = self._extract_code(completion)
        
        program = f"{imports}\n\n{code}\n\n{conversation['test']}\ncheck({conversation['entry_point']})"
        result = execute_code(program)
        return result.success
    
    def _extract_imports(self, prompt: str) -> str:
        imports = []
        for line in prompt.split('\n'):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports.append(stripped)
            elif stripped and not stripped.startswith('#'):
                break
        return '\n'.join(imports)
    
    def _extract_code(self, completion: str) -> str:
        pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, completion, re.DOTALL)
        return matches[0].strip() if matches else completion.strip()

