from typing import Any, List, Optional, Union
import tiktoken
import torch

class Tokenizer:
    def __init__(self, model_name: str = "gpt-2") -> None:
        self.encoding = tiktoken.encoding_for_model(model_name=model_name)
        self.bos_token = self.encoding.max_token_value + 1
        self.eos_token = self.encoding.eot_token  # Use tiktoken's EOT token
        self.pad_token = self.encoding.max_token_value + 2
        
    def encode(self,  batch: List[str],  add_bos: bool = True, add_eos: bool = False, padding: bool = True, max_length:Optional[int] = None ) -> torch.Tensor:
        return torch.tensor(self.encode_to_list(
            batch=batch,
            add_bos=add_bos,
            add_eos=add_eos,
            padding=padding,
            max_length=max_length
        ), dtype=torch.long)

    def encode_to_list(self,  batch: List[str],  add_bos: bool = True, add_eos: bool = False, padding: bool = True, max_length:Optional[int] = None ) -> list[Any]:
        encoded = self.encoding.encode_batch(text=batch)
        
        processed = []
        for enc in encoded:
            tokens = enc.copy()

            if max_length is not None:
                available = max_length - int(add_bos) - int(add_eos)
                tokens = tokens[:available]    

            if add_bos:
                tokens = [self.bos_token] + tokens
            if add_eos:
                tokens = tokens + [self.eos_token]
                
            processed.append(tokens)
        
        if padding:
            max_len = max(len(seq) for seq in processed)
            padded = [seq + [self.pad_token] * (max_len - len(seq)) for seq in processed]
            return padded
        else:
            return [seq for seq in processed] 

    def decode(self,  batch: torch.Tensor,  skip_special_tokens: bool = True) -> List[str]:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
            
        batch_list = batch.tolist()
        
        if skip_special_tokens:
            special = {self.bos_token, self.eos_token, self.pad_token}
            batch_list = [[t for t in seq if t not in special] for seq in batch_list]
        
        return self.encoding.decode_batch(batch_list)
    
    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 2
