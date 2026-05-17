from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Sampler(ABC):

    @abstractmethod
    def sample(self, num_samples: int = 1) -> List[Dict[str, Any]]:
        ...
