from abc import ABC, abstractmethod
from typing import Any


class Sampler(ABC):

    @abstractmethod
    def sample(self, num_samples: int = 1) -> list[dict[str, Any]]:
        ...
