from abc import ABC, abstractmethod

from torch import nn


class Adapter(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> None:  # noqa: ANN003
        pass
