from abc import ABC, abstractmethod

from torch import nn


class Adapter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs):
        pass
