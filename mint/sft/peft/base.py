from abc import ABC, abstractmethod

import torch.nn as nn


class Adapter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs):
        pass
