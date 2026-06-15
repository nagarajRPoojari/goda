from abc import ABC, abstractmethod
from typing import Any


import torch.nn as nn
from enum import Enum
from pathlib import Path

from mint.utils.device import Device
from goda.config import Config
from mint.utils.logger import logger

class Task(Enum):
    MC = "multiple_choice"
    SCHEMA = "schema"
    LM = "language_modeling"


class Evaluator(ABC):
    
    def __init__(self, model: nn.Module, config: Config, device: Device):
        self.model = model
        self.config = config
        self.device = device
        self.process_info = device.process_info()
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, Any]:
        pass

