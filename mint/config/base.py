import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    ...

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if value == 'null' or value == 'None':
                config_dict[key] = None
        
        return cls(**config_dict)
    

class PretrainConfig(Config):
    ...


class SFTConfig(Config):
    ...


class ModelConfig(Config):
    ...