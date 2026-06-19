import tomllib  # <-- Built-in in Python 3.11+
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin


T = TypeVar("T", bound="Config")


@dataclass
class Config:
    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Recursively parses a dictionary into the target dataclass schema."""
        if not data:
            return cls()

        init_kwargs = {}
        class_fields = {f.name: f.type for f in fields(cls)}

        for key, value in data.items():
            if key in class_fields:
                field_type = class_fields[key]

                # Resolve Union/Optional types
                origin = get_origin(field_type)
                if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
                    for arg in get_args(field_type):
                        if is_dataclass(arg):
                            field_type = arg
                            break

                # Parse nested dictionary into nested dataclass
                if isinstance(value, dict) and is_dataclass(field_type):
                    if hasattr(field_type, "from_dict"):
                        init_kwargs[key] = field_type.from_dict(value)
                    else:
                        init_kwargs[key] = cls._load_plain_dataclass(field_type, value)
                else:
                    init_kwargs[key] = value
            else:
                init_kwargs[key] = value

        return cls(**init_kwargs)

    @classmethod
    def _load_plain_dataclass(cls, target_cls: type, data: dict[str, Any]) -> Any:
        """Helper to recursively parse vanilla dataclasses."""
        if not isinstance(data, dict):
            return data

        init_kwargs = {}
        class_fields = {f.name: f.type for f in fields(target_cls)}

        for key, value in data.items():
            if key in class_fields:
                field_type = class_fields[key]
                origin = get_origin(field_type)
                if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
                    for arg in get_args(field_type):
                        if is_dataclass(arg):
                            field_type = arg
                            break

                if isinstance(value, dict) and is_dataclass(field_type):
                    init_kwargs[key] = cls._load_plain_dataclass(field_type, value)
                else:
                    init_kwargs[key] = value
            else:
                init_kwargs[key] = value

        return target_cls(**init_kwargs)

    @classmethod
    def from_toml(cls: type[T], toml_path: str) -> T:
        """Loads a TOML file and passes it to the dictionary parser."""
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")

        # tomllib expects a binary file stream ('rb')
        with open(path, "rb") as f:
            config_dict = tomllib.load(f)

        return cls.from_dict(config_dict)
