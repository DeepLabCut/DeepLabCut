from typing import Callable, Self
from pathlib import Path
from dataclasses import asdict

from omegaconf import OmegaConf, DictConfig
from pydantic import TypeAdapter

from deeplabcut.core.config.utils import read_config_as_dict, write_config, pretty_print


class ConfigMixin:
    """
    Base mixin class for configuration classes.

    Each configuration is defined as a pydantic dataclass and inherits from this
    class to provide common functionality for loading, validating, and working
    with configuration data.

    This mixin provides methods for:
    - Loading configurations from dictionaries or YAML files
    - Validating configuration data against pydantic models
    - Converting configurations to dictionaries
    - Pretty printing configuration data
    """

    @classmethod
    def validate_dict(
        cls,
        cfg_dict: dict,
    ) -> DictConfig:
        """
        Load a dictionary as DictConfig, validating it against the pydantic model.
        Args:
            cfg_dict: the configuration as a dictionary

        Returns:
            The configuration file as a DictConfig
        """
        cfg: DictConfig = OmegaConf.create(cfg_dict)
        resolved: dict = OmegaConf.to_container(cfg, resolve=True)
        TypeAdapter[cls](cls).validate_python(resolved, extra="forbid")
        return cfg

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> Self:
        return cls(**cfg_dict)

    @classmethod
    def from_any(
        cls,
        config: Self | dict | DictConfig | str | Path,
    ) -> Self:
        """
        Create a new instance from various configuration formats.

        Args:
            config: Configuration as a ConfigMixin instance, dictionary, or DictConfig.
                   If already a ConfigMixin instance, returns it as-is.

        Returns:
            A new instance of the ConfigMixin subclass.
        """
        if isinstance(config, ConfigMixin):
            return config
        elif isinstance(config, str | Path):
            return cls.from_yaml(config)
        elif isinstance(config, DictConfig):
            return cls.from_dict(OmegaConf.to_container(config, resolve=True))
        elif isinstance(config, dict):
            return cls.from_dict(config)
        else:
            raise TypeError(
                "Failure to load configuration: Expected pydantic dataclass, "
                f"dictionary, DictConfig, string, or Path. Got {type(config)}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Self:
        return cls.from_dict(read_config_as_dict(yaml_path))

    def to_yaml(self, yaml_path: str | Path, overwrite: bool = True) -> None:
        write_config(yaml_path, self.to_dict(), overwrite=overwrite)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_dictconfig(self) -> DictConfig:
        return OmegaConf.create(self.to_dict())

    def print(
        self,
        indent: int = 0,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        pretty_print(config=self.to_dict(), indent=indent, print_fn=print_fn)
