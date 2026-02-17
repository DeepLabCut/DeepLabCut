from typing import Callable, Mapping
from typing_extensions import Self
from pathlib import Path
from dataclasses import asdict, fields
from enum import Enum
from functools import wraps

from omegaconf import OmegaConf, DictConfig, ListConfig
from pydantic import TypeAdapter
from ruamel.yaml.comments import CommentedMap

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
        TypeAdapter(cls).validate_python(resolved, extra="forbid")
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
    def from_yaml(cls, yaml_path: str | Path, ignore_empty: bool = True) -> Self:
        """
        Load a configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            ignore_empty: If True, empty/None values in the YAML are ignored and
                dataclass defaults are used instead. Defaults to True.

        Returns:
            A new instance of the configuration class.
        """
        # NOTE @deruyter92 2026-02-05: Default ignore_empty is now set to True to match
        # the prior behaviour of read_config. We should consider changing this to False
        # for stricter validation.
        yaml_dict = read_config_as_dict(yaml_path)
        if ignore_empty:
            yaml_dict = {k: v for k, v in yaml_dict.items() if v is not None}
        return cls.from_dict(yaml_dict)

    def to_yaml(self, yaml_path: str | Path, overwrite: bool = True) -> None:
        dict_data = self.to_dict_normalized()
        data = CommentedMap(dict_data)
        for f in fields(self):
            if (comment := f.metadata.get("comment")):
                data.yaml_set_comment_before_after_key(f.name, before=comment)
        write_config(yaml_path, data, overwrite=overwrite)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_dict_normalized(self) -> dict:
        return _normalize_for_serialization(self.to_dict())

    def to_dictconfig(self) -> DictConfig:
        return OmegaConf.create(self.to_dict())

    def print(
        self,
        indent: int = 0,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        pretty_print(config=self.to_dict(), indent=indent, print_fn=print_fn)


def _to_plain(value):
    """Convert a ConfigMixin, DictConfig, or ListConfig to a plain Python object."""
    if isinstance(value, ConfigMixin):
        return value.to_dict()
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def ensure_plain_config(fn: Callable) -> Callable:
    """Decorator that converts config arguments to plain Python dicts.

    Any positional or keyword argument that is a :class:`ConfigMixin`,
    :class:`~omegaconf.DictConfig`, or :class:`~omegaconf.ListConfig` is
    automatically converted to a plain ``dict`` / ``list`` before the
    decorated function is called.

    Example::

        @ensure_plain_config
        def train(model_cfg: dict, lr: float = 1e-3):
            ...

        train(my_pose_config)  # PoseConfig → dict
        train(omega_dict)      # DictConfig → dict
        train(plain_dict)      # dict passed through unchanged
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(_to_plain(a) for a in args)
        kwargs = {k: _to_plain(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


def _normalize_for_serialization(obj):
    """Recursively normalize Paths to strings and Enums to values."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Mapping):
        return type(obj)({k: _normalize_for_serialization(v) for k, v in obj.items()})
    return obj