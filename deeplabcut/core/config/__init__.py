import sys

from dlccore.config import base_config, project_config, validation, versioning
from dlccore.config.base_config import DLCBaseConfig, DLCVersionedConfig
from dlccore.config.project_config import ProjectConfig

from deeplabcut.core.config.utils import (
    create_config_template,
    create_config_template_3d,
    edit_config,
    ensure_plain_config,
    get_yaml_dumper,
    get_yaml_loader,
    normalize_for_serialization,
    pretty_print,
    read_config,
    read_config_as_dict,
    resolve_alias,
    resolve_aliases_in_dict,
    write_config,
    write_config_3d,
    write_config_3d_template,
    write_project_config,
)

# The moved modules keep their old import paths, and patches through them reach dlccore.
for _module in (base_config, project_config, validation, versioning):
    sys.modules[f"{__name__}.{_module.__name__.rpartition('.')[2]}"] = _module
del _module

__all__ = [
    "DLCBaseConfig",
    "DLCVersionedConfig",
    "ProjectConfig",
    "create_config_template",
    "create_config_template_3d",
    "edit_config",
    "pretty_print",
    "read_config",
    "read_config_as_dict",
    "resolve_alias",
    "resolve_aliases_in_dict",
    "write_config",
    "write_config_3d",
    "write_config_3d_template",
    "write_project_config",
    "get_yaml_loader",
    "get_yaml_dumper",
    "versioning",
    "validation",
    "normalize_for_serialization",
    "ensure_plain_config",
]
