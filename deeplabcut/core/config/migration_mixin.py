from typing import Any, Self
from dataclasses import fields
from pydantic import model_validator
from pydantic_core import ArgsKwargs
from deeplabcut.core.config.versioning import migrate_config, CURRENT_CONFIG_VERSION


class MigrationMixin:
    """
    Migration mixin for configuration classes.
    
    This mixin provides a method to migrate the configuration to the current version
    before validation.
    """ 
    @model_validator(mode="wrap")
    @classmethod
    def migrate_then_validate(cls, data: Any, handler: Any) -> Self:
        """Run migration before Pydantic's standard model validation.
        
        Wraps the default validator: migrates raw dict/ArgsKwargs to current
        config version, then delegates to handler for normal validation.
        """
        # If data is already an instance or not a dict, pass through
        if isinstance(data, cls):
            return data

        # Convert to dictionary if ArgsKwargs is passed
        if isinstance(data, ArgsKwargs):
            names = [f.name for f in fields(cls)]
            data = dict(
                zip(names, data.args or []),
                **data.kwargs or {},
            )
        if isinstance(data, dict):
            data = migrate_config(data, target_version=CURRENT_CONFIG_VERSION)
        return handler(data)