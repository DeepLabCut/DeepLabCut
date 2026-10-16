from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BeforeValidator, Field, GetPydanticSchema, InstanceOf
from pydantic_core import PydanticUseDefault


def _describe(value: float, name: str | None = None) -> str:
    return f"{name} ({value})" if name else f"{value}"


def greater_than(
    value: float,
    threshold: float,
    name: str | None = None,
    threshold_name: str | None = None,
) -> None:
    if value <= threshold:
        raise ValueError(f"{_describe(value, name)} must be greater than {_describe(threshold, threshold_name)}")


def less_than(
    value: float,
    threshold: float,
    name: str,
    threshold_name: str | None = None,
) -> None:
    if value >= threshold:
        raise ValueError(f"{_describe(value, name)} must be less than {_describe(threshold, threshold_name)}")


def unique_values(values: Sequence[Any]) -> Sequence[Any]:
    if len(values) != len(set(values)):
        raise ValueError("Values must be unique")
    return values


def validate_crop_bounds(
    *,
    x1: int | None,
    x2: int | None,
    y1: int | None,
    y2: int | None,
) -> None:
    bounds = (x1, x2, y1, y2)
    if any(value is None for value in bounds):
        if any(value is not None for value in bounds):
            raise ValueError("Crop bounds x1, x2, y1, and y2 must either all be set or all be omitted")
        return

    less_than(x1, x2, name="x1", threshold_name="x2")
    less_than(y1, y2, name="y1", threshold_name="y2")


def reject_reserved_keys(
    kwargs: Mapping[str, Any] | None,
    reserved: Mapping[str, str | None],
    name: str,
) -> None:
    """Rejects reserved keys in a pass-through kwargs mapping.

    Example:
        ```python
        class ExampleDataLoaderConfig(DLCBaseConfig):
            batch_size: int = 8
            dataloader_kwargs: dict[str, Any] | None = None

            @field_validator("dataloader_kwargs")
            @classmethod
            def validate_dataloader_kwargs(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
                reject_reserved_keys(v, {"batch_size": "batch_size"}, name="dataloader_kwargs")
                return v
        ```

    Args:
        kwargs: The pass-through kwargs to check (None is treated as empty).
        reserved: The reserved keys that are forbidden in the pass-through kwargs.
        name: Name of the pass-through field, used in the error message.

    Raises:
        ValueError: If ``kwargs`` contains any reserved key.
    """
    clashes = sorted(set(kwargs or {}) & set(reserved))
    if clashes:
        hints = [f"'{key}' (set via `{reserved[key]}`)" if reserved[key] else f"'{key}'" for key in clashes]
        raise ValueError(f"{name} cannot include reserved options: {', '.join(hints)}")


def reject_multiprocessing_only_kwargs(
    kwargs: Mapping[str, Any] | None,
    num_workers: int,
    name: str,
) -> None:
    """Rejects pass-through kwargs that require worker processes.

    ``torch.utils.data.DataLoader`` raises for these itself, but only once training
    starts and naming neither the option nor the config field. Checking them at config
    load keeps the failure where the user can act on it.

    Args:
        kwargs: The pass-through kwargs to check (None is treated as empty).
        num_workers: The configured number of data loader workers.
        name: Name of the pass-through field, used in the error message.

    Raises:
        ValueError: If ``kwargs`` sets a multiprocessing-only option while ``num_workers`` is 0.
    """
    if num_workers > 0:
        return

    # DataLoader only rejects an option that actually asks for something: persistent_workers
    # defaults to False and prefetch_factor to None, so those values pass there too.
    kwargs = kwargs or {}
    set_options = []
    if kwargs.get("persistent_workers"):
        set_options.append("persistent_workers")
    if kwargs.get("prefetch_factor") is not None:
        set_options.append("prefetch_factor")
    if not set_options:
        return

    options = ", ".join(f"'{key}'" for key in set_options)
    raise ValueError(
        f"{name} cannot include {options} with no data loader workers: "
        f"set `dataloader_workers` above 0 to enable multiprocessing"
    )


def _coerce_ndarray(v):
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v, dtype=int)


def _bodypart_pair(values: Sequence[Any]) -> list[str]:
    if len(values) != 2:
        raise ValueError(f"Each bodypart pair must contain exactly two bodyparts, got {len(values)}")
    return list(unique_values(values))


def _use_default_if_none(value: Any) -> Any:
    """Fall back to the field's declared default when the input value is None."""
    if value is None:
        raise PydanticUseDefault
    return value


T = TypeVar("T")

# Normalize an explicit ``null`` to the field's default,
# e.g. ``box_score_thresh: DefaultIfNone[Fraction] = 0.01``
DefaultIfNone: TypeAlias = Annotated[T, BeforeValidator(_use_default_if_none)]

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
UniqueStrList = Annotated[list[str], AfterValidator(unique_values)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
StrictPositiveInt = Annotated[int, Field(ge=1)]
NDArrayInt = Annotated[
    NDArray,
    BeforeValidator(_coerce_ndarray),
    GetPydanticSchema(
        lambda _s, h: h(InstanceOf[np.ndarray]),
        lambda _s, h: h(InstanceOf[np.ndarray]),
    ),
]
BodypartPair = Annotated[list[str], AfterValidator(_bodypart_pair)]
