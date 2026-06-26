from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BeforeValidator, Field, GetPydanticSchema, InstanceOf


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


def _coerce_ndarray(v):
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v, dtype=int)


def _bodypart_pair(values: Sequence[Any]) -> list[str]:
    if len(values) != 2:
        raise ValueError(f"Each bodypart pair must contain exactly two bodyparts, got {len(values)}")
    return list(unique_values(values))


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
