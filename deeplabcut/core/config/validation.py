from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import AfterValidator, Field


def _describe(value: float, name: str | None = None) -> str:
    return f"{name} ({value})" if name else f"{value}"


def greater_than(
    value: float,
    threshold: float,
    name: str | None = None,
    threshold_name: str | None = None,
) -> None:
    if value < threshold:
        raise ValueError(f"{_describe(value, name)} must be greater than {_describe(threshold, threshold_name)}")


def less_than(
    value: float,
    threshold: float,
    name: str,
    threshold_name: str | None = None,
) -> None:
    if value > threshold:
        raise ValueError(f"{_describe(value, name)} must be less than {_describe(threshold, threshold_name)}")


def unique_values(values: Sequence[Any]) -> Sequence[Any]:
    if len(values) != len(set(values)):
        raise ValueError("Values must be unique")
    return values


Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
UniqueStrList = Annotated[list[str], AfterValidator(unique_values)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
StrictPositiveInt = Annotated[int, Field(ge=1)]
