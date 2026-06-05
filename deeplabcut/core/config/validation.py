from typing import Annotated

from pydantic import AfterValidator, Field

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
UniqueStringList = Annotated[list[str], AfterValidator(lambda x: len(x) == len(set(x)))]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
StrictPositiveInt = Annotated[int, Field(ge=1)]


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
