"""Opt-in pandas 2.3 future-behavior checks for CI/local DLC test runs."""

from __future__ import annotations

import os

import pandas as pd
from packaging.version import Version

ENABLED_ENV = "DLC_PANDAS_FUTURE"


def is_enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "").lower() in {"1", "true", "yes"}


def configure_pandas_future_if_enabled() -> None:
    if not is_enabled():
        return

    ver = Version(pd.__version__)
    if ver < Version("2.3") or ver >= Version("3"):
        raise RuntimeError(f"pandas future mode requires pandas 2.3.x, got {pd.__version__}")

    pd.options.future.infer_string = True
    pd.options.mode.copy_on_write = "warn"

    print(
        f"pandas future mode enabled: pandas={pd.__version__}, "
        f"infer_string={pd.options.future.infer_string}, "
        f"copy_on_write={pd.options.mode.copy_on_write}"
    )
