#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for the GUI's eager warm-up of the lazily-loaded public API."""

import importlib.util
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

import deeplabcut
from deeplabcut.gui import warmup


def _warmed_exports() -> list[str]:
    return [name for name in deeplabcut.__all__ if name not in warmup.EXCLUDED_EXPORTS]


def test_warm_up_covers_the_public_api(monkeypatch) -> None:
    monkeypatch.setattr(warmup, "EXTRA_MODULES", ())
    seen: list[str] = []

    warmup.warm_up(on_item=seen.append)

    assert seen == _warmed_exports()


def test_excluded_exports_are_real_exports() -> None:
    """A typo here would silently exclude nothing."""
    unknown = warmup.EXCLUDED_EXPORTS - set(deeplabcut.__all__)
    assert not unknown, f"EXCLUDED_EXPORTS names that are not exports: {sorted(unknown)}"


def test_warm_up_reports_failures_instead_of_raising(monkeypatch) -> None:
    """A broken target must degrade to a report entry, never take the GUI down."""
    monkeypatch.setattr(deeplabcut, "__all__", [])
    monkeypatch.setattr(warmup, "EXTRA_MODULES", ("deeplabcut.not_a_real_module",))

    report = warmup.warm_up()

    assert report.loaded == ()
    assert set(report.failed) == {"deeplabcut.not_a_real_module"}
    assert not report.ok


def test_is_enabled_matches_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv("DLC_GUI_WARMUP", raising=False)
    assert warmup.is_enabled()

    for value in ("0", "false", "no", "NO", " False "):
        monkeypatch.setenv("DLC_GUI_WARMUP", value)
        assert not warmup.is_enabled(), value

    monkeypatch.setenv("DLC_GUI_WARMUP", "1")
    assert warmup.is_enabled()


def test_start_warmup_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DLC_GUI_WARMUP", "0")
    assert warmup.start_warmup() is None


def test_start_warmup_runs_on_a_daemon_thread(monkeypatch) -> None:
    monkeypatch.delenv("DLC_GUI_WARMUP", raising=False)
    monkeypatch.setattr(deeplabcut, "__all__", [])
    monkeypatch.setattr(warmup, "EXTRA_MODULES", ())

    thread = warmup.start_warmup()

    assert thread is not None
    assert thread.daemon  # must never delay interpreter shutdown
    thread.join(timeout=30)
    assert not thread.is_alive()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_warm_up_loads_every_target() -> None:
    """Warm-up swallows failures at runtime, so assert it actually loads here."""
    report = warmup.warm_up()

    assert report.ok, "warm-up targets that failed to load:\n" + "\n".join(
        f"  {name}: {type(exc).__name__}: {exc}" for name, exc in sorted(report.failed.items())
    )
    assert set(report.loaded) == set(_warmed_exports()) | set(warmup.EXTRA_MODULES)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_warm_up_avoids_excluded_dependencies() -> None:
    """Warm-up must not pull in 3D or the deprecated TensorFlow stack.

    Run out-of-process: an earlier test in the session may already have imported
    either one, which would make an in-process check pass for the wrong reason.
    """
    code = "\n".join(
        [
            "import sys",
            "from deeplabcut.gui.warmup import warm_up",
            "assert warm_up().ok",
            "for mod in ('tensorflow', 'deeplabcut.pose_estimation_3d'):",
            "    assert mod not in sys.modules, f'{mod} imported by warm-up'",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
