#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Eager import of DeepLabCut's public API, for the GUI.

The public API is loaded lazily from ``deeplabcut/__init__.pyi``, which keeps
``import deeplabcut`` cheap for scripts and the CLI. In the GUI every deferred
import instead becomes a frozen window on the first click, so the GUI warms the
whole API up front on a background thread.

All modules exposed by ``deeplabcut.__all__`` are loaded eagerly in the warmup,
plus some heavy third-party libraries that are not exposed.

Disable the GUI warmup by setting env variable DLC_GUI_WARMUP=0.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Third-party libraries that are NOT exposed when importing deeplabcut.__all__.
EXTRA_MODULES: tuple[str, ...] = (
    "albumentations",
    "matplotlib",
    "timm",
    "torch",
    "torchvision",
)

# Public exports that the GUI has no tab for and should not be loaded.
EXCLUDED_EXPORTS: frozenset[str] = frozenset(
    {
        # deeplabcut.pose_estimation_3d
        "calibrate_cameras",
        "check_undistortion",
        "create_labeled_video_3d",
        "triangulate",
    }
)


@dataclass(frozen=True)
class WarmupReport:
    """Outcome of a warm-up pass."""

    loaded: tuple[str, ...] = ()
    failed: Mapping[str, Exception] = field(default_factory=dict)
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed


def is_enabled() -> bool:
    """Return whether warm-up should run.

    Set ``DLC_GUI_WARMUP=0`` to keep the GUI lazy, e.g. when profiling startup.
    """
    return os.environ.get("DLC_GUI_WARMUP", "").strip().lower() not in {"0", "false", "no"}


def warm_up(*, on_item: Callable[[str], None] | None = None) -> WarmupReport:
    """Import every public export the GUI can reach, plus the modules the API defers.

    Failures are collected instead of raised, so a missing optional dependency
    surfaces from the tab that needs it rather than from a window that refuses
    to open.

    Args:
        on_item: called with each name before it is loaded, for progress
            reporting. Runs on the calling thread.

    Returns:
        What loaded, what did not, and how long it took.
    """
    import deeplabcut

    started = time.perf_counter()
    loaded: list[str] = []
    failed: dict[str, Exception] = {}

    targets: list[tuple[str, Callable[[], object]]] = [
        (name, lambda n=name: getattr(deeplabcut, n)) for name in deeplabcut.__all__ if name not in EXCLUDED_EXPORTS
    ]
    targets += [(name, lambda n=name: importlib.import_module(n)) for name in EXTRA_MODULES]

    for name, load in targets:
        if on_item is not None:
            on_item(name)
        try:
            load()
        except Exception as exc:
            failed[name] = exc
            logger.debug("GUI warm-up could not load %r: %s", name, exc, exc_info=True)
        else:
            loaded.append(name)

    duration = time.perf_counter() - started
    logger.debug("GUI warm-up loaded %d/%d targets in %.2fs", len(loaded), len(targets), duration)

    return WarmupReport(loaded=tuple(loaded), failed=failed, duration=duration)


def start_warmup(*, on_item: Callable[[str], None] | None = None) -> threading.Thread | None:
    """Run ``warm_up`` on a daemon thread, or return ``None`` if disabled.

    Racing the GUI thread is safe: CPython's per-module import lock makes a
    click that needs a module still being warmed block until that import
    finishes, so it can never see a half-initialised module.
    """
    if not is_enabled():
        return None

    thread = threading.Thread(
        target=warm_up,
        kwargs={"on_item": on_item},
        name="dlc-gui-warmup",
        daemon=True,
    )
    thread.start()
    return thread
