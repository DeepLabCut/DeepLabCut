"""Small compatibility layer for Matplotlib API migrations.

Prefer current Matplotlib APIs and fall back to legacy APIs only when needed.

Legacy helpers should be removed when DeepLabCut drops support for
Matplotlib versions without ``matplotlib.colormaps``.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from deeplabcut.core.deprecation import deprecated

DLC_MATPLOTLIB_LEGACY_DEPRECATED_SINCE = "3.0.1"  # FIXME @C-Achard which one do we want?


def get_colormap(
    name: str | Colormap | None = None,
    lut: int | None = None,
) -> Colormap:
    """Return a colormap, optionally resampled to ``lut`` colors."""
    return plt.get_cmap(name, lut)


def get_colormap_names() -> list[str]:
    """Return the names of registered colormaps."""
    if hasattr(matplotlib, "colormaps"):
        return list(matplotlib.colormaps)

    return list(plt.colormaps())


def register_colormap(
    cmap: Colormap,
    *,
    name: str | None = None,
    force: bool = False,
) -> None:
    """Register a colormap using the newest available API."""
    if hasattr(matplotlib, "colormaps"):
        matplotlib.colormaps.register(
            cmap,
            name=name,
            force=force,
        )
        return

    _legacy_register_colormap(
        cmap,
        name=name,
        force=force,
    )


def unregister_colormap(name: str) -> None:
    """Unregister a colormap using the newest available API."""
    if hasattr(matplotlib, "colormaps"):
        matplotlib.colormaps.unregister(name)
        return

    _legacy_unregister_colormap(name)


@deprecated(
    replacement="matplotlib.colormaps",
    since=DLC_MATPLOTLIB_LEGACY_DEPRECATED_SINCE,
    stacklevel=3,
)
def _legacy_get_colormap_names() -> list[str]:
    """Return registered colormap names using the legacy API."""
    return list(plt.colormaps())


@deprecated(
    replacement="matplotlib.colormaps.register",
    since=DLC_MATPLOTLIB_LEGACY_DEPRECATED_SINCE,
    stacklevel=3,
)
def _legacy_register_colormap(
    cmap: Colormap,
    *,
    name: str | None = None,
    force: bool = False,
) -> None:
    """Register a colormap using the legacy Matplotlib API."""
    matplotlib.cm.register_cmap(
        name=name,
        cmap=cmap,
        override_builtin=force,
    )


@deprecated(
    replacement="matplotlib.colormaps.unregister",
    since=DLC_MATPLOTLIB_LEGACY_DEPRECATED_SINCE,
    stacklevel=3,
)
def _legacy_unregister_colormap(name: str) -> None:
    """Unregister a colormap using the legacy Matplotlib API."""
    matplotlib.cm.unregister_cmap(name)
