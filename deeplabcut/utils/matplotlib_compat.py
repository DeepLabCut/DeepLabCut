"""Small compatibility layer for Matplotlib API migrations.

A single choke point for the Matplotlib APIs DeepLabCut calls from many places,
so that adapting to an upstream change is one edit rather than one per call
site.  Each helper wraps an API that Matplotlib has already moved, or
whose current spelling is deprecated.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

#  Deliberately not every sublist an ``Axes`` exposes
# ``texts``, ``tables`` and ``containers`` are omitted
# because no DeepLabCut caller used them, and these five reproduce the
# behaviour of the removal loops this helper replaced
ARTIST_KINDS = ("lines", "collections", "artists", "patches", "images")

# Matplotlib builds this logger with ``logging.getLogger(__name__)`` in
# ``matplotlib/axes/_axes.py``, so the name is enough to reach it.
AXES_LOGGER_NAME = "matplotlib.axes._axes"


def get_colormap(
    name: str | Colormap | None = None,
    lut: int | None = None,
) -> Colormap:
    """Return a colormap, optionally resampled to ``lut`` colors."""
    return plt.get_cmap(name, lut)


def remove_artists(ax: Axes, *kinds: str) -> None:
    """Remove artists from an ``Axes``, one children sublist at a time.

    Matplotlib 3.7 made the ``Axes`` children sublists (``ax.lines``,
    ``ax.collections``, ...) immutable ``ArtistList`` views and deprecated
    concatenating them with ``+``.  Iterating a concatenation also happened to
    be the only thing making the removal loops safe, since ``+`` returned a
    plain list and therefore a snapshot; iterating a live ``ArtistList`` while
    calling ``Artist.remove()`` skips artists.  This helper snapshots each
    sublist explicitly instead.

    Args:
        ax: The axes to remove artists from.
        *kinds: Names of the children sublists to clear, e.g. ``"collections"``.
            Defaults to every kind in ``ARTIST_KINDS``.
    """
    for kind in kinds or ARTIST_KINDS:
        for artist in list(getattr(ax, kind)):
            artist.remove()


def silence_axes_logger(level: int | str = "ERROR") -> None:
    """Raise the log level of Matplotlib's ``Axes`` logger.

    Suppresses the chatty per-artist messages Matplotlib emits while drawing,
    most notably the "Unable to determine Axes to steal space for Colorbar"
    and invalid-color warnings raised during 3D plotting.

    Args:
        level: Any level accepted by ``logging.Logger.setLevel``.
    """
    logging.getLogger(AXES_LOGGER_NAME).setLevel(level)
