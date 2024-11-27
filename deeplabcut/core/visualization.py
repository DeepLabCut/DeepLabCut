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
"""Visualization methods for """
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def form_figure(nx, ny) -> tuple[plt.Figure, plt.Axes]:
    """Forms a figure on which to plot images"""
    fig, ax = plt.subplots(frameon=False)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.axis("off")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def visualize_scoremaps(
    image: np.ndarray,
    scmap: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots scoremaps as an image overlay.

    Args:
        image: An image as a numpy array of shape (h, w, channels)
        scmap: A scoremap of shape (h, w)

    Returns:
        The figure and axis on which the image scoremap was plot.
    """
    ny, nx = np.shape(image)[:2]
    fig, ax = form_figure(nx, ny)
    ax.imshow(image)
    ax.imshow(scmap, alpha=0.5)
    return fig, ax


def visualize_locrefs(
    image: np.ndarray,
    scmap: np.ndarray,
    locref_x: np.ndarray,
    locref_y: np.ndarray,
    step: int = 5,
    zoom_width: int = 0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a scoremap and the corresponding location refinement fields on an image.

    Args:
        image: An image as a numpy array of shape (h, w, channels)
        scmap: A scoremap of shape (h, w)
        locref_x: The x-coordinate of the location refinement field, of shape (h, w)
        locref_y: The y-coordinate of the location refinement field, of shape (h, w)
        step: The step with which to plot the scoremaps.
        zoom_width: The zoom width with which to plot the scoremaps.

    Returns:
        The figure and axis on which the image scoremap was plot.
    """
    fig, ax = visualize_scoremaps(image, scmap)
    X, Y = np.meshgrid(np.arange(locref_x.shape[1]), np.arange(locref_x.shape[0]))
    M = np.zeros(locref_x.shape, dtype=bool)
    M[scmap < 0.5] = True
    U = np.ma.masked_array(locref_x, mask=M)
    V = np.ma.masked_array(locref_y, mask=M)
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        color="r",
        units="x",
        scale_units="xy",
        scale=1,
        angles="xy",
    )
    if zoom_width > 0:
        maxloc = np.unravel_index(np.argmax(scmap), scmap.shape)
        ax.set_xlim(maxloc[1] - zoom_width, maxloc[1] + zoom_width)
        ax.set_ylim(maxloc[0] + zoom_width, maxloc[0] - zoom_width)
    return fig, ax


def visualize_paf(
    image: np.ndarray,
    paf: np.ndarray,
    step: int = 5,
    colors=None,
) -> tuple[plt.Figure, plt.Axes]:
    """

    Args:
        image:
        paf:
        step:
        colors:

    Returns:

    """
    ny, nx = np.shape(image)[:2]
    fig, ax = form_figure(nx, ny)
    ax.imshow(image)
    n_fields = paf.shape[2]
    if colors is None:
        colors = ["r"] * n_fields
    for n in range(n_fields):
        U = paf[:, :, n, 0]
        V = paf[:, :, n, 1]
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
        M = np.zeros(U.shape, dtype=bool)
        M[U**2 + V**2 < 0.5 * 0.5**2] = True
        U = np.ma.masked_array(U, mask=M)
        V = np.ma.masked_array(V, mask=M)
        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            U[::step, ::step],
            V[::step, ::step],
            scale=50,
            headaxislength=4,
            alpha=1,
            width=0.002,
            color=colors[n],
            angles="xy",
        )
    return fig, ax
