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

from pathlib import Path

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
    """Plots a scoremap and the corresponding location refinement field on an image.

    Args:
        image: An image as a numpy array of shape (h, w, channels)
        scmap: A scoremap of shape (h, w)
        locref_x: The x-coordinate of the location refinement field, of shape (h, w)
        locref_y: The y-coordinate of the location refinement field, of shape (h, w)
        step: The step with which to plot the location refinement field.
        zoom_width: The zoom width with which to plot the scoremaps.

    Returns:
        The figure and axis on which the image scoremap and locref field were plot.
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
    colors: list | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the PAF on top of the image.

    Args:
        image: Shape (height, width, channels). The image on which the model was run.
        paf: Shape (height, width, 2 * len(paf_graph)). The PAF output by the model.
        step: The step with which to plot the scoremaps.
        colors: The colormap to use.

    Returns:
        The figure and axis on which the image PAF was plot.
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


def generate_model_output_plots(
    output_folder: Path,
    image_name: str,
    bodypart_names: list[str],
    bodyparts_to_plot: list[str],
    image: np.ndarray,
    scmap: np.ndarray,
    locref: np.ndarray | None = None,
    paf: np.ndarray | None = None,
    paf_graph: list[tuple[int, int]] | None = None,
    paf_all_in_one: bool = True,
    paf_colormap: str = "rainbow",
    output_suffix: str = "",
) -> None:
    """Generates model output plots (maps) for an image and saves them to disk.

    Args:
        output_folder: The folder in which the plots should be saved.
        image_name: The name of the image for which the plots were generated.
        bodypart_names: The names of bodyparts the model outputs.
        bodyparts_to_plot: The names of bodyparts that should be plot.
        image: Shape (height, width, channels). The image on which the model was run.
        scmap: Shape (height, width, num_bodyparts). The scoremaps output by the model.
        locref: Shape (height, width, num_bodyparts, 2). Optionally, the location
            refinement fields output by the model.
        paf: Shape (height, width, 2 * len(paf_graph)). Optionally, the part-affinity
            fields output by the model.
        paf_graph: Must be set if paf is not None. The PAF graph used to assemble.
        paf_all_in_one: Whether to plot all PAFs in a single image.
        paf_colormap: The colormap to use for the PAF maps.
        output_suffix: The filename suffix for the maps to output.
    """
    def _filename(map_name) -> str:
        return f"{image_name}_{map_name}_{output_suffix}.png"

    to_plot = [
        i for i, bpt in enumerate(bodypart_names) if bpt in bodyparts_to_plot
    ]
    if len(to_plot) > 1:
        map_ = scmap[:, :, to_plot].sum(axis=2)
    elif len(to_plot) == 1 and len(bodypart_names) > 1:
        map_ = scmap[:, :, to_plot[0]]
    else:
        map_ = scmap[..., 0]

    fig1, _ = visualize_scoremaps(image, map_)
    fig1.savefig(output_folder / _filename("scmap"))

    if locref is not None:
        if len(to_plot) > 1:
            map_ = scmap[:, :, to_plot]
            locref_x_ = locref[:, :, to_plot, 0]
            locref_y_ = locref[:, :, to_plot, 1]
            # only get the locref fields around their respective detections
            locref_x_[map_ < 0.5] = 0
            locref_y_[map_ < 0.5] = 0
            # combine locrefs
            map_ = map_.sum(axis=2)
            locref_x_ = locref_x_.sum(axis=2)
            locref_y_ = locref_y_.sum(axis=2)
        elif len(to_plot) == 1 and len(bodypart_names) > 1:
            locref_x_ = locref[:, :, to_plot[0], 0]
            locref_y_ = locref[:, :, to_plot[0], 1]
        else:
            locref_x_ = locref[..., 0]
            locref_y_ = locref[..., 1]

        fig2, _ = visualize_locrefs(image, map_, locref_x_, locref_y_)
        fig2.savefig(output_folder / _filename("locref"))

    if paf is not None:
        if paf_graph is None:
            raise ValueError(f"When plotting the PAF, you must pass the ``paf_graph``")

        edge_list = []
        for n, edge in enumerate(paf_graph):
            if any(ind in to_plot for ind in edge):
                e0, e1 = edge
                edge_list.append(
                    [(2 * n, 2 * n + 1), (bodypart_names[e0], bodypart_names[e1])]
                )

        if paf_all_in_one:
            inds = [elem[0] for elem in edge_list]
            n_inds = len(inds)
            cmap = plt.cm.get_cmap(paf_colormap, n_inds)
            colors = cmap(range(n_inds))
            fig3, _ = visualize_paf(image, paf[:, :, inds], colors=colors)
            fig3.savefig(output_folder / _filename("paf"))
        else:
            for inds, names in edge_list:
                fig3, _ = visualize_paf(image, paf[:, :, [inds]])
                fig3.savefig(output_folder / _filename(f"paf_{'_'.join(names)}"))

    plt.close("all")
