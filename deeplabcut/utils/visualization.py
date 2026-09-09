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

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from skimage import color, io
from tqdm import trange

from deeplabcut.utils import auxfun_videos, auxiliaryfunctions

PlotMode = Literal["bodypart", "individual"]
BoundingBoxColor = Colormap | str | None

logger = logging.getLogger(__name__)


def get_cmap(n: int, name: str = "hsv") -> Colormap:
    """Get the cmap.

    Args:
        n: number of distinct colors
        name: name of matplotlib colormap

    Returns:
         A function that maps each index in 0, 1, ..., n-1 to a distinct
         RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.get_cmap(name, n)


def make_labeled_image(
    frame,
    DataCombined,
    imagenr,
    pcutoff,
    Scorers,
    bodyparts,
    colors,
    cfg,
    labels=None,
    scaling=1,
    ax=None,
):
    """Creating a labeled image with the original human labels, as well as the
    DeepLabCut's!
    """
    if labels is None:
        labels = ["+", ".", "x"]
    alphavalue = cfg["alphavalue"]  # .5
    dotsize = cfg["dotsize"]  # =15

    if ax is None:
        h, w = np.shape(frame)[:2]
        _, ax = prepare_figure_axes(w, h, scaling)
    ax.imshow(frame, "gray")
    for loopscorer in Scorers:
        for bpindex, bp in enumerate(bodyparts):
            if np.isfinite(
                DataCombined[loopscorer][bp]["y"].iloc[imagenr] + DataCombined[loopscorer][bp]["x"].iloc[imagenr]
            ):
                y, x = (
                    int(DataCombined[loopscorer][bp]["y"].iloc[imagenr]),
                    int(DataCombined[loopscorer][bp]["x"].iloc[imagenr]),
                )
                if cfg["scorer"] not in loopscorer:
                    p = DataCombined[loopscorer][bp]["likelihood"].iloc[imagenr]
                    if p > pcutoff:
                        ax.plot(
                            x,
                            y,
                            labels[1],
                            ms=dotsize,
                            alpha=alphavalue,
                            color=colors(int(bpindex)),
                        )
                    else:
                        ax.plot(
                            x,
                            y,
                            labels[2],
                            ms=dotsize,
                            alpha=alphavalue,
                            color=colors(int(bpindex)),
                        )
                else:  # this is the human labeler
                    ax.plot(
                        x,
                        y,
                        labels[0],
                        ms=dotsize,
                        alpha=alphavalue,
                        color=colors(int(bpindex)),
                    )
    return ax


def make_multianimal_labeled_image(
    frame: np.ndarray,
    coords_truth: np.ndarray | list,
    coords_pred: np.ndarray | list,
    probs_pred: np.ndarray | list,
    colors: Colormap,
    dotsize: float | int = 12,
    alphavalue: float = 0.7,
    pcutoff: float = 0.6,
    labels: list = None,
    ax: plt.Axes | None = None,
    bounding_boxes: tuple[np.ndarray, np.ndarray] | None = None,
    bboxes_cutoff: float = 0.6,
    bboxes_color: Colormap | str | None = None,
    color_offset: int = 0,
) -> plt.Axes:
    """Plots groundtruth labels and predictions onto the matplotlib's axes, with the
    specified graphical parameters.

    Args:
        frame: image
        coords_truth: groundtruth labels
        coords_pred: predictions
        probs_pred: prediction probabilities
        colors: colors for poses
        dotsize: size of dot
        alphavalue: transparency for the keypoints
        pcutoff: cut-off confidence value
        labels: labels to use for ground truth, reliable predictions, and not reliable predictions (confidence below
        cut-off value)
        ax: matplotlib plot's axes object
        bounding_boxes: bounding boxes (top-left corner, size) and their respective confidence levels,
        bboxes_cutoff: bounding boxes confidence cutoff threshold.
        bboxes_color: color(s) for the bounding boxes.
            If Colormap is passed -> each bounding box will be colored into its own color from the colormap.
            If string is passed -> all bboxes will be of string's defined color.
            If None -> all bboxes will be colored into a default color.
        color_offset: Index offset applied when selecting colors from the colormap.

    Returns:
        matplotlib Axes object with plotted labels and predictions.
    """
    if labels is None:
        labels = ["+", ".", "x"]
    if ax is None:
        h, w = frame.shape[:2]
        _, ax = prepare_figure_axes(w, h)
    ax.imshow(frame, "gray")

    if bounding_boxes is not None:
        for i, (bbox, bbox_score) in enumerate(zip(bounding_boxes[0], bounding_boxes[1], strict=False)):
            bbox_origin = (bbox[0], bbox[1])
            (bbox_width, bbox_height) = (bbox[2], bbox[3])
            if isinstance(bboxes_color, Colormap):
                bbox_color = bboxes_color(i)
            elif bboxes_color is None:
                bbox_color = "red"
            else:
                bbox_color = bboxes_color
            rectangle = patches.Rectangle(
                bbox_origin,
                bbox_width,
                bbox_height,
                linewidth=1,
                edgecolor=bbox_color,
                facecolor="none",
                linestyle="--" if bbox_score < bboxes_cutoff else "-",
            )
            ax.add_patch(rectangle)

    for n, data in enumerate(zip(coords_truth, coords_pred, probs_pred, strict=False)):
        color = colors(n + color_offset)
        coord_gt, coord_pred, prob_pred = data

        ax.plot(*coord_gt.T, labels[0], ms=dotsize, alpha=alphavalue, color=color)
        if not coord_pred.shape[0]:
            continue

        reliable = np.repeat(prob_pred >= pcutoff, coord_pred.shape[1], axis=1)
        ax.plot(
            *coord_pred[reliable[:, 0]].T,
            labels[1],
            ms=dotsize,
            alpha=alphavalue,
            color=color,
        )
        if not np.all(reliable):
            ax.plot(
                *coord_pred[~reliable[:, 0]].T,
                labels[2],
                ms=dotsize,
                alpha=alphavalue,
                color=color,
            )
    return ax


def plot_and_save_labeled_frame(
    DataCombined,
    ind,
    trainIndices,
    cfg,
    colors,
    comparisonbodyparts,
    DLCscorer,
    foldername,
    fig,
    ax,
    scaling=1,
):
    if isinstance(DataCombined.index[ind], tuple):
        image_path = Path(cfg["project_path"]).joinpath(*DataCombined.index[ind])
    else:
        image_path = Path(cfg["project_path"]) / DataCombined.index[ind]
    frame = io.imread(os.fspath(image_path))
    h, w = np.shape(frame)[:2]
    fig.set_size_inches(w / 100, h / 100)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax = make_labeled_image(
        frame,
        DataCombined,
        ind,
        cfg["pcutoff"],
        [cfg["scorer"], DLCscorer],
        comparisonbodyparts,
        colors,
        cfg,
        scaling=scaling,
        ax=ax,
    )
    save_labeled_frame(fig, image_path, Path(foldername), ind in trainIndices)
    return ax


def save_labeled_frame(
    fig,
    image_path: Path,
    dest_folder: Path,
    belongs_to_train: bool,
) -> None:
    """Save the labeled frame to disk.

    Note: folder creation is handled upstream.
    This function assumes that the destination folder already exists.
    """
    imagename = image_path.parts[-1]
    imfoldername = image_path.parts[-2]
    if belongs_to_train:
        dest = "-".join(("Training", imfoldername, imagename))
    else:
        dest = "-".join(("Test", imfoldername, imagename))
    full_path = os.fspath(dest_folder / dest)

    # Windows throws error if file path is > 260 characters, can fix with prefix.
    # See https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
    if len(full_path) >= 260 and os.name == "nt":
        full_path = "\\\\?\\" + full_path
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(full_path)


def create_minimal_figure(dpi=100):
    fig, ax = plt.subplots(frameon=False, dpi=dpi)
    ax.axis("off")
    ax.invert_yaxis()
    return fig, ax


def erase_artists(ax):
    for artist in [*ax.lines, *ax.collections, *ax.artists, *ax.patches, *ax.images]:
        artist.remove()
    ax.figure.canvas.draw_idle()


def prepare_figure_axes(width, height, scale=1.0, dpi=100):
    fig = plt.figure(frameon=False, figsize=(width * scale / dpi, height * scale / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    return fig, ax


def make_labeled_images_from_dataframe(
    df,
    cfg,
    destfolder=None,
    scale=1.0,
    dpi=100,
    keypoint="+",
    draw_skeleton=True,
    color_by="bodypart",
):
    """Write labeled frames to disk from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the labeled data.
        cfg (dict): Project configuration.
        destfolder (str or Path, optional): Destination folder for labeled images.
        scale (float, optional): Output dimension scaling factor.
        dpi (int, optional): Output resolution.
        keypoint (str, optional): Matplotlib marker used for keypoints.
        draw_skeleton (bool, optional): Whether to draw the configured skeleton.
        color_by (str, optional): Either "bodypart" or "individual".
    """
    columns = df.columns
    bodypart_columns = columns.get_level_values("bodyparts")
    bodypart_names = bodypart_columns.unique()
    bodyparts = bodypart_columns[::2]

    colors = _get_labeled_image_colors(
        columns=columns,
        bodyparts=bodyparts,
        bodypart_names=bodypart_names,
        color_by=color_by,
        colormap=cfg["colormap"],
    )

    should_draw_skeleton = bool(draw_skeleton and cfg["skeleton"])
    ind_bones = _get_bone_indices(bodyparts, cfg["skeleton"]) if should_draw_skeleton else ()

    images_list = [str(Path(cfg["project_path"]).joinpath(*index)) for index in df.index.tolist()]

    # Preserve list.index() behavior by retaining the first occurrence.
    image_indices = {}
    for index, filename in enumerate(images_list):
        image_indices.setdefault(filename, index)

    destfolder = Path(images_list[0]).parent if destfolder is None else Path(destfolder)
    tmpfolder = destfolder.parent / f"{destfolder.name}_labeled"
    auxiliaryfunctions.attempt_to_make_folder(tmpfolder)

    images = io.imread_collection(images_list)
    all_same_shape = _images_have_same_shape(images)

    xy = df.values.reshape(df.shape[0], -1, 2)
    segments = xy[:, ind_bones].swapaxes(1, 2)

    marker_size = cfg["dotsize"]
    alpha = cfg["alphavalue"]
    skeleton_color = cfg["skeleton_color"]

    def output_path(filename):
        stem = Path(filename).stem
        out_name = f"{stem}_{color_by}.png"
        return tmpfolder / out_name

    def save_figure(fig, filename):
        fig.subplots_adjust(
            left=0,
            bottom=0,
            right=1,
            top=1,
            wspace=0,
            hspace=0,
        )
        fig.savefig(output_path(filename), dpi=dpi)

    if all_same_shape:
        h, w = images[0].shape[:2]
        fig, ax = prepare_figure_axes(w, h, scale, dpi)

        image_artist = ax.imshow(np.zeros((h, w)), "gray")
        point_artists = [
            ax.plot(
                [],
                [],
                keypoint,
                ms=marker_size,
                alpha=alpha,
                color=color,
            )[0]
            for color in colors
        ]
        skeleton_artist = LineCollection(
            [],
            colors=skeleton_color,
            alpha=alpha,
        )
        ax.add_collection(skeleton_artist)

        for i in trange(len(images)):
            filename = images.files[i]
            index = image_indices[filename]
            image = images[i]

            if image.ndim == 2 or image.shape[-1] == 1:
                image = color.gray2rgb(image)

            image_artist.set_data(image)

            for artist, coord in zip(point_artists, xy[index], strict=False):
                artist.set_data(*np.expand_dims(coord, axis=1))

            if ind_bones:
                skeleton_artist.set_segments(segments[index])

            save_figure(fig, filename)

        plt.close(fig)
        return

    for i in trange(len(images)):
        filename = images.files[i]
        index = image_indices[filename]
        image = images[i]
        h, w = image.shape[:2]

        fig, ax = prepare_figure_axes(w, h, scale, dpi)
        ax.imshow(image)

        for coord, point_color in zip(xy[index], colors, strict=False):
            ax.plot(
                *coord,
                keypoint,
                ms=marker_size,
                alpha=alpha,
                color=point_color,
            )

        if ind_bones:
            ax.add_collection(
                LineCollection(
                    segments[index],
                    colors=skeleton_color,
                    alpha=alpha,
                )
            )

        save_figure(fig, filename)
        plt.close(fig)


def _get_labeled_image_colors(
    columns,
    bodyparts,
    bodypart_names,
    color_by,
    colormap,
):
    """Return one color per keypoint column."""
    if color_by == "bodypart":
        names = bodypart_names
        values = bodyparts
    elif color_by == "individual":
        try:
            individual_columns = columns.get_level_values("individuals")
        except KeyError as exc:
            raise ValueError("Coloring by individuals requires an 'individuals' column level") from exc

        names = individual_columns.unique()
        values = individual_columns[::2]
    else:
        raise ValueError("`color_by` must be either `bodypart` or `individual`.")

    name_to_index = {name: index for index, name in enumerate(names)}
    color_indices = values.map(name_to_index)
    return get_cmap(len(names), colormap)(color_indices)


def _get_bone_indices(bodyparts, skeleton):
    """Return transposed endpoint indices for all configured bones."""
    positions_by_bodypart = {}

    for index, bodypart in enumerate(bodyparts):
        positions_by_bodypart.setdefault(bodypart, []).append(index)

    bones = []
    for bodypart1, bodypart2 in skeleton:
        # Preserve the original if/elif behavior for identical endpoints.
        if bodypart1 == bodypart2:
            continue

        bones.extend(
            zip(
                positions_by_bodypart.get(bodypart1, ()),
                positions_by_bodypart.get(bodypart2, ()),
                strict=False,
            )
        )

    return tuple(zip(*bones, strict=False))


def _images_have_same_shape(images):
    """Return whether all images have the same height and width."""
    expected_shape = images[0].shape[:2]
    return all(image.shape[:2] == expected_shape for image in images[1:])


def plot_evaluation_results(
    df_combined: pd.DataFrame,
    project_root: Path,
    scorer: str,
    model_name: str,
    output_folder: Path,
    in_train_set: bool,
    plot_unique_bodyparts: bool = False,
    mode: PlotMode = "bodypart",
    colormap: str = "rainbow",
    dot_size: int = 12,
    alpha_value: float = 0.7,
    p_cutoff: float = 0.6,
    bounding_boxes: dict | None = None,
    bboxes_cutoff: float = 0.6,
    bounding_boxes_color: BoundingBoxColor = "auto",
) -> None:
    """Creates labeled images using the results of inference, and saves them to an
    output folder.

    Args:
        df_combined: dataframe with multiindex rows ("labeled-data", video_name,
            image_name) and columns ("scorer", "individuals", "bodyparts", "coords").
            There should be two scorers: scorer (for ground truth data) and model_name
            (for prediction data)
        project_root: the project root directory
        scorer: the name of the scorer for ground truth data in df_combined
        model_name: the name of the model for predictions in df_combined
        output_folder: the directory where images should be saved
        in_train_set: whether df_combined is for train set images
        plot_unique_bodyparts: whether we should plot unique bodyparts
        mode: one of {"bodypart", "individual"}. Determines the keypoint color grouping
        colormap: the colormap to use for keypoints
        dot_size: the dot size to use for keypoints
        alpha_value: the alpha value to use for keypoints
        p_cutoff: the p-cutoff for "confident" keypoints
        bounding_boxes: dictionary with df_combined rows as keys and bounding boxes
            (np array for coordinates and np array for confidence).
            None corresponds to no bounding boxes.
        bboxes_cutoff: bounding boxes confidence cutoff threshold.
        bounding_boxes_color: If plotting bounding boxes, this is the color that will be used for bounding boxes.
            If set to "auto" (default value):
                - if mode is "bodypart", the bbox color will be a default color
                - if mode is "individual", each individual's color will be used for its bounding box
    """
    if bounding_boxes is None:
        bounding_boxes = {}

    if mode not in {"bodypart", "individual"}:
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'bodypart' or 'individual'.")

    for row_index, row in df_combined.iterrows():
        plot_unique_for_row = plot_unique_bodyparts
        if isinstance(row_index, str):
            image_rel_path = Path(row_index)
            data_folder = image_rel_path.parent.parent.name
            video = image_rel_path.parent.name
            image = image_rel_path.name
        else:
            data_folder, video, image = row_index

        image_path = project_root / data_folder / video / image
        frame = auxfun_videos.imread(str(image_path), mode="skimage")

        row_multi = row.loc[row.index.get_level_values("individuals") != "single"]

        df_gt = row_multi[scorer]
        df_predictions = row_multi[model_name]

        gt_individuals = df_gt.index.get_level_values("individuals").unique()
        pred_individuals = df_predictions.index.get_level_values("individuals").unique()

        gt_bodyparts = df_gt.index.get_level_values("bodyparts").unique()
        pred_bodyparts = df_predictions.index.get_level_values("bodyparts").unique()

        if len(gt_individuals) != len(pred_individuals):
            logger.warning(
                f"Warning: Individual count mismatch for {image}\n"
                f"  Ground truth individual count: {len(gt_individuals)}\n"
                f"  Predictions individual count: {len(pred_individuals)}\n"
                "  Skipping visualization for this image"
            )
            continue

        if list(gt_bodyparts) != list(pred_bodyparts):  # keep ordering of bodyparts
            logger.warning(
                f"Warning: Bodypart mismatch for {image}\n"
                f"  Ground truth: {list(gt_bodyparts)}\n"
                f"  Predictions: {list(pred_bodyparts)}\n"
                "  Skipping visualization for this image"
            )
            continue

        individuals = len(gt_individuals)
        bodyparts = len(gt_bodyparts)

        # Shape (num_individuals, num_bodyparts, xy)
        try:
            ground_truth = df_gt.to_numpy().reshape((individuals, bodyparts, 2))
            predictions = df_predictions.to_numpy().reshape((individuals, bodyparts, 3))
        except ValueError:
            # Handle cases where the actual data size doesn't match expected shape
            actual_size_gt = df_gt.size
            actual_size_pred = df_predictions.size
            expected_size_gt = individuals * bodyparts * 2
            expected_size_pred = individuals * bodyparts * 3

            logger.warning(
                f"Warning: DataFrame reshape failed for {image}\n"
                f"  Expected: {individuals} individual(s), {bodyparts} bodypart(s)\n"
                f"  Ground truth: {actual_size_gt} elements (expected {expected_size_gt})\n"
                f"  Predictions: {actual_size_pred} elements (expected {expected_size_pred})\n"
                "  Skipping visualization for this image"
            )
            continue

        bboxes = bounding_boxes.get(row_index)

        if plot_unique_for_row:
            row_unique = row.loc[row.index.get_level_values("individuals") == "single"]
            if row_unique.empty:
                plot_unique_for_row = False
            else:
                unique_gt = row_unique[scorer]
                unique_pred = row_unique[model_name]

                gt_unique_bodyparts = unique_gt.index.get_level_values("bodyparts").unique()
                pred_unique_bodyparts = unique_pred.index.get_level_values("bodyparts").unique()

                if list(gt_unique_bodyparts) != list(pred_unique_bodyparts):
                    logger.warning(f"Warning: Unique bodypart mismatch for {image}, skipping unique bodyparts")
                    plot_unique_for_row = False
                else:
                    unique_bodyparts = len(gt_unique_bodyparts)

                    try:
                        unique_ground_truth = unique_gt.to_numpy().reshape((1, unique_bodyparts, 2))
                        unique_predictions = unique_pred.to_numpy().reshape((1, unique_bodyparts, 3))
                    except ValueError:
                        # Handle cases where unique bodyparts reshape fails
                        logger.warning(
                            f"Warning: Unique bodyparts reshape failed for {image}, skipping unique bodyparts"
                        )
                        plot_unique_for_row = False

        fig, ax = create_minimal_figure()
        try:
            h, w = frame.shape[:2]
            fig.set_size_inches(w / 100, h / 100)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            # ax.invert_yaxis()

            if mode == "bodypart":
                num_colors = bodyparts
                if plot_unique_for_row:
                    num_colors += unique_bodyparts

                colors = get_cmap(num_colors, name=colormap)
                predictions = predictions.swapaxes(0, 1)
                ground_truth = ground_truth.swapaxes(0, 1)
            else:
                colors = get_cmap(individuals + 1, name=colormap)

            if bounding_boxes_color == "auto":
                bboxes_color = None if mode == "bodypart" else get_cmap(individuals + 1, name=colormap)
            else:
                bboxes_color = bounding_boxes_color

            ax = make_multianimal_labeled_image(
                frame=frame,
                coords_truth=ground_truth,
                coords_pred=predictions[:, :, :2],
                probs_pred=predictions[:, :, 2:],
                colors=colors,
                dotsize=dot_size,
                alphavalue=alpha_value,
                pcutoff=p_cutoff,
                ax=ax,
                bounding_boxes=bboxes,
                bboxes_cutoff=bboxes_cutoff,
                bboxes_color=bboxes_color,
            )
            if plot_unique_for_row:
                if mode == "bodypart":
                    unique_predictions = unique_predictions.swapaxes(0, 1)
                    unique_ground_truth = unique_ground_truth.swapaxes(0, 1)
                ax = make_multianimal_labeled_image(
                    frame=frame,
                    coords_truth=unique_ground_truth,
                    coords_pred=unique_predictions[:, :, :2],
                    probs_pred=unique_predictions[:, :, 2:],
                    colors=colors,
                    color_offset=bodyparts if mode == "bodypart" else individuals,
                    dotsize=dot_size,
                    alphavalue=alpha_value,
                    pcutoff=p_cutoff,
                    ax=ax,
                )

            save_labeled_frame(
                fig,
                image_path,
                output_folder,
                belongs_to_train=in_train_set,
            )
            erase_artists(ax)
        finally:
            plt.close(fig)
