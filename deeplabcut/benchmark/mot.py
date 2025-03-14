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

import warnings

import motmetrics as mm
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from deeplabcut.core import trackingutils


def convert_bboxes_to_xywh(bboxes: NDArray, inplace: bool = False) -> NDArray:
    """
    Converts bounding box coordinates from [x_min, y_min, x_max, y_max] format
    to [x, y, width, height] format.

    Parameters
    ----------
    bbox : numpy.ndarray
        A 2D array of shape (N, M), where N is the number of bounding boxes
        and M >= 4. The first four columns represent the bounding box in the format
        [x_min, y_min, x_max, y_max].
    inplace : bool, optional
        If True, modifies the input array in place. If False, returns a copy of
        the array with the converted bounding box format. Defaults to False.

    Returns
    -------
    numpy.ndarray or None
        If `inplace` is False, returns a new array of the same shape as `bbox`
        with the format [x, y, width, height]. If `inplace` is True, the input
        array is modified directly, and nothing is returned.
    """
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    if not inplace:
        new_bboxes = bboxes.copy()
        new_bboxes[:, 2] = w
        new_bboxes[:, 3] = h
        return new_bboxes
    bboxes[:, 2] = w
    bboxes[:, 3] = h

_convert_bboxes_to_xywh = convert_bboxes_to_xywh


def reconstruct_bboxes_from_bodyparts(
    data: pd.DataFrame, margin: float, to_xywh: bool = False
) -> NDArray:
    """
    Reconstructs bounding boxes from body part coordinates and likelihoods.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing body part data with a multi-level column index.
        The expected levels include 'x', 'y', and 'likelihood', where:
        - 'x' and 'y' contain the coordinates of the body parts.
        - 'likelihood' contains the confidence scores for each body part.
    margin : float
        The margin to add/subtract from the minimum/maximum coordinates when defining the bounding box.
    to_xywh : bool, optional
        If True, converts the bounding box format from [x_min, y_min, x_max, y_max]
        to [x, y, width, height]. Defaults to False.

    Returns
    -------
    numpy.ndarray
        An array of shape (N, 5), where N is the number of rows in `data`.
        Each row represents a bounding box with the following values:
        - [x_min, y_min, x_max, y_max, likelihood]
        If `to_xywh` is True, the format will be [x, y, width, height, likelihood].

    Notes
    -----
    - NaN values in the input data are ignored when computing the bounding box dimensions.
    - Warnings related to NaN values are suppressed during calculations.
    """
    x = data.xs("x", axis=1, level="coords")
    y = data.xs("y", axis=1, level="coords")
    p = data.xs("likelihood", axis=1, level="coords")
    xy = np.stack([x, y], axis=2)
    bboxes = np.full((data.shape[0], 5), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bboxes[:, :2] = np.nanmin(xy, axis=1) - margin
        bboxes[:, 2:4] = np.nanmax(xy, axis=1) + margin
        bboxes[:, 4] = np.nanmean(p, axis=1)
    if to_xywh:
        convert_bboxes_to_xywh(bboxes, inplace=True)
    return bboxes


def reconstruct_all_bboxes(
    data: pd.DataFrame, margin: float, to_xywh: bool = False
) -> NDArray:
    """
    Reconstructs bounding boxes for multiple individuals from body part data.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing body part data with a multi-level column index.
        The expected levels include:
        - 'individuals': Names of the individuals (e.g., animals).
        - 'x', 'y', and 'likelihood': Coordinate and confidence data for body parts.
    margin : float
        The margin to add/subtract from the minimum/maximum coordinates when defining the bounding box.
    to_xywh : bool
        If True, converts the bounding box format from [x_min, y_min, x_max, y_max]
        to [x, y, width, height].

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (A, F, 5), where:
        - A is the number of individuals (excluding 'single', if present).
        - F is the number of frames (rows) in the input `data`.
        - Each bounding box is represented as [x_min, y_min, x_max, y_max, likelihood].
          If `to_xywh` is True, the format will be [x, y, width, height, likelihood].

    Notes
    -----
    - Individuals are extracted from the 'individuals' level of the DataFrame columns.
    - If an individual named 'single' exists, it is excluded from the bounding box computation.
    - NaN values in the input data are ignored during calculations.
    """
    animals = data.columns.get_level_values("individuals").unique().tolist()
    try:
        animals.remove("single")
    except ValueError:
        pass
    bboxes = np.full((len(animals), data.shape[0], 5), np.nan)
    for n, animal in enumerate(animals):
        bboxes[n] = reconstruct_bboxes_from_bodyparts(
            data.xs(animal, axis=1, level="individuals"), margin, to_xywh
        )
    return bboxes


def compute_mot_metrics(
    h5_file_gt: str,
    h5_file_pred: str,
    tracker_type: str = "bbox",
    **kwargs,
) -> mm.MOTAccumulator:
    df_gt = pd.read_hdf(h5_file_gt)
    df = pd.read_hdf(h5_file_pred)
    if tracker_type == "bbox":
        func = reconstruct_all_bboxes
    elif tracker_type == "ellipse":
        func = trackingutils.reconstruct_all_ellipses
    else:
        raise ValueError(f"Unrecognized tracker type {tracker_type}.")

    trackers_gt = func(df_gt, **kwargs)
    trackers = func(df, **kwargs)
    return _compute_mot_metrics(
        trackers_gt, trackers, tracker_type,
    )


def _compute_mot_metrics(
    trackers_ground_truth: NDArray,
    trackers: NDArray,
    tracker_type: str = "bbox",
) -> mm.MOTAccumulator:
    if trackers_ground_truth.shape != trackers.shape:
        raise ValueError(
            "Dimensions mismatch. There must be as many `trackers_ground_truth` as there are `trackers`."
        )

    if tracker_type == "bbox":
        sl = slice(0, 4)
        cost_func = mm.distances.iou_matrix
    elif tracker_type == "ellipse":
        sl = slice(0, 5)

        def cost_func(ellipses_gt, ellipses_hyp):
            cost_matrix = np.zeros((len(ellipses_gt), len(ellipses_hyp)))
            gt_el = [trackingutils.Ellipse(*e[:5]) for e in ellipses_gt]
            hyp_el = [trackingutils.Ellipse(*e[:5]) for e in ellipses_hyp]
            for i, el in enumerate(gt_el):
                for j, tracker in enumerate(hyp_el):
                    cost_matrix[i, j] = 1 - el.calc_similarity_with(tracker)
            return cost_matrix

    else:
        raise ValueError(f"Unrecognized tracker type {tracker_type}.")

    ids = np.arange(trackers_ground_truth.shape[0])
    acc = mm.MOTAccumulator(auto_id=True)
    for i in range(trackers_ground_truth.shape[1]):
        trackers_gt = trackers_ground_truth[:, i, sl]
        trackers_hyp = trackers[:, i, sl]
        empty_gt = np.isnan(trackers_gt).any(axis=1)
        empty_hyp = np.isnan(trackers_hyp).any(axis=1)
        trackers_gt = trackers_gt[~empty_gt]
        trackers_hyp = trackers_hyp[~empty_hyp]
        cost = cost_func(trackers_gt, trackers_hyp)
        acc.update(ids[~empty_gt], ids[~empty_hyp], cost)
    return acc


def print_all_metrics(
    accumulators: list[mm.MOTAccumulator], all_params: list[str] | None = None
):
    if not all_params:
        names = [f"iter{i + 1}" for i in range(len(accumulators))]
    else:
        s = "_".join("{}" for _ in range(len(all_params[0])))
        names = [s.format(*params.values()) for params in all_params]
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators, metrics=mm.metrics.motchallenge_metrics, names=names
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    return summary
