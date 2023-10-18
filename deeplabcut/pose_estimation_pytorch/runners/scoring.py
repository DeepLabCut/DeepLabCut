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

import numpy as np
import pandas as pd

from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import (
    Assembly,
    evaluate_assembly,
)


def get_scores(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    pcutoff: float | None = None,
    bodyparts: list[str] | None = None,
) -> dict[str, float]:
    """Computes for the different scores given the grount truth and the predictions.

    The different scores computed are based on the COCO metrics: https://cocodataset.org/#keypoints-eval
    RMSE (Root Mean Square Error)
    OKS mAP (Mean Average Precision)
    OKS mAR (Mean Average Recall)

    Args:
        prediction: prediction df, should already be matched to ground truth using
            Hungarian Algorithm (Ref: https://brilliant.org/wiki/hungarian-matching/)
        target: ground truth dataframe
        pcutoff: the value used to compute the pcutoff scores
        bodyparts: names of the bodyparts. Defaults to None.

    Returns:
        scores: A dictionary of scores containign the following keys:
                      ['rmse', 'rmse_pcutoff', 'mAP', 'mAR', 'mAP_pcutoff', 'mAR_pcutoff']

    Examples:
        >>> # Define the p-cutoff, prediction, and target DataFrames
        >>> pcutoff = 0.5
        >>> prediction = pd.DataFrame(...)  # Your DataFrame here
        >>> target = pd.DataFrame(...)      # Your DataFrame here
        >>> # Compute the scores
        >>> scores = get_scores(prediction, target, pcutoff)
        >>> print(scores)
        {
            'rmse': 0.156,
            'rmse_pcutoff': 0.115,
            'mAP': 84.2,
            'mAR': 74.5,
            'mAP_pcutoff': 91.3,
            'mAR_pcutoff': 82.5
        }  # Sample output scores
    """
    if pcutoff is None:
        pcutoff = -1

    rmse, rmse_p = get_rmse(prediction, target, pcutoff=pcutoff, bodyparts=bodyparts)
    oks, oks_p = get_oks(prediction, target, pcutoff=pcutoff, bodyparts=bodyparts)
    return {
        "rmse": np.nanmean(rmse),
        "rmse_pcutoff": np.nanmean(rmse_p),
        "mAP": 100 * oks["mAP"],
        "mAR": 100 * oks["mAR"],
        "mAP_pcutoff": 100 * oks_p["mAP"],
        "mAR_pcutoff": 100 * oks_p["mAR"],
    }


def get_rmse(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    pcutoff: float = -1,
    bodyparts: list[str] | None = None,
) -> tuple[float, float]:
    """Computes the root mean square error (rmse) for predictions vs the ground truth labels

    Assumes hungarian algorithm matching (https://brilliant.org/wiki/hungarian-matching/))
    has already be applied to match predicted animals and ground truth ones.

    Args:
        prediction: prediction dataframe
        target: target dataframe
        pcutoff: Confidence lower bound for a keypoint to be considered as detected.
                                    Defaults to -1.
        bodyparts: list of the bodyparts names. Defaults to None.

    Returns:
        rmse: rmse without cutoff
        rmse_p : rmse with cutoff

    Example:
        >>> # Define the prediction and target DataFrames
        >>> prediction = pd.DataFrame(...)  # Your DataFrame here
        >>> target = pd.DataFrame(...)      # Your DataFrame here
        >>> # Compute the RMSE values
        >>> rmse, rmse_pcutoff = get_rmse(prediction, target, pcutoff=0.5)
        >>> print(rmse, rmse_pcutoff)
        0.145 0.105  # Sample output RMSE values
    """
    scorer_pred = prediction.columns[0][0]
    scorer_target = target.columns[0][0]
    mask = prediction[scorer_pred].xs("likelihood", level=2, axis=1) >= pcutoff
    if bodyparts:
        diff = (
            target[scorer_target][bodyparts] - prediction[scorer_pred][bodyparts]
        ) ** 2
    else:
        diff = (target[scorer_target] - prediction[scorer_pred]) ** 2
    mse = diff.xs("x", level=2, axis=1) + diff.xs("y", level=2, axis=1)
    rmse = np.sqrt(mse)
    rmse_p = np.sqrt(mse[mask])

    return rmse, rmse_p


def get_oks(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    oks_sigma=0.1,
    margin=0,
    symmetric_kpts=None,
    pcutoff: float = -1,
    bodyparts: list[str] | None = None,
) -> tuple[dict, dict]:
    """Computes the object keypoint similarity (OKS) scores for predictions.

    OKS is defined in https://cocodataset.org/#keypoints-eval

    Args:
        prediction: prediction dataframe
        target: target dataframe
        oks_sigma: Sigma for oks conputation. Defaults to 0.1.
        margin: margin used for bbox computation. Defaults to 0.
        symmetric_kpts: Not supported yet. Defaults to None.
        pcutoff: Confidence lower bound for a keypoint to be considered as detected.
                                    Defaults to -1.
        bodyparts: list of the bodyparts names. Defaults to None.

    Returns:
        oks_raw: oks scores without p_cutoff
        oks_pcutoff: oks scores with pcutoff

    Examples:
        >>> # Define the prediction and target DataFrames
        >>> prediction = pd.DataFrame(...)  # Your DataFrame here
        >>> target = pd.DataFrame(...)      # Your DataFrame here
        >>> # Compute the OKS scores
        >>> oks, oks_pcutoff = get_oks(prediction, target, oks_sigma=0.2, pcutoff=0.5)
        >>> print(oks, oks_pcutoff)
        {'mAP': 0.842, 'mAR': 0.745} {'mAP': 0.913, 'mAR': 0.825}  # Sample output OKS scores
    """

    scorer_pred = prediction.columns[0][0]
    scorer_target = target.columns[0][0]

    if bodyparts is not None:
        idx_slice = pd.IndexSlice[:, :, bodyparts, :]
        prediction = prediction.loc[:, idx_slice]
        target = target.loc[:, idx_slice]
    mask = prediction[scorer_pred].xs("likelihood", level=2, axis=1) >= pcutoff

    # Convert predictions to DLC assemblies
    assemblies_pred_raw, unique_pred_raw = conv_df_to_assemblies(
        prediction[scorer_pred]
    )
    assemblies_gt_raw, unique_gt_raw = conv_df_to_assemblies(target[scorer_target])

    assemblies_pred_masked, unique_pred_masked = conv_df_to_assemblies(
        prediction[scorer_pred][mask]
    )
    oks_assemblies_raw = evaluate_assembly(
        assemblies_pred_raw,
        assemblies_gt_raw,
        oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
    )
    if unique_pred_raw is not None and unique_gt_raw is not None:
        oks_unique_raw = evaluate_assembly(
            unique_pred_raw,
            unique_gt_raw,
            oks_sigma,
            margin=margin,
            symmetric_kpts=symmetric_kpts,
        )

    oks_pcutoff = evaluate_assembly(
        assemblies_pred_masked,
        assemblies_gt_raw,
        oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
    )
    if unique_pred_masked is not None and unique_gt_raw is not None:
        oks_unique_masked = evaluate_assembly(
            unique_pred_masked,
            unique_gt_raw,
            oks_sigma,
            margin=margin,
            symmetric_kpts=symmetric_kpts,
        )

    return oks_assemblies_raw, oks_pcutoff


def conv_df_to_assemblies(df: pd.DataFrame) -> tuple[dict, dict | None]:
    """
    Convert a dataframe to an assemblies dictionary

    Args:
        df : dataframe of coordinates/predictions,
        df is expected to have a multi_index of shape (num_animals, num_keypoints, 2 or 3)
    Returns:
        assemblies: dictionary of the assemblies of keypoints
        if there are unique bodyparts, a dictionary containing unique bodyparts
    """
    individuals = df.columns.get_level_values(0)
    df_bodyparts = df.loc[:, individuals != "single"]
    assemblies = _df_to_dict(df_bodyparts)

    unique_keypoints = None
    if "single" in individuals:
        df_unique = df.loc[:, individuals == "single"]
        unique_keypoints = _df_to_dict(df_unique)

    return assemblies, unique_keypoints


def _df_to_dict(df: pd.DataFrame) -> dict:
    data = {}
    num_animals = len(df.columns.get_level_values(0).unique())
    num_kpts = len(df.columns.get_level_values(1).unique())
    for image_path in df.index:
        row = df.loc[image_path].to_numpy()
        row = row.reshape(num_animals, num_kpts, -1)

        kpt_lst = []
        for i in range(num_animals):
            ass = Assembly.from_array(row[i])
            if len(ass):
                kpt_lst.append(ass)

        data[image_path] = kpt_lst

    return data
