#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import numpy as np
import pickle
from sklearn.metrics import accuracy_score

from deeplabcut.core.crossvalutils import find_closest_neighbors
from deeplabcut.core.inferenceutils import (
    Assembly,
    evaluate_assembly,
)
from deeplabcut.pose_estimation_pytorch.post_processing import (
    rmse_match_prediction_to_gt,
)
from deeplabcut.utils.auxiliaryfunctions import read_config


def get_scores(
    poses: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    unique_bodypart_poses: dict[str, np.ndarray] | None = None,
    unique_bodypart_gt: dict[str, np.ndarray] | None = None,
    pcutoff: float = -1,
    bbox_margin: int = 0,
) -> dict[str, float]:
    """Computes for the different scores given the ground truth and the predictions.

    The poses and ground truth should already be aligned to the ground truth (the scores
    will be computed assuming individual i in the poses matches to individual i in the
    ground truth)

    The different scores computed are based on the COCO metrics: https://cocodataset.org/#keypoints-eval
    RMSE (Root Mean Square Error)
    OKS mAP (Mean Average Precision)
    OKS mAR (Mean Average Recall)

    Args:
        poses: the predicted poses for each image in the format
            {'image': keypoints with shape (num_individuals, num_keypoints, 3)}
        ground_truth: ground truth keypoints for each image in the format
            {'image': keypoints with shape (num_individuals, num_keypoints, 3)}
        pcutoff: the pcutoff used to use
        unique_bodypart_poses: the predicted poses for unique bodyparts
        unique_bodypart_gt: the ground truth for unique bodyparts
        bbox_margin: the margin used to create bounding boxes from keypoints to compute
            keypoint mAP.

    Returns:
        a dictionary of scores containign the following keys
            ['rmse', 'rmse_pcutoff', 'mAP', 'mAR', 'mAP_pcutoff', 'mAR_pcutoff']

    Examples:
        >>> # Define the p-cutoff, prediction, and target DataFrames
        >>> pcutoff = 0.5
        >>> prediction = {"img0": [[[0.1, 0.5, 0.4], [5.2, 3.3, 0.9]], ...], ...}
        >>> ground_truth = {"img0": [[[0, 0], [5, 3]], ...], ...}
        >>> # Compute the scores
        >>> scores = get_scores(poses, ground_truth, pcutoff)
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
    if not len(poses) == len(ground_truth):
        raise ValueError(
            "The prediction and ground truth dicts must contain the same number of "
            f"images (poses={len(poses)}, gt={len(ground_truth)})"
        )

    ground_truth = {
        image: mask_invisible(gt_pose, mask_value=np.nan)
        for image, gt_pose in ground_truth.items()
    }

    image_paths = list(poses)
    pred_poses = build_keypoint_array(poses, image_paths)[..., :3].reshape((-1, 3))
    gt_poses = build_keypoint_array(ground_truth, image_paths).reshape((-1, 2))
    if unique_bodypart_poses is not None:
        unique_bodypart_gt = {
            image: mask_invisible(gt_pose, mask_value=np.nan)
            for image, gt_pose in unique_bodypart_gt.items()
        }
        pred_poses = np.concatenate(
            [
                pred_poses,
                build_keypoint_array(unique_bodypart_poses, image_paths)[
                    ..., :3
                ].reshape((-1, 3)),
            ]
        )
        gt_poses = np.concatenate(
            [
                gt_poses,
                build_keypoint_array(unique_bodypart_gt, image_paths).reshape((-1, 2)),
            ]
        )

    pred_poses[pred_poses == -1] = np.nan
    rmse, rmse_pcutoff = compute_rmse(pred_poses, gt_poses, pcutoff=pcutoff)

    oks = compute_oks(poses, ground_truth, margin=bbox_margin, pcutoff=None)
    oks_pcutoff = compute_oks(poses, ground_truth, margin=bbox_margin, pcutoff=pcutoff)

    return {
        "rmse": rmse,
        "rmse_pcutoff": rmse_pcutoff,
        "mAP": 100 * oks["mAP"],
        "mAR": 100 * oks["mAR"],
        "mAP_pcutoff": 100 * oks_pcutoff["mAP"],
        "mAR_pcutoff": 100 * oks_pcutoff["mAR"],
    }


def build_keypoint_array(
    keypoints: dict[str, np.ndarray], keys: list[str]
) -> np.ndarray:
    """Stacks arrays of keypoints in a given order

    Args:
        keypoints: the keypoint arrays to stack
        keys: the order of keys to use to stack the arrays

    Returns:
        the stacked arrays
    """
    image_keypoints = []
    for image_key in keys:
        image_keypoints.append(keypoints[image_key])
    return np.stack(image_keypoints)


def compute_rmse(
    pred: np.ndarray, ground_truth: np.ndarray, pcutoff: float = -1
) -> tuple[float, float]:
    """Computes the root mean square error (rmse) for predictions vs the ground truth labels

    Assumes that poses have been aligned to ground truth (keypoint i in the pred array
    corresponds to keypoint i in the ground_truth array)

    Args:
        pred: (n, 3) the predicted keypoints in format x, y, score
        ground_truth: (n, 2) the ground truth keypoints
        pcutoff: the pcutoff score

    Returns:
        the RMSE and RMSE with pcutoff values
    """
    if pred.shape[0] != ground_truth.shape[0]:
        raise ValueError(
            "Prediction and target arrays must have same number of elements!"
        )

    mask = pred[:, 2] >= pcutoff
    square_distances = (pred[:, :2] - ground_truth) ** 2
    mean_square_errors = np.sum(square_distances, axis=1)
    rmse = np.nanmean(np.sqrt(mean_square_errors)).item()
    rmse_p = np.nan
    if len(mean_square_errors[mask]) > 0:
        rmse_p = np.nanmean(np.sqrt(mean_square_errors[mask])).item()
    return rmse, rmse_p


def compute_oks(
    pred: dict[str, np.array],
    ground_truth: dict[str, np.array],
    oks_sigma=0.1,
    margin=0,
    symmetric_kpts=None,
    pcutoff: float | None = None,
) -> dict:
    """Computes the

    Assumes that poses have been aligned to ground truth (for an image, individual i in
    the pred array corresponds to individual i in the ground_truth array)

    Args:
        pred: the predicted poses for each image in the format
            {'image': keypoints with shape (num_individuals, num_keypoints, 3)}
        ground_truth: ground truth keypoints for each image in the format
            {'image': keypoints with shape (num_individuals, num_keypoints, 3)}
        oks_sigma: sigma for OKS computation.
        margin: margin used for bbox computation.
        symmetric_kpts: TODO: not supported yet
        pcutoff: the pcutoff used to use

    Returns:
        the OKS scores
    """
    masked_pred = {}
    for image_path, keypoints_with_scores in pred.items():
        keypoints = keypoints_with_scores[:, :, :2].copy()
        if pcutoff is not None:
            keypoints[keypoints_with_scores[:, :, 2] < pcutoff] = np.nan
        masked_pred[image_path] = keypoints

    assemblies_pred = build_assemblies(masked_pred)
    assemblies_gt = build_assemblies(ground_truth)
    return evaluate_assembly(
        assemblies_pred,
        assemblies_gt,
        oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
        greedy_matching=True,
        with_tqdm=False,
    )


def compute_identity_scores(
    individuals: list[str],
    bodyparts: list[str],
    predictions: dict[str, np.ndarray],
    identity_scores: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    FIXME: With DLCRNet all heatmap "peaks" above 0.01 were kept, with 1 keypoint and
     1 identity score map per peak. Then, for each ground truth keypoint, we selected
     the prediction closest to it, and evaluated the identity score in that position.
     This is no longer the case, as we're now evaluating after assembly. So we only
     have num_individuals assemblies.

    Args:
        individuals:
        bodyparts:
        predictions: (num_assemblies, num_bodyparts, 3)
        identity_scores: (num_assemblies, num_bodyparts, num_individuals)
        ground_truth: (num_individuals, num_bodyparts, 3)

    Returns:

    """
    if not len(predictions) == len(ground_truth):
        raise ValueError("Mismatch between number of predictions and ground truth")

    all_bpts = np.asarray(len(individuals) * bodyparts)
    ids = np.full((len(predictions), len(all_bpts), 2), np.nan)
    for i, (image, pred) in enumerate(predictions.items()):
        for j in range(len(individuals)):
            for k in range(len(bodyparts)):
                bpt_idx = len(bodyparts) * j + k
                ids[i, bpt_idx, 0] = j

        gt = mask_invisible(ground_truth[image], mask_value=np.nan)
        id_scores = identity_scores[image]

        # reorder to (bodypart, individual, ...)
        gt = gt.transpose((1, 0, 2))
        pred = pred.transpose((1, 0, 2))[..., :2]
        id_scores = id_scores.transpose((1, 0, 2))
        for bpt, bpt_gt, bpt_pred, bpt_id_scores in zip(bodyparts, gt, pred, id_scores):
            # assign ground truth keypoints to the closest prediction, so the ID score
            # is the closest possible to the ID score computed with "ground truth"
            indices_gt = np.flatnonzero(np.all(~np.isnan(bpt_gt), axis=1))
            neighbors = find_closest_neighbors(bpt_gt[indices_gt], bpt_pred, k=3)
            found = neighbors != -1
            indices = np.flatnonzero(all_bpts == bpt)
            # Get the predicted identity of each bodypart by taking the argmax
            ids[i, indices[indices_gt[found]], 1] = np.argmax(
                bpt_id_scores[neighbors[found]], axis=1
            )

    ids = ids.reshape((len(predictions), len(individuals), len(bodyparts), 2))
    results = {}
    for i, bpt in enumerate(bodyparts):
        temp = ids[:, :, i].reshape((-1, 2))
        valid = np.isfinite(temp).all(axis=1)
        y_true, y_pred = temp[valid].T
        results[f"{bpt}_accuracy"] = accuracy_score(y_true, y_pred)

    return results


def _match_identity_preds_to_gt(
    config_path: str, full_pickle_path: str
) -> tuple[np.ndarray, list]:
    with open(full_pickle_path, "rb") as f:
        data = pickle.load(f)
    metadata = data.pop("metadata")
    cfg = read_config(config_path)
    all_ids = cfg["individuals"].copy()
    all_bpts = cfg["multianimalbodyparts"] * len(all_ids)
    n_multibodyparts = len(all_bpts)
    if cfg["uniquebodyparts"]:
        all_ids += ["single"]
        all_bpts += cfg["uniquebodyparts"]
    all_bpts = np.asarray(all_bpts)
    joints = metadata["all_joints_names"]
    ids = np.full((len(data), len(all_bpts), 2), np.nan)
    for i, dict_ in enumerate(data.values()):
        id_gt, _, df_gt = dict_["groundtruth"]
        for j, id_ in enumerate(id_gt):
            if id_.size:
                ids[i, j, 0] = all_ids.index(id_)

        df = df_gt.unstack("coords").reindex(joints, level="bodyparts")
        xy_pred = dict_["prediction"]["coordinates"][0]
        for bpt, xy_gt in df.groupby(level="bodyparts"):
            inds_gt = np.flatnonzero(np.all(~np.isnan(xy_gt), axis=1))
            n_joint = joints.index(bpt)
            xy = xy_pred[n_joint]
            if inds_gt.size and xy.size:
                # Pick the predictions closest to ground truth,
                # rather than the ones the model has most confident in
                xy_gt_values = xy_gt.iloc[inds_gt].values
                neighbors = find_closest_neighbors(xy_gt_values, xy, k=3)
                found = neighbors != -1
                inds = np.flatnonzero(all_bpts == bpt)
                id_ = dict_["prediction"]["identity"][n_joint]
                ids[i, inds[inds_gt[found]], 1] = np.argmax(
                    id_[neighbors[found]], axis=1
                )
    ids = ids[:, :n_multibodyparts].reshape((len(data), len(cfg["individuals"]), -1, 2))
    return ids, list(data)


def compute_id_accuracy(ids: np.ndarray, mask_test: np.ndarray) -> np.ndarray:
    nbpts = ids.shape[2]  # ids shape is (n_images, n_individuals, n_bodyparts, 2)
    accu = np.empty((nbpts, 2))
    for i in range(nbpts):
        temp = ids[:, :, i].reshape((-1, 2))
        valid = np.isfinite(temp).all(axis=1)
        y_true, y_pred = temp[valid].T
        mask = np.repeat(mask_test, ids.shape[1])[valid]
        ac_train = accuracy_score(y_true[~mask], y_pred[~mask])
        ac_test = accuracy_score(y_true[mask], y_pred[mask])
        accu[i] = ac_train, ac_test
    return accu


def build_assemblies(poses: dict[str, np.ndarray]) -> dict[str, list[Assembly]]:
    """
    Builds assemblies from a pose array

    Args:
        poses: {image: keypoints with shape (num_individuals, num_keypoints, 2)}

    Returns:
        the assemblies for each image
    """
    assemblies = {}
    for image_path, keypoints in poses.items():
        image_assemblies = []
        for idv_bodyparts in keypoints:
            assembly = Assembly.from_array(idv_bodyparts)
            if len(assembly):
                image_assemblies.append(assembly)

        assemblies[image_path] = image_assemblies

    return assemblies


def pair_predicted_individuals_with_gt(
    predictions: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """TODO: implement with OKS as well
    Uses RMSE to match predicted individuals to frame annotations for a batch of
    frames. This method is preferred to OKS, as OKS needs at least 2 annotated
    keypoints per animal (to compute area)

    The poses array is modified in-place, where the order of elements are
    swapped in 2nd dimension (individuals) such that the keypoints in predictions[img][i]
    is matched to the ground truth annotations of df_target[img][i]

    Args:
        predictions: {image_path: predicted pose of shape (individual, keypoints, 3)}
        ground_truth: the ground truth annotations to align

    Returns:
        the same dictionary as the input predictions, but where the "individual" axis
        for each prediction is aligned with the ground truth data
    """
    matched_poses = {}
    for image, pose in predictions.items():
        match_individuals = rmse_match_prediction_to_gt(pose, ground_truth[image])
        matched_poses[image] = pose[match_individuals]

    return matched_poses


def mask_invisible(
    keypoints: np.ndarray, mask_value: int | float | np.nan = -1.0
) -> np.ndarray:
    """
    Masks keypoints that are not visible in an array.

    Args:
        keypoints: a keypoint array of shape (..., 3), where the last axis contains
            the x, y and visibility values (0 == invisible)
        mask_value: the value to give to the keypoints that are masked

    Returns:
        a keypoint array of shape (..., 2) with the coordinates of the keypoints marked
        as invisible replaced with the mask value
    """
    keypoints = keypoints.copy()
    not_visible = keypoints[..., 2] <= 0
    keypoints[not_visible, :2] = mask_value
    return keypoints[..., :2]
