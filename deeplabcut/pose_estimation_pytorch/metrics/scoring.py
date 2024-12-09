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
from deeplabcut.utils.auxiliaryfunctions import read_config


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
