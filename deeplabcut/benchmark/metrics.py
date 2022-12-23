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

"""Evaluation metrics for the DeepLabCut benchmark."""

import sys
import unittest.mock

# TODO(stes) mocking a few modules to rely in fewer dependencies, without
# causing import errors when using deeplabcut.
MOCK_MODULES = ["statsmodels", "statsmodels.api", "pytables"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = unittest.mock.MagicMock()

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

import deeplabcut.benchmark.utils
from deeplabcut.pose_estimation_tensorflow.core import evaluate_multianimal
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils
from deeplabcut.utils.conversioncode import guarantee_multiindex_rows


def _format_gt_data(h5file):
    df = pd.read_hdf(h5file)

    def _get_unique_level_values(header, level):
        return header.get_level_values(level).unique().to_list()

    animals = _get_unique_level_values(df.columns, "individuals")
    kpts = _get_unique_level_values(df.columns, "bodyparts")
    try:
        n_unique = len(
            _get_unique_level_values(
                df.xs("single", level="individuals", axis=1).columns, "bodyparts"
            )
        )
    except KeyError:
        n_unique = 0
    guarantee_multiindex_rows(df)
    file_paths = [os.path.join(*row) for row in df.index.to_list()]
    temp = (
        df.stack("individuals", dropna=False)
        .reindex(animals, level="individuals")
        .reindex(kpts, level="bodyparts", axis=1)
    )
    data = temp.to_numpy().reshape((len(file_paths), len(animals), -1, 2))
    meta = {"animals": animals, "keypoints": kpts, "n_unique": n_unique}
    return {
        "annotations": dict(zip(file_paths, data)),
        "metadata": meta,
    }


def calc_prediction_errors(preds, gt):
    kpts_gt = gt["metadata"]["keypoints"]
    kpts_pred = preds["metadata"]["keypoints"]
    map_ = {kpts_gt.index(kpt): i for i, kpt in enumerate(kpts_pred)}
    annot = gt["annotations"]

    # Map image paths from predicted data to GT as the first are typically
    # absolute whereas the latter are relative to the project path.
    def _map(strings, substrings):
        lookup = dict()
        strings_ = strings.copy()
        substrings_ = substrings.copy()
        while strings_:
            string = strings_.pop()
            for s in substrings_:
                if string.endswith(s):
                    lookup[string] = s
                    substrings_.remove(s)
                    break
        return lookup

    map_images = _map(list(preds["predictions"]), list(annot))

    errors = np.full(
        (
            len(preds["predictions"]),
            len(gt["metadata"]["animals"]),
            len(kpts_gt),
            2,  # Hold distance to GT and confidence
        ),
        np.nan,
    )
    for n, (path, preds_) in enumerate(preds["predictions"].items()):
        if not preds_:
            continue
        xy_gt = annot[map_images[path]].swapaxes(0, 1)
        xy_pred = preds_["coordinates"][0]
        conf_pred = preds_["confidence"]
        for i, xy_gt_ in enumerate(xy_gt):
            visible = np.flatnonzero(np.all(~np.isnan(xy_gt_), axis=1))
            xy_pred_ = xy_pred[map_[i]]
            if visible.size and xy_pred_.size:
                # Pick the predictions closest to ground truth,
                # rather than the ones the model has most confident in.
                neighbors = evaluate_multianimal._find_closest_neighbors(
                    xy_gt_[visible], xy_pred_, k=3
                )
                found = neighbors != -1
                if ~np.any(found):
                    continue
                min_dists = np.linalg.norm(
                    xy_gt_[visible][found] - xy_pred_[neighbors[found]],
                    axis=1,
                )
                conf_pred_ = conf_pred[map_[i]]
                errors[n, visible[found], i, 0] = min_dists
                errors[n, visible[found], i, 1] = conf_pred_[neighbors[found], 0]
    return errors


def conv_obj_to_assemblies(eval_results_obj, keypoint_names):
    """Convert predictions to deeplabcut assemblies."""
    assemblies = {}
    for image_path, results in eval_results_obj.items():
        lst = []
        for dict_ in results:
            ass = inferenceutils.Assembly(len(keypoint_names))
            for i, kpt in enumerate(keypoint_names):
                xy = dict_["pose"][kpt]
                if ~np.isnan(xy).all():
                    joint = inferenceutils.Joint(pos=(xy), label=i)
                    ass.add_joint(joint)
            # TODO(jeylau) add affinity.setter to Assembly
            ass._affinity = dict_["score"]
            ass._links = [None]
            if len(ass):
                lst.append(ass)
        assemblies[image_path] = lst
    return assemblies


def calc_map_from_obj(
    eval_results_obj,
    h5_file,
    metadata_file,
    oks_sigma=0.1,
    margin=0,
    symmetric_kpts=None,
    drop_kpts=None,
):
    """Calculate mean average precision (mAP) based on predictions."""
    df = pd.read_hdf(h5_file)
    try:
        df.drop("single", level="individuals", axis=1, inplace=True)
    except KeyError:
        pass
    n_animals = len(df.columns.get_level_values("individuals").unique())
    kpts = list(df.columns.get_level_values("bodyparts").unique())
    image_paths = list(eval_results_obj)
    ground_truth = (
        df.loc[image_paths].to_numpy().reshape((len(image_paths), n_animals, -1, 2))
    )
    temp = np.ones((*ground_truth.shape[:3], 3))
    temp[..., :2] = ground_truth
    assemblies_gt = inferenceutils._parse_ground_truth_data(temp)
    with open(metadata_file, "rb") as f:
        inds_test = set(pickle.load(f)[2])
    assemblies_gt_test = {k: v for k, v in assemblies_gt.items() if k in inds_test}

    # TODO(stes): remove/rewrite
    if drop_kpts is not None:
        temp = {}
        for k, v in assemblies_gt_test.items():
            lst = []
            for a in v:
                arr = np.delete(a.data[:, :3], drop_kpts, axis=0)
                a = inferenceutils.Assembly.from_array(arr)
                lst.append(a)
            temp[k] = lst
        assemblies_gt_test = temp
        for ind in sorted(drop_kpts, reverse=True):
            kpts.pop(ind)

    assemblies_pred_ = conv_obj_to_assemblies(eval_results_obj, kpts)
    assemblies_pred = dict(enumerate(assemblies_pred_.values()))

    with deeplabcut.benchmark.utils.DisableOutput():
        oks = inferenceutils.evaluate_assembly(
            assemblies_pred,
            assemblies_gt_test,
            oks_sigma,
            margin=margin,
            symmetric_kpts=symmetric_kpts,
        )
    return oks["mAP"]


def calc_rmse_from_obj(
    eval_results_obj,
    h5_file,
    metadata_file,
    drop_kpts=None,
):
    """Calc prediction errors for submissions."""
    gt = _format_gt_data(h5_file)
    kpts = gt["metadata"]["keypoints"]
    if drop_kpts:
        for k, v in gt["annotations"].items():
            gt["annotations"][k] = np.delete(v, drop_kpts, axis=1)
        for ind in sorted(drop_kpts, reverse=True):
            kpts.pop(ind)
    with open(metadata_file, "rb") as f:
        inds_test = set(pickle.load(f)[2])
    test_objects = {
        k: v for i, (k, v) in enumerate(eval_results_obj.items()) if i in inds_test
    }
    assemblies_pred = conv_obj_to_assemblies(test_objects, kpts)
    preds = defaultdict(dict)
    preds["metadata"]["keypoints"] = kpts
    for image, assemblies in assemblies_pred.items():
        if assemblies:
            arr = np.stack([a.data for a in assemblies]).swapaxes(0, 1)
            data = [xy[~np.isnan(xy).any(axis=1)] for xy in arr[..., :2]]
            temp = {
                "coordinates": tuple([data]),
                "confidence": list(np.expand_dims(arr[..., 2], axis=2)),
            }
            preds["predictions"][image] = temp
    with deeplabcut.benchmark.utils.DisableOutput():
        errors = calc_prediction_errors(preds, gt)
    return np.nanmean(errors[..., 0])
