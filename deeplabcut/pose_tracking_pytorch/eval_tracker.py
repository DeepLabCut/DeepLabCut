import motmetrics as mm
import numpy as np
import pandas as pd
import pickle
import re
import warnings
from deeplabcut.pose_estimation_tensorflow.lib import (
    trackingutils,
    inferenceutils,
)
import glob
from mmappickle import mmapdict
from functools import partial
from deeplabcut.utils.auxfun_videos import VideoWriter
from pathlib import Path
from deeplabcut import utils
import os
from deeplabcut.refine_training_dataset.stitch import TrackletStitcher
from scipy.optimize import linear_sum_assignment
from deeplabcut.utils import auxiliaryfunctions
from tqdm import trange
import torch
import argparse


def reconstruct_bbox_from_bodyparts(data, margin, to_xywh=False):
    bbox = np.full((data.shape[0], 5), np.nan)
    x = data.xs("x", axis=1, level=-1)
    y = data.xs("y", axis=1, level=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bbox[:, 0] = np.nanmin(x, axis=1) - margin
        bbox[:, 1] = np.nanmin(y, axis=1) - margin
        bbox[:, 2] = np.nanmax(x, axis=1) + margin
        bbox[:, 3] = np.nanmax(y, axis=1) + margin
        bbox[:, -1] = np.nanmean(data.xs("likelihood", axis=1, level=-1), axis=1)
    if to_xywh:
        convert_bbox_to_xywh(bbox, inplace=True)
    return bbox


def reconstruct_all_bboxes(data, margin, to_xywh):
    animals = data.columns.get_level_values("individuals").unique().tolist()
    try:
        animals.remove("single")
    except ValueError:
        pass
    bboxes = np.full((len(animals), data.shape[0], 5), np.nan)
    for n, animal in enumerate(animals):
        bboxes[n] = reconstruct_bbox_from_bodyparts(
            data.xs(animal, axis=1, level=1), margin, to_xywh
        )
    return bboxes


def convert_bbox_to_xywh(bbox, inplace=False):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    if not inplace:
        new_bbox = bbox.copy()
        new_bbox[:, 2] = w
        new_bbox[:, 3] = h
        return new_bbox
    bbox[:, 2] = w
    bbox[:, 3] = h


def print_all_metrics(accumulators, all_params=None):
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


def compute_mot_metrics_bboxes(bboxes, bboxes_ground_truth):
    if bboxes.shape != bboxes_ground_truth.shape:
        raise ValueError("Dimension mismatch. Check the inputs.")

    ids = np.array(list(range(bboxes_ground_truth.shape[0])))
    acc = mm.MOTAccumulator(auto_id=True)
    for i in range(bboxes_ground_truth.shape[1]):
        bboxes_hyp = bboxes[:, i, :4]
        bboxes_gt = bboxes_ground_truth[:, i, :4]
        empty_hyp = np.isnan(bboxes_hyp).any(axis=1)
        empty_gt = np.isnan(bboxes_gt).any(axis=1)
        bboxes_hyp = bboxes_hyp[~empty_hyp]
        bboxes_gt = bboxes_gt[~empty_gt]
        dist = mm.distances.iou_matrix(bboxes_gt, bboxes_hyp)
        acc.update(ids[~empty_gt], ids[~empty_hyp], dist)
    return acc


def evaluate_tracker(
    config_path,
    inference_config_path,
    ground_truth_h5_file,
    full_pickle_file,
    params,
    ignore_bodyparts=None,
    track_method="ellipse",
    identity_only=False,
    n_tracks=0,
    funcs=None,
):
    cfg = auxiliaryfunctions.read_config(config_path)
    inference_cfg = auxiliaryfunctions.read_plainconfig(inference_config_path)
    testing_cfg = inference_cfg.copy()
    ground_truth = pd.read_hdf(ground_truth_h5_file)
    with open(full_pickle_file, "rb") as file:
        data = pickle.load(file)

    if track_method == "box":
        ground_truth_data = reconstruct_all_bboxes(
            ground_truth, inference_cfg.get("boundingboxslack", 0), to_xywh=True
        )
    elif track_method == "ellipse":
        ground_truth_data = trackingutils.reconstruct_all_ellipses(
            ground_truth, inference_cfg.get("sd", 2)
        )
    else:
        raise ValueError(f"Unknown track method {track_method}")

    ids = np.arange(ground_truth_data.shape[0])
    multi_bpts = cfg["multianimalbodyparts"]
    ass = inferenceutils.Assembler(
        data,
        max_n_individuals=testing_cfg["topktoretain"],
        n_multibodyparts=len(multi_bpts),
        pcutoff=testing_cfg.get("pcutoff", 0.1),
        min_affinity=testing_cfg.get("pafthreshold", 0.05),
        identity_only=identity_only,
    )
    ass.assemble(chunk_size=0)

    keep = set(multi_bpts).difference(ignore_bodyparts or [])
    keep_inds = sorted(multi_bpts.index(bpt) for bpt in keep)

    ntot = len(params)
    accumulators = []
    accumulators_with_stitcher = []
    for n, vals in enumerate(params, start=1):
        print(f"Combination {n}/{ntot}")
        testing_cfg.update(vals)

        if track_method == "box":
            mot_tracker = trackingutils.Sort(testing_cfg)
        else:
            mot_tracker = trackingutils.SORTEllipse(
                testing_cfg.get("max_age", 1),
                testing_cfg.get("min_hits", 1),
                testing_cfg.get("iou_threshold", 0.6),
                inference_cfg.get("sd", 2),
            )
        acc = mm.MOTAccumulator(auto_id=True)
        tracklets = dict()
        tracklets["header"] = pd.MultiIndex.from_product(
            [[""], ass.metadata["joint_names"], ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        if cfg[
            "uniquebodyparts"
        ]:  # Initialize storage of the 'single' individual track
            tracklets["single"] = {}
            _single = {}
            for index, imname in enumerate(ass.metadata["imnames"]):
                single_detection = ass.unique.get(index)
                if single_detection is None:
                    continue
                imindex = int(re.findall(r"\d+", imname)[0])
                _single[imindex] = single_detection
            tracklets["single"].update(_single)

        for i in trange(len(ass.metadata["imnames"])):
            imname = ass.metadata["imnames"][i]
            assemblies = ass.assemblies.get(i)
            if assemblies is None:
                continue
            animals = np.stack([ass.data for ass in assemblies])
            if not identity_only:
                if track_method == "box":
                    bboxes = trackingutils.calc_bboxes_from_keypoints(
                        animals[:, keep_inds],
                        testing_cfg.get("boundingboxslack", 0),
                        offset=0,
                    )  # TODO: get cropping parameters and utilize!
                    trackers = mot_tracker.update(bboxes)

                    bboxes_hyp = convert_bbox_to_xywh(trackers[:, :4])
                    bboxes_gt = ground_truth_data[:, i, :4]
                    ids_gt = ids.copy()
                    empty_gt = np.isnan(bboxes_gt).any(axis=1)
                    if empty_gt.any():
                        bboxes_gt = bboxes_gt[~empty_gt]
                        ids_gt = ids_gt[~empty_gt]

                    empty_hyp = np.isnan(bboxes_hyp).any(axis=1)
                    if empty_gt.any():
                        bboxes_hyp = bboxes_hyp[~empty_hyp]

                    dist = mm.distances.iou_matrix(bboxes_gt, bboxes_hyp)
                    acc.update(ids_gt, trackers[~empty_hyp, -1], dist)
                else:
                    xy = animals[:, keep_inds, :2]
                    trackers = mot_tracker.track(xy)
                    ellipses_hyp = trackers[:, :5]
                    ellipses_gt = ground_truth_data[:, i]
                    ids_gt = ids.copy()
                    empty = np.isnan(ellipses_gt).any(axis=1)
                    if empty.any():
                        ellipses_gt = ellipses_gt[~empty]
                        ids_gt = ids_gt[~empty]
                    cost_matrix = np.zeros((len(ellipses_gt), len(ellipses_hyp)))
                    gt_el = [trackingutils.Ellipse(*e) for e in ellipses_gt]
                    hyp_el = [trackingutils.Ellipse(*e[:5]) for e in ellipses_hyp]
                    for i, el in enumerate(gt_el):
                        for j, tracker in enumerate(hyp_el):
                            cost_matrix[i, j] = 1 - el.calc_similarity_with(tracker)
                    acc.update(ids_gt, trackers[:, -1], cost_matrix)

                trackingutils.fill_tracklets(tracklets, trackers, animals, imname)
            # else:
            #     mat = np.zeros((len(assemblies), testing_cfg["topktoretain"]))
            #     for nrow, assembly in enumerate(assemblies):
            #         for k, v in assembly.soft_identity.items():
            #             mat[nrow, k] = v
            #     inds = linear_sum_assignment(mat, maximize=True)
            #     trackers = np.c_[inds][:, ::-1]
            #     # acc.update()

        accumulators.append(acc)
        if n_tracks:
            if track_method == "ellipse":
                ground_truth_data = reconstruct_all_bboxes(
                    ground_truth, inference_cfg.get("boundingboxslack", 0), to_xywh=True
                )
            for func in funcs:
                t = tracklets.copy()
                stitcher = TrackletStitcher.from_dict_of_dict(t, n_tracks)
                stitcher.build_graph(weight_func=func)
                stitcher.stitch()
                df = stitcher.format_df().reindex(range(ground_truth_data.shape[1]))
                bboxes_with_stitcher = reconstruct_all_bboxes(
                    df,
                    testing_cfg["boundingboxslack"],
                    to_xywh=True,
                )
                temp = compute_mot_metrics_bboxes(
                    bboxes_with_stitcher, ground_truth_data
                )
                accumulators_with_stitcher.append(temp)

    summary = print_all_metrics(accumulators, params)
    if len(accumulators_with_stitcher):
        summary2 = print_all_metrics(accumulators_with_stitcher)
    else:
        summary2 = None
    return summary, summary2


from deeplabcut.pose_tracking_pytorch import inference


def calc_proximity_and_visibility_indices(hdf_file):

    import pandas as pd
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist

    df = pd.read_hdf(hdf_file)

    df = df.droplevel("scorer", axis=1).dropna(axis=1, how="all")
    df.drop("likelihood", axis=1, level=-1, inplace=True)
    if "single" in df:
        df.drop("single", axis=1, level=0, inplace=True)
    n_animals = len(df.columns.get_level_values("individuals").unique())

    temp = df.groupby("individuals", axis=1).count()
    mask = temp >= 2 * 2
    counts = mask.sum(axis=1)
    viz = counts / n_animals
    df.dropna(how="all", inplace=True)
    coords = df.to_numpy().reshape((df.shape[0], n_animals, -1, 2))

    centroids = np.expand_dims(np.nanmean(coords, axis=2), 2)

    index = np.zeros(coords.shape[:2])
    for i in range(coords.shape[0]):
        c = centroids[i]
        xy = coords[i]
        radii = np.zeros(coords.shape[1])
        for j in range(coords.shape[1]):
            radii[j] = np.nanmax(cdist(c[j], xy[j]))
        tree = cKDTree(xy.reshape((-1, 2)))
        n_all = tree.query_ball_point(c.squeeze(), np.ceil(radii), return_length=True)

        n = np.isfinite(xy).any(axis=2).sum(axis=1)

        index[i] = (n_all - n) / n

    prox = np.nanmean(index, axis=1)
    prox = prox[~np.isnan(prox)]
    return prox, viz
