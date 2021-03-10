"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import pickle
import shutil
import warnings
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

from deeplabcut.pose_estimation_tensorflow import return_evaluate_network_data
from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import Assembler
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions


def _set_up_evaluation(data):
    params = dict()
    params["joint_names"] = data["metadata"]["all_joints_names"]
    params["num_joints"] = len(params["joint_names"])
    partaffinityfield_graph = data["metadata"]["PAFgraph"]
    params["paf"] = np.arange(len(partaffinityfield_graph))
    params["paf_graph"] = params["paf_links"] = [
        partaffinityfield_graph[l] for l in params["paf"]
    ]
    params["bpts"] = params["ibpts"] = range(params["num_joints"])
    params["imnames"] = [fn for fn in list(data) if fn != "metadata"]
    return params


def compute_crossval_metrics(
    config_path,
    inference_cfg,
    shuffle=1,
    trainingsetindex=0,
    modelprefix="",
    snapshotindex=-1,
    dcorr=5,
    leastbpts=3,
):

    fns = return_evaluate_network_data(
        config_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )

    predictionsfn = fns[snapshotindex]
    data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(predictionsfn)
    params = _set_up_evaluation(data)

    n_images = len(params["imnames"])
    poses = []
    poses_gt = []
    stats = np.full(
        (n_images, 7), np.nan
    )  # RMSE, hits, misses, false_pos, num_detections, pck
    columns = ["train_iter", "train_frac", "shuffle"]
    columns += [
        "_".join((b, a))
        for a in ("train", "test")
        for b in ("rmse", "hits", "misses", "falsepos", "ndetects", "pck", "rpck")
    ]
    for n, imname in enumerate(params["imnames"]):
        animals = assemble_individuals(
            inference_cfg,
            data[imname],
            params["num_joints"],
            params["bpts"],
            params["ibpts"],
            params["paf"],
            params["paf_graph"],
            evaluation=True,
        )
        if animals is None:
            continue

        n_animals = len(animals)
        if n_animals:
            _, _, GT = data[imname]["groundtruth"]
            GT = GT.droplevel("scorer").unstack(level=["bodyparts", "coords"])
            gt = GT.values.reshape((GT.shape[0], -1, 2))
            poses_gt.append(gt)

            if (
                leastbpts > 0
            ):  # ONLY KEEP animals with at least as many bpts (to get rid of crops that cannot be assembled)
                gt = gt[np.nansum(gt, axis=(1, 2)) > leastbpts]

            poses.append(animals)
            ani = animals[:, : gt.shape[1], :2]
            mat = np.full((gt.shape[0], n_animals), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(len(gt)):
                    for j in range(len(animals)):
                        mat[i, j] = np.sqrt(
                            np.nanmean(np.sum((gt[i] - ani[j, :, :2]) ** 2, axis=1))
                        )

            if np.nansum(mat) > 0:  # also assures at least one not nan np.size(mat)>0:
                mat[np.isnan(mat)] = np.nanmax(mat) + 1
                row_indices, col_indices = linear_sum_assignment(mat)
                stats[n, 0] = mat[row_indices, col_indices].mean()  # rmse

                gt_annot = np.any(~np.isnan(gt), axis=2)
                gt_matched = gt_annot[row_indices].flatten()

                dlc_annot = np.any(~np.isnan(ani), axis=2)  # DLC assemblies
                dlc_matched = dlc_annot[col_indices].flatten()

                stats[n, 1] = np.logical_and(gt_matched, dlc_matched).sum()  # hits
                stats[n, 2] = gt_annot.sum() - stats[n, 1]  # misses
                stats[n, 3] = np.logical_and(
                    ~gt_matched, dlc_matched
                ).sum()  # additional detections
                stats[n, 4] = n_animals

                numgtpts = gt_annot.sum()
                # animal & bpt-wise distance!
                if numgtpts > 0:
                    # corrkps=np.sum((gt[row_indices]-ani[col_indices])**2,axis=2)<dcorr**2
                    dists = np.sum((gt[row_indices] - ani[col_indices]) ** 2, axis=2)
                    corrkps = dists[np.isfinite(dists)] < dcorr ** 2
                    pck = (
                        corrkps.sum() * 1.0 / numgtpts
                    )  # weigh by actually annotated ones!
                    rpck = (
                        np.sum(
                            np.exp(-dists[np.isfinite(dists)] * 1.0 / (2 * dcorr ** 2))
                        )
                        * 1.0
                        / numgtpts
                    )

                else:
                    pck = 1.0  # does that make sense? As a convention fully correct...
                    rpck = 1.0

                stats[n, 5] = pck
                stats[n, 6] = rpck

    train_iter = int(predictionsfn.split("-")[-1].split(".")[0])
    train_frac = int(predictionsfn.split("trainset")[1].split("shuffle")[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = np.r_[
            train_iter,
            train_frac,
            shuffle,
            np.nanmean(stats[metadata["data"]["trainIndices"]], axis=0),
            np.nanmean(stats[metadata["data"]["testIndices"]], axis=0),
        ]

    return pd.DataFrame(res.reshape((1, -1)), columns=columns), poses_gt, poses


def compute_crossval_metrics_preloadeddata(
    params,
    columns,
    inference_cfg,
    data,
    trainIndices,
    testIndices,
    train_iter,
    train_frac,
    shuffle,
    lowerbound,
    upperbound,
    dcorr,
    leastbpts,
):
    n_images = len(params["imnames"])
    stats = np.full(
        (n_images, 7), np.nan
    )  # RMSE, hits, misses, false_pos, num_detections, pck, rpck
    for n, imname in enumerate(params["imnames"]):
        animals = assemble_individuals(
            inference_cfg,
            data[imname],
            params["num_joints"],
            params["bpts"],
            params["ibpts"],
            params["paf"],
            params["paf_graph"],
            lowerbound,
            upperbound,
            evaluation=True,
        )
        if animals is None:
            continue

        n_animals = len(animals)
        if n_animals:
            _, _, GT = data[imname]["groundtruth"]
            GT = GT.droplevel("scorer").unstack(level=["bodyparts", "coords"])
            gt = GT.values.reshape((GT.shape[0], -1, 2))

            if (
                leastbpts > 0
            ):  # ONLY KEEP animals with at least as many bpts (to get rid of crops that cannot be assembled)
                gt = gt[np.nansum(gt, axis=(1, 2)) > leastbpts]

                ani = animals[:, : gt.shape[1], :2]
            mat = np.full((gt.shape[0], n_animals), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(len(gt)):
                    for j in range(len(animals)):
                        mat[i, j] = np.sqrt(
                            np.nanmean(np.sum((gt[i] - ani[j, :, :2]) ** 2, axis=1))
                        )

            if np.nansum(mat) > 0:  # np.size(mat)>0:
                mat[np.isnan(mat)] = np.nanmax(mat) + 1
                row_indices, col_indices = linear_sum_assignment(mat)
                stats[n, 0] = mat[row_indices, col_indices].mean()  # rmse

                gt_annot = np.any(~np.isnan(gt), axis=2)
                gt_matched = gt_annot[row_indices].flatten()

                dlc_annot = np.any(~np.isnan(ani), axis=2)  # DLC assemblies
                dlc_matched = dlc_annot[col_indices].flatten()

                stats[n, 1] = np.logical_and(gt_matched, dlc_matched).sum()  # hits
                stats[n, 2] = gt_annot.sum() - stats[n, 1]  # misses
                stats[n, 3] = np.logical_and(
                    ~gt_matched, dlc_matched
                ).sum()  # additional detections
                stats[n, 4] = n_animals

                numgtpts = gt_annot.sum()
                # animal & bpt-wise distance!
                if numgtpts > 0:
                    # corrkps=np.sum((gt[row_indices]-ani[col_indices])**2,axis=2)<dcorr**2
                    dists = np.sum((gt[row_indices] - ani[col_indices]) ** 2, axis=2)
                    corrkps = dists[np.isfinite(dists)] < dcorr ** 2
                    pck = (
                        corrkps.sum() * 1.0 / numgtpts
                    )  # weigh by actually annotated ones!
                    rpck = (
                        np.sum(
                            np.exp(-dists[np.isfinite(dists)] * 1.0 / (2 * dcorr ** 2))
                        )
                        * 1.0
                        / numgtpts
                    )

                else:
                    pck = 1.0  # does that make sense? As a convention fully correct...
                    rpck = 1.0

                stats[n, 5] = pck
                stats[n, 6] = rpck

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = np.r_[
            train_iter,
            train_frac,
            shuffle,
            np.nanmean(stats[trainIndices], axis=0),
            np.nanmean(stats[testIndices], axis=0),
        ]

    return pd.DataFrame(res.reshape((1, -1)), columns=columns)


def bayesian_search(
    config_path,
    inferencecfg,
    pbounds,
    edgewisecondition=True,
    shuffle=1,
    trainingsetindex=0,
    modelprefix="",
    snapshotindex=-1,
    target="rpck_test",
    maximize=True,
    init_points=20,
    n_iter=50,
    acq="ei",
    log_file=None,
    dcorr=5,
    leastbpts=3,
    printingintermediatevalues=True,
):  #

    if "rpck" in target:
        assert maximize == True

    if "rmse" in target:
        assert maximize == False

    cfg = auxiliaryfunctions.read_config(config_path)
    evaluationfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.GetEvaluationFolder(
                cfg["TrainingFraction"][int(trainingsetindex)],
                shuffle,
                cfg,
                modelprefix=modelprefix,
            )
        ),
    )

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        cfg["TrainingFraction"][int(trainingsetindex)],
        cfg["iteration"],
        modelprefix=modelprefix,
    )

    # load params
    fns = return_evaluate_network_data(
        config_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )
    predictionsfn = fns[snapshotindex]
    data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(predictionsfn)
    params = _set_up_evaluation(data)
    columns = ["train_iter", "train_frac", "shuffle"]
    columns += [
        "_".join((b, a))
        for a in ("train", "test")
        for b in ("rmse", "hits", "misses", "falsepos", "ndetects", "pck", "rpck")
    ]

    train_iter = trainingsetindex  # int(predictionsfn.split('-')[-1].split('.')[0])
    train_frac = cfg["TrainingFraction"][
        train_iter
    ]  # int(predictionsfn.split('trainset')[1].split('shuffle')[0])
    trainIndices = metadata["data"]["trainIndices"]
    testIndices = metadata["data"]["testIndices"]

    if edgewisecondition:
        mf = str(
            auxiliaryfunctions.GetModelFolder(
                cfg["TrainingFraction"][int(trainingsetindex)],
                shuffle,
                cfg,
                modelprefix=modelprefix,
            )
        )
        modelfolder = os.path.join(cfg["project_path"], mf)
        path_inferencebounds_config = (
            Path(modelfolder) / "test" / "inferencebounds.yaml"
        )
        try:
            inferenceboundscfg = auxiliaryfunctions.read_plainconfig(
                path_inferencebounds_config
            )
        except FileNotFoundError:
            print("Computing distances...")
            from deeplabcut.pose_estimation_tensorflow import calculatepafdistancebounds

            inferenceboundscfg = calculatepafdistancebounds(
                config_path, shuffle, trainingsetindex
            )
            auxiliaryfunctions.write_plainconfig(
                path_inferencebounds_config, inferenceboundscfg
            )

        partaffinityfield_graph = params["paf_graph"]
        upperbound = np.array(
            [
                float(
                    inferenceboundscfg[str(edge[0]) + "_" + str(edge[1])]["intra_max"]
                )
                for edge in partaffinityfield_graph
            ]
        )
        lowerbound = np.array(
            [
                float(
                    inferenceboundscfg[str(edge[0]) + "_" + str(edge[1])]["intra_min"]
                )
                for edge in partaffinityfield_graph
            ]
        )

        upperbound *= inferencecfg["upperbound_factor"]
        lowerbound *= inferencecfg["lowerbound_factor"]

    else:
        lowerbound = None
        upperbound = None

    def dlc_hyperparams(**kwargs):
        inferencecfg.update(kwargs)
        # Ensure type consistency
        for k, (bound, _) in pbounds.items():
            inferencecfg[k] = type(bound)(inferencecfg[k])

        stats = compute_crossval_metrics_preloadeddata(
            params,
            columns,
            inferencecfg,
            data,
            trainIndices,
            testIndices,
            train_iter,
            train_frac,
            shuffle,
            lowerbound,
            upperbound,
            dcorr=dcorr,
            leastbpts=leastbpts,
        )

        # stats = compute_crossval_metrics(config_path, inferencecfg, shuffle,trainingsetindex,
        #                                    dcorr=dcorr,leastbpts=leastbpts,modelprefix=modelprefix)

        if printingintermediatevalues:
            print(
                "rpck",
                stats["rpck_test"].values[0],
                "rpck train:",
                stats["rpck_train"].values[0],
            )
            print(
                "rmse",
                stats["rmse_test"].values[0],
                "miss",
                stats["misses_test"].values[0],
                "hit",
                stats["hits_test"].values[0],
            )

        # val = stats['rmse_test'].values[0]*(1+stats['misses_test'].values[0]*1./stats['hits_test'].values[0])
        val = stats[target].values[0]
        if np.isnan(val):
            if maximize:  # pck case
                val = -1e9  # random small number
            else:  # RMSE, return a large RMSE
                val = 1e9

        if not maximize:
            val = -val

        return val

    opt = BayesianOptimization(f=dlc_hyperparams, pbounds=pbounds, random_state=42)
    if log_file:
        load_logs(opt, log_file)
    logger = JSONLogger(
        path=os.path.join(evaluationfolder, "opti_log" + DLCscorer + ".json")
    )
    opt.subscribe(Events.OPTIMIZATION_STEP, logger)
    opt.maximize(init_points=init_points, n_iter=n_iter, acq=acq)

    inferencecfg.update(opt.max["params"])
    for k, (bound, _) in pbounds.items():
        tmp = type(bound)(inferencecfg[k])
        if isinstance(tmp, np.floating):
            tmp = np.round(tmp, 2).item()
        inferencecfg[k] = tmp

    return inferencecfg, opt


def _form_original_path(path):
    root, filename = os.path.split(path)
    return os.path.join(root, filename.split("c")[0])


def _unsorted_unique(array):
    _, inds = np.unique(array, return_index=True)
    return np.asarray(array)[np.sort(inds)]


def _rebuild_uncropped_metadata(
    metadata,
    image_paths,
    output_name="",
):
    train_inds_orig = set(metadata["data"]["trainIndices"])
    train_inds, test_inds = [], []
    for k, (basename, group) in tqdm(
        enumerate(groupby(image_paths, _form_original_path))
    ):
        imnames_ = list(group)
        if image_paths.index(imnames_[0]) in train_inds_orig:
            train_inds.append(k)
        else:
            test_inds.append(k)
    meta_new = metadata.copy()
    meta_new["data"]["trainIndices"] = train_inds
    meta_new["data"]["testIndices"] = test_inds

    if output_name:
        with open(output_name, "wb") as file:
            pickle.dump(meta_new, file)

    return meta_new


def _rebuild_uncropped_data(
    data,
    params,
    output_name="",
):
    """
    Reconstruct predicted data as if they had been obtained on full size images.
    This is required to evaluate part affinity fields and cross-validate
    and animal assembly.

    Parameters
    ----------
    data : dict
        Dictionary of predicted data as loaded from
        _full.pickle files under evaluation-results.

    params : dict
        Evaluation settings. Formed from the metadata using _set_up_evaluation().

    output_name : str
        If passed, dump the uncropped data into a pickle file of the same name.

    Returns
    -------
    (uncropped data, list of image paths)

    """
    image_paths = params["imnames"]
    bodyparts = params["joint_names"]
    idx = (
        data[image_paths[0]]["groundtruth"][2]
        .unstack("coords")
        .reindex(bodyparts, level="bodyparts")
        .index
    )
    individuals = idx.get_level_values("individuals").unique()
    has_single = "single" in individuals
    n_individuals = len(individuals) - has_single

    data_new = dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for basename, group in tqdm(groupby(image_paths, _form_original_path)):
            imnames_ = list(group)
            n_crops = len(imnames_)

            # Sort crop patches to maximize the probability they overlap with others
            all_coords_gt = [
                data[imname]["groundtruth"][2].to_numpy().reshape((-1, 2))
                for imname in imnames_
            ]
            overlap = np.zeros((n_crops, n_crops))
            inds = list(zip(*np.triu_indices(n_crops, k=1)))
            masks = dict()  # Cache boolean masks
            for i1, i2 in inds:
                if i1 not in masks:
                    masks[i1] = np.isfinite(all_coords_gt[i1]).any(axis=1)
                if i2 not in masks:
                    masks[i2] = np.isfinite(all_coords_gt[i2]).any(axis=1)
                overlap[i1, i2] = overlap[i2, i1] = np.sum(masks[i1] & masks[i2])
            count = np.count_nonzero(overlap, axis=1)
            imnames = [imnames_[i] for i in np.argsort(count)[::-1]]

            # Form the ground truth back
            ref_gt = None
            all_trans = np.zeros(
                (len(imnames), 2)
            )  # Store translations w.r.t. first ref crop
            for i, imname in enumerate(imnames):
                coords_gt = data[imname]["groundtruth"][2].to_numpy().reshape((-1, 2))
                if ref_gt is None:
                    ref_gt = coords_gt
                    continue
                trans = np.nanmean(coords_gt - ref_gt, axis=0)
                if np.all(~np.isnan(trans)):
                    all_trans[i] = trans
                empty = np.isnan(ref_gt)
                has_value = ~np.isnan(coords_gt)
                mask = np.any(empty & has_value, axis=1)
                if mask.any():
                    coords_gt_trans = coords_gt - all_trans[i]
                    ref_gt[mask] = coords_gt_trans[mask]

            # Match detections across crops
            temp = pd.DataFrame(ref_gt, index=idx, columns=["x", "y"])
            temp.columns.names = ["coords"]
            if has_single:
                temp.drop("single", level="individuals", inplace=True)
            ref_pred = np.full(
                (n_individuals, len(temp) // n_individuals, 4), np.nan
            )  # Hold x, y, prob, dist
            costs = dict()
            shape = n_individuals, n_individuals
            for ind in params["paf"]:
                costs[ind] = {
                    "m1": np.zeros(shape),
                    "distance": np.full(shape, np.inf),
                }
            if not np.isnan(temp.to_numpy()).all():
                ref_gt_ = dict()
                for bpt, df_ in temp.groupby("bodyparts"):
                    values = df_.to_numpy()
                    inds = np.flatnonzero(np.all(~np.isnan(values), axis=1))
                    ref_gt_[bpt] = values, inds
                for i, imname in enumerate(imnames):
                    coords_pred = data[imname]["prediction"]["coordinates"][0]
                    probs_pred = data[imname]["prediction"]["confidence"]
                    costs_pred = data[imname]["prediction"]["costs"]
                    map_ = dict()
                    for n, bpt in enumerate(ref_gt_):
                        xy_gt, inds_gt = ref_gt_[bpt]
                        ind = bodyparts.index(bpt)
                        xy = coords_pred[ind]
                        prob = probs_pred[ind]
                        if inds_gt.size and xy.size:
                            xy_trans = xy - all_trans[i]
                            d = cdist(xy_gt[inds_gt], xy_trans)
                            rows, cols = linear_sum_assignment(d)
                            probs_ = prob[cols]
                            dists_ = d[rows, cols]
                            inds_rows = inds_gt[rows]
                            map_[n] = inds_rows, cols
                            is_free = np.isnan(ref_pred[inds_rows, n]).all(axis=1)
                            closer = dists_ < ref_pred[inds_rows, n, -1]
                            mask = np.logical_or(is_free, closer)
                            if mask.any():
                                coords_ = xy_trans[cols]
                                sl = inds_rows[mask]
                                ref_pred[sl, n, :2] = coords_[mask]
                                ref_pred[sl, n, 2] = probs_[mask].squeeze()
                                ref_pred[sl, n, 3] = dists_[mask]
                    # Store the costs associated with the retained candidates
                    for n, (ind1, ind2) in enumerate(params["paf_graph"]):
                        if ind1 in map_ and ind2 in map_:
                            sl1 = np.ix_(map_[ind1][0], map_[ind2][0])
                            sl2 = np.ix_(map_[ind1][1], map_[ind2][1])
                            mask = costs_pred[n]["m1"][sl2] > costs[n]["m1"][sl1]
                            if mask.any():
                                inds_lin = (sl1[0] * n_individuals + sl1[1])[mask]
                                costs[n]["m1"].ravel()[inds_lin] = costs_pred[n]["m1"][
                                    sl2
                                ][mask]
                                costs[n]["distance"].ravel()[inds_lin] = costs_pred[n][
                                    "distance"
                                ][sl2][mask]

            ref_pred_ = ref_pred.swapaxes(0, 1)
            coordinates = ref_pred_[..., :2]
            confidence = ref_pred_[..., 2]
            pred_dict = {
                "prediction": {
                    "coordinates": (coordinates,),
                    "confidence": confidence,
                    "costs": costs,
                },
                "groundtruth": (None, None, temp.stack(dropna=False)),
            }
            data_new[basename + ".png"] = pred_dict

    image_paths = list(data_new)
    data_new["metadata"] = data["metadata"]

    if output_name:
        with open(output_name, "wb") as file:
            pickle.dump(data_new, file)

    return data_new, image_paths


def _rebuild_uncropped_in(
    base_folder,
):
    for dirpath, dirnames, filenames in os.walk(base_folder):
        for file in filenames:
            if file.endswith("_full.pickle"):
                full_data_file = os.path.join(dirpath, file)
                metadata_file = full_data_file.replace("_full.", "_meta.")
                with open(full_data_file, "rb") as file:
                    data = pickle.load(file)
                with open(metadata_file, "rb") as file:
                    metadata = pickle.load(file)
                params = _set_up_evaluation(data)
                _rebuild_uncropped_data(
                    data, params, full_data_file.replace(".pickle", "_uncropped.pickle")
                )
                _rebuild_uncropped_metadata(
                    metadata,
                    params["imnames"],
                    metadata_file.replace(".pickle", "_uncropped.pickle"),
                )


def _find_closest_neighbors(query, ref, k=3):
    n_preds = ref.shape[0]
    tree = cKDTree(ref)
    dist, inds = tree.query(query, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(query), -1, dtype=int)
    picked = set()
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == n_preds:
            break
    return neighbors


def _calc_separability(
    vals_left,
    vals_right,
    n_bins=101,
    metric="jeffries",
    max_sensitivity=False,
):
    if metric not in ("jeffries", "auc"):
        raise ValueError("`metric` should be either 'jeffries' or 'auc'.")

    bins = np.linspace(0, 1, n_bins)
    hist_left = np.histogram(vals_left, bins=bins)[0]
    hist_left = hist_left / hist_left.sum()
    hist_right = np.histogram(vals_right, bins=bins)[0]
    hist_right = hist_right / hist_right.sum()
    tpr = np.cumsum(hist_right)
    if metric == "jeffries":
        sep = np.sqrt(
            2 * (1 - np.sum(np.sqrt(hist_left * hist_right)))
        )  # Jeffries-Matusita distance
    else:
        sep = np.trapz(np.cumsum(hist_left), tpr)
    if max_sensitivity:
        threshold = bins[max(1, np.argmax(tpr > 0))]
    else:
        threshold = bins[np.argmin(1 - np.cumsum(hist_left) + tpr)]
    return sep, threshold


def _calc_within_between_pafs(
    data,
    metadata,
    per_bodypart=True,
    test_set_only=True,
):
    test_inds = set(metadata["data"]["testIndices"])
    within_train = defaultdict(list)
    within_test = defaultdict(list)
    between_train = defaultdict(list)
    between_test = defaultdict(list)
    mask_diag = None
    for i, dict_ in enumerate(data.values()):
        is_test = i in test_inds
        if test_set_only and not is_test:
            continue
        costs = dict_["prediction"]["costs"]
        for k, v in costs.items():
            paf = v["m1"]
            nonzero = paf != 0
            if mask_diag is None:
                mask_diag = np.eye(paf.shape[0], dtype=bool)
            within_vals = paf[np.logical_and(mask_diag, nonzero)]
            between_vals = paf[np.logical_and(~mask_diag, nonzero)]
            if is_test:
                within_test[k].extend(within_vals)
                between_test[k].extend(between_vals)
            else:
                within_train[k].extend(within_vals)
                between_train[k].extend(between_vals)
    if not per_bodypart:
        within_train = np.concatenate([*within_train.values()])
        within_test = np.concatenate([*within_test.values()])
        between_train = np.concatenate([*between_train.values()])
        between_test = np.concatenate([*between_test.values()])
    return (within_train, within_test), (between_train, between_test)


def _calibrate_distances(data, metadata):
    d = data.copy()
    d.pop('metadata', None)
    train_inds = set(metadata["data"]["trainIndices"])
    dists = defaultdict(list)
    for i, dict_ in enumerate(tqdm(d.values())):
        if i in train_inds:
            for e, v in dict_['prediction']['costs'].items():
                dists[e].extend(np.diag(v['distance']))
    arr = np.ma.masked_invalid(np.vstack(list(dists.values())))
    av = arr.mean(axis=1)
    sd = arr.std(axis=1)
    funcs = dict()
    for ind in av.nonzero()[0]:
        funcs[ind] = lambda x, ind=ind: np.exp(-(x - av[ind]) ** 2 / sd[ind] ** 2)
    return funcs


def _benchmark_paf_graphs(
    config,
    inference_cfg,
    data,
    params,
    paf_inds,
    greedy=False,
    calibration_file="",
):
    paf_graph = params["paf_graph"]
    image_paths = params["imnames"]
    bodyparts = params["joint_names"]
    idx = (
        data[image_paths[0]]["groundtruth"][2]
        .unstack("coords")
        .reindex(bodyparts, level="bodyparts")
        .index
    )
    individuals = idx.get_level_values("individuals").unique()
    n_individuals = len(individuals)
    map_ = dict(zip(individuals, range(n_individuals)))

    # Form ground truth beforehand
    ground_truth = []
    for i, imname in enumerate(image_paths):
        temp = data[imname]["groundtruth"][2]
        ground_truth.append(temp.to_numpy().reshape((-1, 2)))
    ground_truth = np.stack(ground_truth)
    ids = np.vectorize(map_.get)(idx.get_level_values("individuals").to_numpy())
    ground_truth = np.insert(ground_truth, 2, ids, axis=2)

    # Assemble animals on the full set of detections
    paf_inds = sorted(paf_inds, key=len)
    n_graphs = len(paf_inds)
    all_scores = []
    n_multi = len(auxfun_multianimal.extractindividualsandbodyparts(config)[2])
    for j, paf in enumerate(paf_inds, start=1):
        print(f"Graph {j}|{n_graphs}")
        graph = [paf_graph[i] for i in paf]
        ass = Assembler(
            data,
            max_n_individuals=inference_cfg["topktoretain"],
            n_multibodyparts=n_multi,
            graph=graph,
            paf_inds=paf,
            greedy=greedy,
            pcutoff=inference_cfg["pcutoff"],
            min_affinity=inference_cfg["paf_threshold"]
        )
        if calibration_file:
            ass.calibrate(calibration_file)

        scores = np.full((len(image_paths), 2), np.nan)
        for i, imname in enumerate(tqdm(image_paths)):
            gt = ground_truth[i]
            gt = gt[~np.isnan(gt).any(axis=1)]
            if len(np.unique(gt[:, 2])) < 2:
                continue

            animals, unique = ass._assemble(data[imname]["prediction"], i)

            # Count the number of missed bodyparts
            n_animals = len(animals)
            n_dets = gt.shape[0]
            if not n_animals:
                if n_dets:
                    scores[i, 0] = 1
            else:
                animals = [
                    np.c_[animal, np.ones(animal.shape[0]) * n]
                    for n, animal in enumerate(animals)
                ]
                hyp = np.concatenate(animals)
                hyp = hyp[~np.isnan(hyp).any(axis=1)]
                scores[i, 0] = (n_dets - hyp.shape[0]) / n_dets
                neighbors = _find_closest_neighbors(gt[:, :2], hyp[:, :2])
                valid = neighbors != -1
                id_gt = gt[valid, 2]
                id_hyp = hyp[neighbors[valid], -1]
                mat = contingency_matrix(id_gt, id_hyp)
                purity = mat.max(axis=0).sum() / mat.sum()
                scores[i, 1] = purity
        all_scores.append((scores, paf))

    dfs = []
    for score, inds in all_scores:
        df = pd.DataFrame(score, columns=["miss", "purity"])
        df["ngraph"] = len(inds)
        dfs.append(df)
    big_df = pd.concat(dfs)
    group = big_df.groupby("ngraph")
    return (
        all_scores,
        group.agg(["mean", "std"]).T,
    )


def compare_best_and_worst_graphs(
    config,
    inference_config,
    full_data_file,
    metadata_file,
    pcutoff=0.3,
    greedy=False,
    metric="auc",
    naive_edges=None,
):
    cfg = auxiliaryfunctions.read_plainconfig(inference_config)
    cfg_temp = cfg.copy()
    cfg_temp["pcutoff"] = pcutoff

    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)

    params = _set_up_evaluation(data)
    _ = data.pop("metadata")
    to_ignore = _filter_unwanted_paf_connections(config, params["paf_graph"])
    paf_inds_best, thresholds = _get_n_best_paf_graphs(
        data,
        metadata,
        params["paf_graph"],
        ignore_inds=to_ignore,
        metric=metric,
    )
    results_best = _benchmark_paf_graphs(
        config,
        cfg_temp,
        data,
        params,
        paf_inds_best,
        greedy,
    )
    paf_inds_worst, thresholds = _get_n_best_paf_graphs(
        data,
        metadata,
        params["paf_graph"],
        which="worst",
        ignore_inds=to_ignore,
        metric=metric,
    )
    results_worst = _benchmark_paf_graphs(
        config,
        cfg_temp,
        data,
        params,
        paf_inds_worst,
        greedy,
    )
    if naive_edges is None:
        inds = sorted(set(ind for i in thresholds for ind in params["paf_graph"][i]))
        naive_edges = list(zip(inds, inds[1:]))
    naive_graph = [params["paf_graph"].index(list(edge)) for edge in naive_edges]
    paf_inds_naive, thresholds = _get_n_best_paf_graphs(
        data,
        metadata,
        params["paf_graph"],
        root=naive_graph,
        which="worst",
        ignore_inds=to_ignore,
        metric=metric,
    )
    results_naive = _benchmark_paf_graphs(
        config,
        cfg_temp,
        data,
        params,
        paf_inds_naive,
        greedy,
    )
    return pd.concat(
        (results_naive[1], results_best[1], results_worst[1]),
        keys=["naive", "best", "worst"],
    )


def _get_n_best_paf_graphs(
    data,
    metadata,
    full_graph,
    n_graphs=10,
    root=None,
    which="best",
    ignore_inds=None,
    metric="auc",
):
    if which not in ("best", "worst"):
        raise ValueError('`which` must be either "best" or "worst"')

    (_, within_test), (_, between_test) = _calc_within_between_pafs(data, metadata)

    # Handle unlabeled bodyparts...
    existing_edges = set(k for k, v in within_test.items() if v)
    if ignore_inds is not None:
        existing_edges = existing_edges.difference(ignore_inds)
    existing_edges = list(existing_edges)
    scores, thresholds = zip(
        *[
            _calc_separability(b_test, w_test, metric=metric)
            for n, (w_test, b_test) in enumerate(
                zip(within_test.values(), between_test.values())
            )
            if n in existing_edges
        ]
    )

    # Find minimal skeleton
    G = nx.Graph()
    for edge, score in zip(existing_edges, scores):
        G.add_edge(*full_graph[edge], weight=score)
    if which == "best":
        order = np.asarray(existing_edges)[np.argsort(scores)[::-1]]
        if root is None:
            root = []
            for edge in nx.maximum_spanning_edges(G, data=False):
                root.append(full_graph.index(list(edge)))
    else:
        order = np.asarray(existing_edges)[np.argsort(scores)]
        if root is None:
            root = []
            for edge in nx.minimum_spanning_edges(G, data=False):
                root.append(full_graph.index(list(edge)))

    n_edges = len(existing_edges) - len(root)
    lengths = np.linspace(0, n_edges, min(n_graphs, n_edges + 1), dtype=int)[1:]
    order = order[np.isin(order, root, invert=True)]
    paf_inds = [root]
    for length in lengths:
        paf_inds.append(root + list(order[:length]))
    return paf_inds, dict(zip(existing_edges, thresholds))


def _filter_unwanted_paf_connections(
    config,
    paf_graph,
):
    """Get rid of skeleton connections between multi and unique body parts."""
    from itertools import combinations

    cfg = auxiliaryfunctions.read_config(config)
    multi = auxfun_multianimal.extractindividualsandbodyparts(cfg)[2]
    desired = list(combinations(range(len(multi)), 2))
    return [i for i, edge in enumerate(paf_graph) if tuple(edge) not in desired]


def cross_validate_paf_graphs(
    config,
    inference_config,
    full_data_file,
    metadata_file,
    output_name="",
    pcutoff=0.3,
    greedy=False,
    calibrate=False,
    overwrite_config=False,
):
    cfg = auxiliaryfunctions.read_plainconfig(inference_config)
    cfg_temp = cfg.copy()
    cfg_temp["pcutoff"] = pcutoff

    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)

    params = _set_up_evaluation(data)
    _ = data.pop("metadata")
    to_ignore = _filter_unwanted_paf_connections(config, params["paf_graph"])
    paf_inds, thresholds = _get_n_best_paf_graphs(
        data,
        metadata,
        params["paf_graph"],
        ignore_inds=to_ignore,
    )

    if calibrate:
        trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
        calibration_file = os.path.join(
            cfg["project_path"],
            str(trainingsetfolder),
            "CollectedData_" + cfg["scorer"] + ".h5",
        )
    else:
        calibration_file = ""

    results = _benchmark_paf_graphs(
        config,
        cfg_temp,
        data,
        params,
        paf_inds,
        greedy,
        calibration_file,
    )
    # Select optimal PAF graph
    df = results[1]
    size_opt = np.argmax((1 - df.loc["miss", "mean"]) * df.loc["purity", "mean"])
    best_graph = [params["paf_graph"][ind] for ind in paf_inds[size_opt]]
    pose_config = inference_config.replace("inference_cfg", "pose_cfg")
    if not overwrite_config:
        shutil.copy(pose_config, pose_config.replace(".yaml", "_old.yaml"))
    auxiliaryfunctions.edit_config(pose_config, {"partaffinityfield_graph": best_graph})
    if output_name:
        with open(output_name, "wb") as file:
            pickle.dump([results], file)
