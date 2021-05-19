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
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import (
    Assembler,
    evaluate_assembly,
    _parse_ground_truth_data,
)
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


def _form_original_path(path):
    root, filename = os.path.split(path)
    base, ext = os.path.splitext(filename)
    return os.path.join(root, filename.split("c")[0] + ext)


def _unsorted_unique(array):
    _, inds = np.unique(array, return_index=True)
    return np.asarray(array)[np.sort(inds)]


def _rebuild_uncropped_metadata(metadata, image_paths, output_name=""):
    train_inds_orig = set(metadata["data"]["trainIndices"])
    train_inds, test_inds = [], []
    for k, (_, group) in tqdm(enumerate(groupby(image_paths, _form_original_path))):
        if image_paths.index(next(group)) in train_inds_orig:
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


def _rebuild_uncropped_data(data, params, output_name=""):
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
                (n_individuals, len(temp) // n_individuals, 4 + n_individuals), np.nan
            )  # Hold x, y, prob, dist, ids
            costs = dict()
            shape = n_individuals, n_individuals
            for ind in params["paf"]:
                costs[ind] = {"m1": np.zeros(shape), "distance": np.full(shape, np.inf)}
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
                    try:
                        ids_pred = data[imname]["prediction"]["identity"]
                    except KeyError:
                        ids_pred = None
                    map_ = dict()
                    for n, bpt in enumerate(ref_gt_):
                        xy_gt, inds_gt = ref_gt_[bpt]
                        ind = bodyparts.index(bpt)
                        xy = coords_pred[ind]
                        prob = probs_pred[ind]
                        ids = None if ids_pred is None else ids_pred[ind]
                        if inds_gt.size and xy.size:
                            xy_trans = xy - all_trans[i]
                            d = cdist(xy_gt[inds_gt], xy_trans)
                            rows, cols = linear_sum_assignment(d)
                            probs_ = prob[cols]
                            ids_ = ids[cols] if ids is not None else None
                            dists_ = d[rows, cols]
                            inds_rows = inds_gt[rows]
                            map_[n] = inds_rows, cols
                            is_free = np.isnan(ref_pred[inds_rows, n]).all(axis=1)
                            closer = dists_ < ref_pred[inds_rows, n, 3]
                            mask = np.logical_or(is_free, closer)
                            if mask.any():
                                coords_ = xy_trans[cols]
                                sl = inds_rows[mask]
                                ref_pred[sl, n, :2] = coords_[mask]
                                ref_pred[sl, n, 2] = probs_[mask].squeeze()
                                ref_pred[sl, n, 3] = dists_[mask]
                                if ids_ is not None:
                                    ref_pred[sl, n, 4:] = ids_[mask]
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
            identities = ref_pred_[..., 4:]
            if ~np.all(np.isnan(identities)):
                pred_dict["prediction"]["identity"] = identities
            data_new[basename] = pred_dict

    image_paths = list(data_new)
    data_new["metadata"] = data["metadata"]

    if output_name:
        with open(output_name, "wb") as file:
            pickle.dump(data_new, file)

    return data_new, image_paths


def _rebuild_uncropped_in(base_folder,):
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
    vals_left, vals_right, n_bins=101, metric="jeffries", max_sensitivity=False
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


def _calc_within_between_pafs(data, metadata, per_bodypart=True, train_set_only=True):
    train_inds = set(metadata["data"]["trainIndices"])
    within_train = defaultdict(list)
    within_test = defaultdict(list)
    between_train = defaultdict(list)
    between_test = defaultdict(list)
    mask_diag = None
    for i, (key, dict_) in enumerate(data.items()):
        if key == "metadata":
            continue
        is_train = i in train_inds
        if train_set_only and not is_train:
            continue
        costs = dict_["prediction"]["costs"]
        for k, v in costs.items():
            paf = v["m1"]
            nonzero = paf != 0
            if mask_diag is None:
                mask_diag = np.eye(paf.shape[0], dtype=bool)
            within_vals = paf[np.logical_and(mask_diag, nonzero)]
            between_vals = paf[np.logical_and(~mask_diag, nonzero)]
            if is_train:
                within_train[k].extend(within_vals)
                between_train[k].extend(between_vals)
            else:
                within_test[k].extend(within_vals)
                between_test[k].extend(between_vals)
    if not per_bodypart:
        within_train = np.concatenate([*within_train.values()])
        within_test = np.concatenate([*within_test.values()])
        between_train = np.concatenate([*between_train.values()])
        between_test = np.concatenate([*between_test.values()])
    return (within_train, within_test), (between_train, between_test)


def _benchmark_paf_graphs(
    config,
    inference_cfg,
    data,
    paf_inds,
    greedy=False,
    add_discarded=True,
    identity_only=False,
    calibration_file="",
    oks_sigma=0.1,
):
    n_multi = len(auxfun_multianimal.extractindividualsandbodyparts(config)[2])
    data_ = {"metadata": data.pop("metadata")}
    for k, v in data.items():
        data_[k] = v["prediction"]
    ass = Assembler(
        data_,
        max_n_individuals=inference_cfg["topktoretain"],
        n_multibodyparts=n_multi,
        greedy=greedy,
        pcutoff=inference_cfg.get("pcutoff", 0.1),
        min_affinity=inference_cfg.get("pafthreshold", 0.1),
        add_discarded=add_discarded,
        identity_only=identity_only,
    )
    if calibration_file:
        ass.calibrate(calibration_file)

    params = ass.metadata
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
    temp = np.ones((*ground_truth.shape[:2], 3))
    temp[..., :2] = ground_truth
    temp = temp.reshape((temp.shape[0], n_individuals, -1, 3))
    ass_true_dict = _parse_ground_truth_data(temp)
    ids = np.vectorize(map_.get)(idx.get_level_values("individuals").to_numpy())
    ground_truth = np.insert(ground_truth, 2, ids, axis=2)

    # Assemble animals on the full set of detections
    paf_inds = sorted(paf_inds, key=len)
    paf_graph = ass.graph
    n_graphs = len(paf_inds)
    all_scores = []
    all_metrics = []
    for j, paf in enumerate(paf_inds, start=1):
        print(f"Graph {j}|{n_graphs}")
        graph = [paf_graph[i] for i in paf]
        ass.paf_inds = paf
        ass.graph = graph
        ass.assemble()
        oks = evaluate_assembly(ass.assemblies, ass_true_dict, oks_sigma)
        all_metrics.append(oks)
        scores = np.full((len(image_paths), 2), np.nan)
        for i, imname in enumerate(tqdm(image_paths)):
            gt = ground_truth[i]
            gt = gt[~np.isnan(gt).any(axis=1)]
            if len(np.unique(gt[:, 2])) < 2:  # Only consider frames with 2+ animals
                continue

            # Count the number of unassembled bodyparts
            n_dets = len(gt)
            animals = ass.assemblies.get(i)
            if animals is None:
                if n_dets:
                    scores[i, 0] = 1
            else:
                animals = [
                    np.c_[animal.data, np.ones(animal.data.shape[0]) * n]
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
    return (all_scores, group.agg(["mean", "std"]).T, all_metrics)


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

    (within_train, within_test), (between_train, _) = _calc_within_between_pafs(
        data, metadata, train_set_only=False
    )

    # Handle unlabeled bodyparts...
    existing_edges = set(k for k, v in within_test.items() if v)
    if ignore_inds is not None:
        existing_edges = existing_edges.difference(ignore_inds)
    existing_edges = list(existing_edges)
    scores, thresholds = zip(
        *[
            _calc_separability(b_train, w_train, metric=metric)
            for n, (w_train, b_train) in enumerate(
                zip(within_train.values(), between_train.values())
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
    return paf_inds, dict(zip(existing_edges, scores))


def cross_validate_paf_graphs(
    config,
    inference_config,
    full_data_file,
    metadata_file,
    output_name="",
    pcutoff=0.1,
    greedy=False,
    add_discarded=True,
    calibrate=False,
    overwrite_config=True,
):
    cfg = auxiliaryfunctions.read_config(config)
    inf_cfg = auxiliaryfunctions.read_plainconfig(inference_config)
    inf_cfg_temp = inf_cfg.copy()
    inf_cfg_temp["pcutoff"] = pcutoff

    with open(full_data_file, "rb") as file:
        data = pickle.load(file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)

    params = _set_up_evaluation(data)
    to_ignore = auxfun_multianimal.filter_unwanted_paf_connections(
        cfg, params["paf_graph"]
    )
    paf_inds, paf_scores = _get_n_best_paf_graphs(
        data, metadata, params["paf_graph"], ignore_inds=to_ignore
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
        cfg,
        inf_cfg_temp,
        data,
        paf_inds,
        greedy,
        add_discarded,
        calibration_file=calibration_file,
    )
    # Select optimal PAF graph
    df = results[1]
    size_opt = np.argmax((1 - df.loc["miss", "mean"]) * df.loc["purity", "mean"])
    pose_config = inference_config.replace("inference_cfg", "pose_cfg")
    if not overwrite_config:
        shutil.copy(pose_config, pose_config.replace(".yaml", "_old.yaml"))
    inds = list(paf_inds[size_opt])
    auxiliaryfunctions.edit_config(
        pose_config, {"paf_best": [int(ind) for ind in inds]}
    )
    if output_name:
        with open(output_name, "wb") as file:
            pickle.dump([results], file)
