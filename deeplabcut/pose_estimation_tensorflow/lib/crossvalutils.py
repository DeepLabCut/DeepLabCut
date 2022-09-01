"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import pickle
import shutil
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
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


def _calc_within_between_pafs(
    data, metadata, per_edge=True, train_set_only=True,
):
    data = deepcopy(data)
    train_inds = set(metadata["data"]["trainIndices"])
    graph = data["metadata"]["PAFgraph"]
    within_train = defaultdict(list)
    within_test = defaultdict(list)
    between_train = defaultdict(list)
    between_test = defaultdict(list)
    for i, (key, dict_) in enumerate(data.items()):
        if key == "metadata":
            continue

        is_train = i in train_inds
        if train_set_only and not is_train:
            continue

        df = dict_["groundtruth"][2]
        try:
            df.drop("single", level="individuals", inplace=True)
        except KeyError:
            pass
        bpts = df.index.get_level_values("bodyparts").unique().to_list()
        coords_gt = (
            df.unstack(["individuals", "coords"])
            .reindex(bpts, level="bodyparts")
            .to_numpy()
            .reshape((len(bpts), -1, 2))
        )
        if np.isnan(coords_gt).all():
            continue

        coords = dict_["prediction"]["coordinates"][0]
        # Get animal IDs and corresponding indices in the arrays of detections
        lookup = dict()
        for i, (coord, coord_gt) in enumerate(zip(coords, coords_gt)):
            inds = np.flatnonzero(np.all(~np.isnan(coord), axis=1))
            inds_gt = np.flatnonzero(np.all(~np.isnan(coord_gt), axis=1))
            if inds.size and inds_gt.size:
                neighbors = _find_closest_neighbors(coord_gt[inds_gt], coord[inds], k=3)
                found = neighbors != -1
                lookup[i] = dict(zip(inds_gt[found], inds[neighbors[found]]))

        costs = dict_["prediction"]["costs"]
        for k, v in costs.items():
            paf = v["m1"]
            mask_within = np.zeros(paf.shape, dtype=bool)
            s, t = graph[k]
            if s not in lookup or t not in lookup:
                continue
            lu_s = lookup[s]
            lu_t = lookup[t]
            common_id = set(lu_s).intersection(lu_t)
            for id_ in common_id:
                mask_within[lu_s[id_], lu_t[id_]] = True
            within_vals = paf[mask_within]
            between_vals = paf[~mask_within]
            if is_train:
                within_train[k].extend(within_vals)
                between_train[k].extend(between_vals)
            else:
                within_test[k].extend(within_vals)
                between_test[k].extend(between_vals)
    if not per_edge:
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
    margin=0,
    symmetric_kpts=None,
    split_inds=None,
):
    metadata = data.pop("metadata")
    multi_bpts_orig = auxfun_multianimal.extractindividualsandbodyparts(config)[2]
    multi_bpts = [j for j in metadata["all_joints_names"] if j in multi_bpts_orig]
    n_multi = len(multi_bpts)
    data_ = {"metadata": metadata}
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
    mask_multi = idx.get_level_values("individuals") != "single"
    if not mask_multi.all():
        idx = idx.drop("single", level="individuals")
    individuals = idx.get_level_values("individuals").unique()
    n_individuals = len(individuals)
    map_ = dict(zip(individuals, range(n_individuals)))

    # Form ground truth beforehand
    ground_truth = []
    for i, imname in enumerate(image_paths):
        temp = data[imname]["groundtruth"][2].reindex(multi_bpts, level="bodyparts")
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
    n_graphs = len(paf_inds)
    all_scores = []
    all_metrics = []
    all_assemblies = []
    for j, paf in enumerate(paf_inds, start=1):
        print(f"Graph {j}|{n_graphs}")
        ass.paf_inds = paf
        ass.assemble()
        all_assemblies.append((ass.assemblies, ass.unique, ass.metadata["imnames"]))
        if split_inds is not None:
            oks = []
            for inds in split_inds:
                ass_gt = {k: v for k, v in ass_true_dict.items() if k in inds}
                oks.append(
                    evaluate_assembly(
                        ass.assemblies,
                        ass_gt,
                        oks_sigma,
                        margin=margin,
                        symmetric_kpts=symmetric_kpts,
                        greedy_matching=inference_cfg.get("greedy_oks", False),
                    )
                )
        else:
            oks = evaluate_assembly(
                ass.assemblies,
                ass_true_dict,
                oks_sigma,
                margin=margin,
                symmetric_kpts=symmetric_kpts,
                greedy_matching=inference_cfg.get("greedy_oks", False),
            )
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
                scores[i, 0] = max(0, (n_dets - hyp.shape[0]) / n_dets)
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
    return (all_scores, group.agg(["mean", "std"]).T, all_metrics, all_assemblies)


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

    (within_train, _), (between_train, _) = _calc_within_between_pafs(
        data, metadata, train_set_only=True,
    )
    # Handle unlabeled bodyparts...
    existing_edges = set(k for k, v in within_train.items() if v)
    if ignore_inds is not None:
        existing_edges = existing_edges.difference(ignore_inds)
    existing_edges = list(existing_edges)

    if not any(between_train.values()):
        # Only 1 animal, let us return the full graph indices only
        return ([existing_edges], dict(zip(existing_edges, [0] * len(existing_edges))))

    scores, _ = zip(
        *[
            _calc_separability(between_train[n], within_train[n], metric=metric)
            for n in existing_edges
        ]
    )

    # Find minimal skeleton
    G = nx.Graph()
    for edge, score in zip(existing_edges, scores):
        if np.isfinite(score):
            G.add_edge(*full_graph[edge], weight=score)
    if which == "best":
        order = np.asarray(existing_edges)[np.argsort(scores)[::-1]]
        if root is None:
            root = []
            for edge in nx.maximum_spanning_edges(G, data=False):
                root.append(full_graph.index(sorted(edge)))
    else:
        order = np.asarray(existing_edges)[np.argsort(scores)]
        if root is None:
            root = []
            for edge in nx.minimum_spanning_edges(G, data=False):
                root.append(full_graph.index(sorted(edge)))

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
    oks_sigma=0.1,
    margin=0,
    greedy=False,
    add_discarded=True,
    calibrate=False,
    overwrite_config=True,
    n_graphs=10,
    paf_inds=None,
    symmetric_kpts=None,
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
    best_graphs = _get_n_best_paf_graphs(
        data, metadata, params["paf_graph"], ignore_inds=to_ignore, n_graphs=n_graphs,
    )
    paf_scores = best_graphs[1]
    if paf_inds is None:
        paf_inds = best_graphs[0]

    if calibrate:
        trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
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
        oks_sigma=oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
        calibration_file=calibration_file,
        split_inds=[metadata["data"]["trainIndices"], metadata["data"]["testIndices"],],
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
    return results[:3], paf_scores, results[3][size_opt]
