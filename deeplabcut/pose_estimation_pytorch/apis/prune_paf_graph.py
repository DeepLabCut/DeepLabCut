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

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

import deeplabcut.core.metrics as metrics
import deeplabcut.pose_estimation_pytorch.apis.utils as utils
import deeplabcut.pose_estimation_pytorch.data as data
import deeplabcut.pose_estimation_pytorch.models.predictors as predictors
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deeplabcut.core.crossvalutils import find_closest_neighbors
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.models.predictors.paf_predictor import Graph


@torch.no_grad()
def benchmark_paf_graphs(
    loader: data.Loader,
    snapshot_path: Path,
    verbose: bool = False,
    overwrite: bool = False,
    update_config: bool = True,
) -> list[dict]:
    """Prunes the PAF graph to maximize performance

    Args:
        loader: The loader for the model to prune.
        snapshot_path: The path to the snapshot with which to prune the model.
        verbose: Verbose pruning of the model.
        overwrite: Whether to overwrite the graph if it was already pruned.
        update_config: Whether to update the model configuration with the pruned graph.

    Returns:
        A list of dictionaries containing results for each pruned graph.

        If the graph was already pruned, a single element is returned with an
        "edges_to_keep" key,  containing the indices of edges to keep in the graph.

        Otherwise, a list of graphs that were evaluated is returned, with "key_metric",
        "edges_to_keep" and "metrics" keys. The list is sorted by "key_metric" (which
        is pose mAP).
    """
    runner = utils.get_pose_inference_runner(loader.model_cfg, snapshot_path)
    device = runner.device
    preprocessor = runner.preprocessor
    model = runner.model
    predictor = model.heads.bodypart.predictor

    # only benchmark the PAF graph if the PAF indices contain all edges
    if not overwrite and len(predictor.edges_to_keep) < len(predictor.graph):
        return [dict(edges_to_keep=predictor.edges_to_keep)]

    model.to(device)
    model.eval()

    if not isinstance(predictor, predictors.PartAffinityFieldPredictor):
        raise ValueError(f"Predictor should be a PartAffinityFieldPredictor.")

    if verbose:
        print("-------------------------------------------------")
        print("Benchmarking different Part-Affinity Field Graphs")
        print("  (1/3) Obtaining the best graph candidates")

    gt_train = loader.ground_truth_keypoints("train")
    best_paf_edges, _ = get_n_best_paf_graphs(
        model,
        gt_train,
        preprocessor,
        device,
        predictor.graph,
        n_graphs=10,
    )

    if verbose:
        print("  (2/3) Running test inference")

    gt_test = loader.ground_truth_keypoints("test")
    images_test = [img_path for img_path in gt_test]

    predictions = {graph_id: {} for graph_id in range(len(best_paf_edges))}
    with torch.no_grad():
        for image_path in tqdm(images_test):
            image, _ = preprocessor(image_path, {})
            outputs = model(image.to(device))
            for graph_id, edges in enumerate(best_paf_edges):
                predictor.set_paf_edges_to_keep(edges)
                pred_pose = model.get_predictions(outputs)["bodypart"]["poses"]
                predictions[graph_id][image_path] = pred_pose.cpu().numpy()[0]

    if verbose:
        print("  (3/3) Evaluating Graphs")

    results = []
    for graph_id, pred_pose in predictions.items():
        edges_to_keep = [int(i) for i in best_paf_edges[graph_id]]
        graph_metrics = metrics.compute_metrics(
            gt_test,
            pred_pose,
            single_animal=False,
            pcutoff=0.6,
        )
        results.append(
            dict(
                edges_to_keep=edges_to_keep,
                key_metric=graph_metrics["mAP"],
                metrics=graph_metrics,
            )
        )

        if verbose:
            print("    ---")
            print(f"    |Graph {graph_id}: {len(edges_to_keep)} edges")
            print(f"    |   mAP: {graph_metrics['mAP']}")
            print(f"    |   mAR: {graph_metrics['mAR']}")
            print(f"    |   edges: {edges_to_keep}")
            print()

    results = list(sorted(results, key=lambda r: 1 - r["key_metric"]))

    if update_config and len(results) > 0:
        best_results = results[0]
        best_edges = best_results["edges_to_keep"]
        graph_metrics = best_results["metrics"]

        if verbose:
            print("Selecting the following Graph")
            print(60 * "-")
            print(f"|Graph with {len(best_edges)} edges")
            print(f"|   mAP: {graph_metrics['mAP']}")
            print(f"|   mAR: {graph_metrics['mAR']}")
            print(f"|   edges: {best_edges}")
            print()

        # update the edges to keep in the PyTorch configuration file
        loader.update_model_cfg(
            {"model.heads.bodypart.predictor.edges_to_keep": best_edges}
        )

        # update the edges indices
        test_config = loader.model_folder.parent / "test" / "pose_cfg.yaml"
        auxiliaryfunctions.edit_config(str(test_config), dict(paf_best=best_edges))

    return results


def _calc_separability(
    vals_left: np.ndarray,
    vals_right: np.ndarray,
    n_bins: int = 101,
    metric: str = "jeffries",
    max_sensitivity: bool = False,
) -> tuple[float, float]:
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


@torch.no_grad()
def compute_within_between_paf_costs(
    model: PoseModel,
    ground_truth: dict[str, np.ndarray],
    preprocessor: data.Preprocessor,
    device: str,
) -> tuple[defaultdict, defaultdict]:
    predictor = model.heads.bodypart.predictor
    images = [img_path for img_path in ground_truth]

    within = defaultdict(list)
    between = defaultdict(list)
    for image_path in tqdm(images):
        image, _ = preprocessor(image_path, {})
        outputs = model(image.to(device))
        preds = model.get_predictions(outputs)["bodypart"]["preds"][0]
        gt_pose_with_vis = ground_truth[image_path].transpose((1, 0, 2))

        # mask non-visible keypoints
        gt_pose = gt_pose_with_vis[..., :2].copy()
        gt_pose[gt_pose_with_vis[..., 2] <= 0] = np.nan

        if np.isnan(gt_pose).all():
            continue

        coords_pred = preds["coordinates"][0]
        costs_pred = preds["costs"]

        # Get animal IDs and corresponding indices in the arrays of detections
        lookup = dict()
        for i, (coord_pred, coord_gt) in enumerate(zip(coords_pred, gt_pose)):
            inds = np.flatnonzero(np.all(~np.isnan(coord_pred), axis=1))
            inds_gt = np.flatnonzero(np.all(~np.isnan(coord_gt), axis=1))
            if inds.size and inds_gt.size:
                neighbors = find_closest_neighbors(
                    coord_gt[inds_gt], coord_pred[inds], k=3
                )
                found = neighbors != -1
                lookup[i] = dict(zip(inds_gt[found], inds[neighbors[found]]))

        for k, v in costs_pred.items():
            paf = v["m1"]
            mask_within = np.zeros(paf.shape, dtype=bool)
            s, t = predictor.graph[k]
            if s not in lookup or t not in lookup:
                continue
            lu_s = lookup[s]
            lu_t = lookup[t]
            common_id = set(lu_s).intersection(lu_t)
            for id_ in common_id:
                mask_within[lu_s[id_], lu_t[id_]] = True
            within_vals = paf[mask_within]
            between_vals = paf[~mask_within]
            within[k].extend(within_vals)
            between[k].extend(between_vals)

    return within, between


def get_n_best_paf_graphs(
    model: PoseModel,
    ground_truth: dict[str, np.ndarray],
    preprocessor: data.Preprocessor,
    device: str,
    full_graph: Graph,
    root_edges: list[int] | None = None,
    n_graphs: int = 10,
    metric: str = "auc",
) -> tuple[list[list[int]], dict[int, float]]:
    return_preds = model.heads.bodypart.predictor.return_preds
    model.heads.bodypart.predictor.return_preds = True

    within_train, between_train = compute_within_between_paf_costs(
        model, ground_truth, preprocessor, device
    )
    existing_edges = list(set(k for k, v in within_train.items() if v))

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

    order = np.asarray(existing_edges)[np.argsort(scores)[::-1]]
    if root_edges is None:
        root_edges = []
        for edge in nx.maximum_spanning_edges(G, data=False):
            root_edges.append(full_graph.index(sorted(edge)))

    n_edges = len(existing_edges) - len(root_edges)
    lengths = np.linspace(0, n_edges, min(n_graphs, n_edges + 1), dtype=int)[1:]
    order = order[np.isin(order, root_edges, invert=True)]
    best_edges = [root_edges]
    for length in lengths:
        best_edges.append(root_edges + list(order[:length]))

    model.heads.bodypart.predictor.return_preds = return_preds
    return best_edges, dict(zip(existing_edges, scores))
