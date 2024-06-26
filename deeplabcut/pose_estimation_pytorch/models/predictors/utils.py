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

import albumentations as A
import networkx as nx
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplabcut.core.crossvalutils import find_closest_neighbors
from deeplabcut.pose_estimation_pytorch import Loader
from deeplabcut.pose_estimation_pytorch.metrics.scoring import (
    get_scores,
    pair_predicted_individuals_with_gt,
)
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.models.predictors.paf_predictor import Graph


def _calc_separability(
    vals_left: ArrayLike,
    vals_right: ArrayLike,
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


def get_n_best_paf_graphs(
    model: PoseModel,
    train_dataloader: DataLoader,
    full_graph: Graph,
    root_edges: list[int] | None = None,
    n_graphs: int = 10,
    metric: str = "auc",
    device: str = "cuda",
) -> tuple[list[list[int]], dict[int, float]]:
    within_train, between_train = compute_within_between_paf_costs(
        model, train_dataloader, device
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
    return best_edges, dict(zip(existing_edges, scores))


def compute_within_between_paf_costs(
    model: PoseModel, dataloader: DataLoader, device: str = "cuda"
) -> tuple[defaultdict[list]]:
    model.to(device)
    predictor = model.heads.bodypart.predictor
    within = defaultdict(list)
    between = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["image"].to(device)
            preds = model.get_predictions(model(inputs))["bodypart"]

            for coords_gt, preds_ in zip(
                batch["annotations"]["keypoints"], preds["preds"]
            ):
                coords_gt = coords_gt.permute(1, 0, 2).detach().cpu().numpy()
                if np.isnan(coords_gt).all():
                    continue

                coords_pred = preds_["coordinates"][0]
                costs_pred = preds_["costs"]

                # Get animal IDs and corresponding indices in the arrays of detections
                lookup = dict()
                for i, (coord_pred, coord_gt) in enumerate(zip(coords_pred, coords_gt)):
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


def benchmark_paf_graphs(
    model: PoseModel,
    loader: Loader,
    transform: A.BaseCompose,
    batch_size: int = 8,
    device: str = "cuda",
) -> tuple[list[dict[str, float]], list[dict[str, NDArray]], list[list[int]]]:
    predictor = model.heads.bodypart.predictor
    train_dataset = loader.create_dataset(mode="train", task="BU", transform=transform)
    valid_dataset = loader.create_dataset(mode="test", task="BU", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    best_paf_edges, _ = get_n_best_paf_graphs(
        model, train_dataloader, predictor.graph, device=device
    )
    poses_gt = loader.ground_truth_keypoints("test")
    results = []
    poses = []
    for edges in best_paf_edges:
        predictor.edges_to_keep = predictor.assembler.paf_inds = edges
        paths = []
        poses_ = []
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                paths.extend(batch["path"])
                inputs = batch["image"].to(device)
                # FIXME We can do better than the repetition below
                preds = model.get_predictions(model(inputs))["bodypart"]
                poses_.extend(preds["poses"])
        poses_ = torch.stack(poses_).detach().cpu().numpy()
        poses_ = dict(zip(paths, poses_))
        poses_ = pair_predicted_individuals_with_gt(poses_, poses_gt)
        poses.append(poses_)
        results.append(get_scores(poses_, poses_gt))
    return results, poses, best_paf_edges
