"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import heapq
import itertools
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt, erf
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist
from scipy.stats import gaussian_kde, chi2
from tqdm import tqdm
from typing import Tuple


###################################
#### auxiliaryfunctions
###################################


def distance(v, w):
    return np.sqrt(np.sum((v - w) ** 2))


def minmax(array, slack=10):
    return np.nanmin(array) - slack, np.nanmax(array) + slack


def individual2boundingbox(cfg, animals, X1=0):
    boundingboxes = np.zeros((len(animals), 5)) * np.nan

    for id, individual in enumerate(animals):
        boundingboxes[id, 0:4:2] = minmax(
            individual[::3] + X1, slack=cfg.boundingboxslack
        )
        boundingboxes[id, 1:4:2] = minmax(individual[1::3], slack=cfg.boundingboxslack)
        boundingboxes[id, 4] = np.nanmean(
            individual[2::3]
        )  # average likelihood of all bpts

    return boundingboxes


def calc_bboxes_from_keypoints(data, slack=0, offset=0):
    if data.shape[-1] < 3:
        raise ValueError("Data should be of shape (n_animals, n_bodyparts, 3)")

    if data.ndim != 3:
        data = np.expand_dims(data, axis=0)
    bboxes = np.full((data.shape[0], 5), np.nan)
    bboxes[:, :2] = np.nanmin(data[..., :2], axis=1) - slack  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(data[..., :2], axis=1) + slack  # X2, Y2
    bboxes[:, -1] = np.nanmean(data[..., 2])  # Average confidence
    bboxes[:, [0, 2]] += offset
    return bboxes


##########################################################
#### conversion & greedy bodypart matching code
##########################################################


def convertdetectiondict2listoflist(dataimage, BPTS, withid=False, evaluation=False):
    """Arranges data into list of list with the following entries:
    [(x, y, score, global index of detection)] (all detections per bodypart).

    Also includes id if available. [x,y,score,id,global id]"""

    if evaluation:
        detectedcoordinates = dataimage["prediction"]["coordinates"][0]
        detectedlikelihood = dataimage["prediction"]["confidence"]
    else:
        detectedcoordinates = dataimage["coordinates"][0]
        detectedlikelihood = dataimage["confidence"]

    all_detections = []
    detection_counter = 0
    for bpt in BPTS:
        if withid:  # (x, y, likelihood, identity)
            detections_with_likelihood = list(
                zip(
                    detectedcoordinates[bpt][:, 0],
                    detectedcoordinates[bpt][:, 1],
                    detectedlikelihood[bpt].flatten(),
                    np.argmax(dataimage["identity"][bpt], 1),
                )
            )
        else:  # (x, y, likelihood)
            detections_with_likelihood = list(
                zip(
                    detectedcoordinates[bpt][:, 0],
                    detectedcoordinates[bpt][:, 1],
                    detectedlikelihood[bpt].flatten(),
                )
            )
        idx = range(
            detection_counter, detection_counter + len(detections_with_likelihood)
        )
        all_detections.append(
            [detections_with_likelihood[i] + (idx[i],) for i in range(len(idx))]
        )
        detection_counter += len(detections_with_likelihood)

    return all_detections


def extract_strong_connections(
    cfg,
    dataimage,
    all_detections,
    iBPTS,
    partaffinityfield_graph,
    PAF,
    paf_thresholds,
    lowerbound=None,
    upperbound=None,
    evaluation=False,
):
    """Auxiliary function;  Returns list of connections (limbs) of a particular type.

    Specifically, per edge a list containing is returned: [index start (global), index stop (global) score, score with detection likelihoods, index start (local), index stop (local)]
    Thereby, index start and stop refer to the index of the bpt from the beginning to the end of the edge. Local index refers to the index within the list of bodyparts, and
    global to the index for all the detections (in all_detections)

    Parameters
    ----------
    cfg : dictionary
        configuation file for inference parameters

    dataimage: dict
        predictions (detections + paf scores) for a particular image

    all_detections: list of lists
        result of convertdetectiondict2listoflist

    iBPTS: list
        source of bpts.

    partaffinityfield_graph: list of part affinity matchconnections

    PAF: PAF is a subset of the indices that should be used in the partaffinityfield_graph

    paf_thresholds : list of floats
        List holding the PAF thresholds of individual graph edges
    """
    all_connections = []
    missing_connections = []
    costs = dataimage["prediction"]["costs"] if evaluation else dataimage["costs"]
    for edge in range(len(partaffinityfield_graph)):
        a, b = partaffinityfield_graph[edge]
        paf_threshold = paf_thresholds[edge]
        cand_a = all_detections[
            iBPTS[a]
        ]  # convert bpt index to the one in all_detections!
        cand_b = all_detections[iBPTS[b]]
        n_a = len(cand_a)
        n_b = len(cand_b)
        if n_a != 0 and n_b != 0:
            scores = costs[PAF[edge]][cfg["method"]]
            dist = costs[PAF[edge]]["distance"]
            connection_candidate = []
            for i in range(n_a):
                si = cand_a[i][2]  # likelihoood for detection
                for j in range(n_b):
                    sj = cand_b[j][2]  # likelihoood for detection
                    score_with_dist_prior = abs(scores[i, j])
                    d = dist[i, j]
                    if lowerbound is None and upperbound is None:
                        if (
                            score_with_dist_prior > paf_threshold
                            and cfg["distnormalizationLOWER"] <= d < cfg["distnormalization"]
                            and si * sj > cfg["detectionthresholdsquare"]
                        ):
                            connection_candidate.append(
                                [
                                    i,
                                    j,
                                    score_with_dist_prior,
                                    d,
                                ]
                            )
                    else:
                        if (
                            score_with_dist_prior > paf_threshold
                            and lowerbound[edge] <= d < upperbound[edge]
                            and si * sj > cfg["detectionthresholdsquare"]
                        ):
                            connection_candidate.append(
                                [
                                    i,
                                    j,
                                    score_with_dist_prior,
                                    d,
                                ]
                            )

            # sort candidate connections by score
            connection_candidate = sorted(
                connection_candidate, key=lambda x: x[2], reverse=True
            )
            connection = []
            i_seen = set()
            j_seen = set()
            nrows = min(n_a, n_b)
            for i, j, score, d in connection_candidate:
                if i not in i_seen and j not in j_seen:
                    i_seen.add(i)
                    j_seen.add(j)
                    ii = int(cand_a[i][-1])  # global index!
                    jj = int(cand_b[j][-1])
                    connection.append([ii, jj, score, d, i, j])
                    if len(connection) == nrows:
                        break
            all_connections.append(connection)
        else:
            missing_connections.append(edge)
            all_connections.append([])
    return all_connections, missing_connections


def _nest_detections_in_arrays(
    data_dict,
):
    all_detections = []
    coords = data_dict["coordinates"][0]
    if all(np.isnan(xy).all() for xy in coords):
        return all_detections
    conf = data_dict["confidence"]
    ids = data_dict.get("identity", [[None]] * len(coords))
    count = 0
    for xy, p, id_ in zip(coords, conf, ids):
        if not xy.size:
            all_detections.append([])
            continue
        temp = np.c_[(xy, p, np.arange(count, count + len(xy)))]
        if id_[0] is not None:
            temp = np.insert(temp, -1, np.argmax(id_, axis=1), axis=1)
        count += len(xy)
        all_detections.append(temp)
    return all_detections


def _extract_strong_connections(
    all_detections,
    graph,
    paf_inds,
    costs,
    pcutoff=0.1,
    dist_funcs=None,
    method="m1",
):
    all_connections = []
    for i, (a, b) in enumerate(graph):
        dets_a = all_detections[a]
        dets_b = all_detections[b]
        if not (np.any(dets_a) and np.any(dets_b)):
            continue
        dist = costs[paf_inds[i]]["distance"]
        if np.isinf(dist).all():
            continue

        # Filter out low confidence detections
        # inds_a = np.argsort(dets_a[:, 2])[::-1][:max_ndets]
        # inds_b = np.argsort(dets_b[:, 2])[::-1][:max_ndets]
        # keep_a = inds_a[dets_a[inds_a, 2] >= pcutoff]
        # keep_b = inds_b[dets_b[inds_b, 2] >= pcutoff]
        # sl = np.ix_(inds_a, inds_b)
        # w = costs[paf_inds[i]][method][sl]
        w = costs[paf_inds[i]][method].copy()
        w[np.isnan(w)] = 0  # FIXME Why is it even NaN??
        # dist = dist[sl]
        # dets_a = dets_a[inds_a]
        # dets_b = dets_b[inds_b]
        # if dist_funcs:
        #     w += dist_funcs[i](dist)
        row_inds, col_inds = linear_sum_assignment(w, maximize=True)
        connections = []
        for row, col in zip(row_inds, col_inds):
            d = dist[row, col]
            if dist_funcs is not None:
                bounds = dist_funcs[i]
                if not bounds[0] <= d <= bounds[1]:
                    continue
            if not (dets_a[row, 2] >= pcutoff and dets_b[col, 2] >= pcutoff):
                continue
            connections.append(
                [int(dets_a[row, -1]), int(dets_b[col, -1]), w[row, col], d, row, col]
            )
        all_connections.append(connections)
    return all_connections


def _link_detections(
    all_detections,
    all_connections,
    max_individuals,
    use_springs=False,
):
    n_bodyparts = len(all_detections)
    candidates = []
    missing_detections = []
    for i, array in enumerate(all_detections):
        if not np.any(array):
            missing_detections.append(i)
        else:
            array = np.c_[(array, [i] * len(array))]
            for row in array:
                candidates.append(row)
    candidates = np.asarray(candidates)
    all_inds = candidates[:, -1].astype(int)

    # idx = np.argsort([sum(i[2] for i in sub) / len(sub) for sub in all_connections])[::-1]
    # subsets = np.empty((0, n_bodyparts))
    # ambiguous = []
    # for i in idx:
    #     connections = all_connections[i]
    #     for connection in connections:
    #         nodes = list(connection[:2])
    #         inds = all_inds[nodes]
    #         mask = np.any(subsets[:, inds] == nodes, axis=1)
    #         subset_inds = np.flatnonzero(mask)
    #         found = subset_inds.size
    #         if found == 1:
    #             subset = subsets[subset_inds[0]]
    #             if np.all(subset[inds] == nodes):  # Nodes correctly found in the same subset
    #                 continue
    #             if np.all(subset[inds] != -1):
    #                 ambiguous.append(connection)
    #             elif subset[inds[0]] == -1:
    #                 subset[inds[0]] = nodes[0]
    #             else:
    #                 subset[inds[1]] = nodes[1]
    #         elif found == 2:
    #             membership = np.sum(subsets[subset_inds] >= 0, axis=0)
    #             if not np.any(membership == 2):
    #                 subsets = _merge_disjoint_subsets(subsets, *subset_inds)
    #             else:
    #                 ambiguous.append(connection)
    #         else:
    #             row = -1 * np.ones(n_bodyparts)
    #             row[inds] = nodes
    #             subsets = np.vstack((subsets, row))

    G = nx.Graph()
    G.add_weighted_edges_from(
        [(a, b, c) for arrays in all_connections for a, b, c, *_ in arrays],
    )
    subsets0 = np.empty((0, n_bodyparts))
    # Fill the subsets with unambiguous skeletons
    for chain in list(nx.connected_components(G)):
        if len(chain) == n_bodyparts - len(missing_detections):
            row = -1 * np.ones(n_bodyparts)
            nodes = list(chain)
            row[all_inds[nodes]] = nodes
            subsets0 = np.vstack((subsets0, row))
            G.remove_nodes_from(nodes)

    # Sort connections in descending order of part affinity
    connections = sorted(
        G.edges.data('weight'),
        key=lambda x: x[2],
        reverse=True
    )
    subsets = np.empty((0, n_bodyparts))
    ambiguous = []
    for connection in connections:
        nodes = list(connection[:2])
        inds = all_inds[nodes]
        mask = np.any(subsets[:, inds] == nodes, axis=1)
        subset_inds = np.flatnonzero(mask)
        found = subset_inds.size
        if found == 1:
            subset = subsets[subset_inds[0]]
            if np.all(subset[inds] == nodes):  # Nodes correctly found in the same subset
                continue
            if np.all(subset[inds] != -1):
                ambiguous.append(connection)
            elif subset[inds[0]] == -1:
                subset[inds[0]] = nodes[0]
            else:
                subset[inds[1]] = nodes[1]
        elif found == 2:
            membership = np.sum(subsets[subset_inds] >= 0, axis=0)
            if not np.any(membership == 2):
                subsets = _merge_disjoint_subsets(subsets, *subset_inds)
            else:
                ambiguous.append(connection)
        else:
            row = -1 * np.ones(n_bodyparts)
            row[inds] = nodes
            subsets = np.vstack((subsets, row))

    nrows = len(subsets)
    left = max_individuals - len(subsets0)
    if nrows > left > 0:
        subsets = subsets[np.argsort(np.sum(subsets == -1, axis=1))]
        for nrow in range(nrows - 1, left - 1, -1):
            mask = (subsets[:left] >= 0).astype(int)
            row = subsets[nrow]
            mask2 = (row >= 0).astype(int)
            temp = mask + mask2
            free_rows = np.flatnonzero(~np.any(temp == 2, axis=1))
            if free_rows.size == 1:
                subsets = _merge_disjoint_subsets(subsets, free_rows[0], nrow)

    if ambiguous and use_springs:
        # Get rid of connections that are no longer ambiguous
        for ambi in ambiguous[::-1]:
            rows, _ = np.where(np.isin(subsets, ambi[:2]))
            if len(rows) < 2 or rows[0] == rows[1]:
                ambiguous.remove(ambi)
        # Fix ambiguous connections starting from those nodes of highest degree.
        G2 = G.edge_subgraph(ambi[:2] for ambi in ambiguous)
        all_ids = np.arange(subsets.shape[0])[:, np.newaxis] * np.ones(subsets.shape[1])
        counts = dict(G2.degree)
        degs = [k for k, _ in sorted(G2.degree(weight='weight'), key=lambda x: x[1], reverse=True)]
        spring = SpringEmbedder(G)
        for ind in degs:
            if counts[ind] > 0:
                temp = np.c_[subsets.ravel(), all_ids.ravel()].astype(int)
                temp = temp[np.all(temp != -1, axis=1)]
                inds_ = list(temp[:, 0])
                # Infer an ambiguous node's ID from its neighbors in transformed space.
                neighbors = spring.find_neighbors(ind)
                mask = [inds_.index(n) for n in neighbors.flatten() if n in inds_]
                ids = temp[mask]
                curr_id = ids[0, 1]
                id_, count = np.unique(ids[1:, 1], return_counts=True)
                neigh_id = id_[np.argsort(count)[::-1]][0]
                if curr_id == neigh_id:  # No ID mismatch
                    counts[ind] = 0
                    for neigh in G2.neighbors(ind):
                        counts[neigh] -= 1
                    continue
                row, col = np.argwhere(subsets == ind).squeeze()
                new_ind = subsets[neigh_id, col]
                if new_ind == -1:
                    subsets[[neigh_id, row], col] = ind, new_ind
                    counts[ind] = 0
                    for neigh in G2.neighbors(ind):
                        counts[neigh] -= 1
                else:  # Ensure swapping does not make things worse
                    if np.all(ids[2:, 1] == ids[1, 1]):
                        subsets[[neigh_id, row], col] = ind, new_ind
                        counts[ind] = 0
                        for neigh in G2.neighbors(ind):
                            counts[neigh] -= 1
                    else:
                        # Swap bodyparts only if doing so reduces the tension of
                        # the springs comprising the ambiguous subsets.
                        curr_inds = subsets[curr_id]
                        new_inds = curr_inds.copy()
                        new_inds[col] = new_ind
                        curr_coords = [spring.layout[i] for i in curr_inds if i != -1]
                        new_coords = [spring.layout[i] for i in new_inds if i != -1]
                        curr_e = sum(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                     for (x1, y1), (x2, y2) in itertools.combinations(curr_coords, 2))
                        new_e = sum(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    for (x1, y1), (x2, y2) in itertools.combinations(new_coords, 2))
                        if new_e < curr_e:
                            subsets[[neigh_id, row], col] = ind, new_ind
                        counts[ind] = 0
                        for neigh in G2.neighbors(ind):
                            counts[neigh] -= 1
        subsets = subsets[np.any(subsets != -1, axis=1)]

    subsets = np.vstack((subsets0, subsets))
    return subsets[:max_individuals], candidates


def _merge_disjoint_subsets(subsets, row1, row2):
    subsets[row1] += subsets[row2] + 1
    return np.delete(subsets, row2, axis=0)


class SpringEmbedder:
    def __init__(self, graph, n_iter=50):
        self.graph = graph.copy()
        self.layout = nx.spring_layout(graph, iterations=n_iter, seed=42)
        self.nodes = np.asarray(list(self.layout))
        self.coords = np.stack(list(self.layout.values()))
        self.tree = cKDTree(self.coords)

    def find_neighbors(self, nodes, n_neighbors=6):
        xy = self.coords[np.isin(self.nodes, nodes)]
        neighbors = self.tree.query(xy, k=n_neighbors)[1]
        return self.nodes[neighbors]

    def view(self, node_colors=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.axis(False)
        nx.draw_networkx(
            self.graph, pos=self.layout, ax=ax, node_color=node_colors,
        )


def link_joints_to_individuals(
    all_detections,
    all_connections,
    max_individuals,
    use_springs=False,
    link_unconnected=True,
    sort_by="affinity",
):
    if sort_by not in ("affinity", "degree"):
        raise ValueError("`sort_by` must either be 'affinity' or 'degree'.")

    candidates = []
    missing_detections = []
    for i, sublist in enumerate(all_detections):
        if not sublist:
            missing_detections.append(i)
        else:
            for item in sublist:
                candidates.append(item + [i])
    candidates = np.asarray(candidates)
    all_inds = candidates[:, -1].astype(int)

    G = nx.Graph()
    G.add_weighted_edges_from(
        [(a, b, c) for arrays in all_connections for a, b, c, *_ in arrays],
    )
    mask_unconnected = ~np.isin(candidates[:, -2], list(G))

    n_bodyparts = len(all_detections)
    subsets = np.empty((0, n_bodyparts))
    # Fill the subsets with unambiguous (partial) skeletons
    for chain in sorted(nx.connected_components(G), key=len, reverse=True):
        if len(chain) <= n_bodyparts - len(missing_detections):
            row = -1 * np.ones(n_bodyparts)
            nodes = list(chain)
            row[all_inds[nodes]] = nodes
            subsets = np.vstack((subsets, row))
            G.remove_nodes_from(nodes)

    if sort_by == 'affinity':
        # Sort connections in descending order of part affinity
        connections = sorted(
            G.edges.data('weight'),
            key=lambda x: x[2],
            reverse=True
        )
    else:
        # Alternatively, sort edges in decreasing order
        # of their vertices' weighted degree. This is to encourage
        # dense, strong connections to come first.
        nodes_weight = defaultdict(int)
        for ind1, ind2, score in G.edges.data('weight'):
            nodes_weight[ind1] += score
            nodes_weight[ind2] += score
        connections = sorted(
            G.edges.data('weight'),
            key=lambda x: nodes_weight[x[0]] + nodes_weight[x[1]],
            reverse=True
        )

    ambiguous = []
    for connection in connections:
        nodes = list(connection[:2])
        inds = all_inds[nodes]
        mask = np.any(subsets[:, inds] == nodes, axis=1)
        subset_inds = np.flatnonzero(mask)
        found = subset_inds.size
        if found == 1:
            subset = subsets[subset_inds[0]]
            if np.all(subset[inds] == nodes):  # Nodes correctly found in the same subset
                continue
            if np.all(subset[inds] != -1):
                ambiguous.append(connection)
            elif subset[inds[0]] == -1:
                subset[inds[0]] = nodes[0]
            else:
                subset[inds[1]] = nodes[1]
        elif found == 2:
            membership = np.sum(subsets[subset_inds] >= 0, axis=0)
            if not np.any(membership == 2):
                subsets = _merge_disjoint_subsets(subsets, *subset_inds)
            else:
                ambiguous.append(connection)
        else:
            row = -1 * np.ones(n_bodyparts)
            row[inds] = nodes
            subsets = np.vstack((subsets, row))

    if ambiguous and use_springs:
        # Get rid of connections that are no longer ambiguous
        for ambi in ambiguous[::-1]:
            rows, _ = np.where(np.isin(subsets, ambi[:2]))
            if len(rows) < 2 or rows[0] == rows[1]:
                ambiguous.remove(ambi)
        # Fix ambiguous connections starting from those nodes of highest degree.
        G2 = G.edge_subgraph(ambi[:2] for ambi in ambiguous)
        all_ids = np.arange(subsets.shape[0])[:, np.newaxis] * np.ones(subsets.shape[1])
        counts = dict(G2.degree)
        degs = [k for k, _ in sorted(G2.degree(weight='weight'), key=lambda x: x[1], reverse=True)]
        spring = SpringEmbedder(G)
        for ind in degs:
            if counts[ind] > 0:
                temp = np.c_[subsets.ravel(), all_ids.ravel()].astype(int)
                temp = temp[np.all(temp != -1, axis=1)]
                inds_ = list(temp[:, 0])
                # Infer an ambiguous node's ID from its neighbors in transformed space.
                neighbors = spring.find_neighbors(ind)
                mask = [inds_.index(n) for n in neighbors.flatten() if n in inds_]
                ids = temp[mask]
                curr_id = ids[0, 1]
                id_, count = np.unique(ids[1:, 1], return_counts=True)
                neigh_id = id_[np.argsort(count)[::-1]][0]
                if curr_id == neigh_id:  # No ID mismatch
                    counts[ind] = 0
                    for neigh in G2.neighbors(ind):
                        counts[neigh] -= 1
                    continue
                row, col = np.argwhere(subsets == ind).squeeze()
                new_ind = subsets[neigh_id, col]
                if new_ind == -1:
                    subsets[[neigh_id, row], col] = ind, new_ind
                    counts[ind] = 0
                    for neigh in G2.neighbors(ind):
                        counts[neigh] -= 1
                else:  # Ensure swapping does not make things worse
                    if np.all(ids[2:, 1] == ids[1, 1]):
                        subsets[[neigh_id, row], col] = ind, new_ind
                        counts[ind] = 0
                        for neigh in G2.neighbors(ind):
                            counts[neigh] -= 1
                    else:
                        # Swap bodyparts only if doing so reduces the tension of
                        # the springs comprising the ambiguous subsets.
                        curr_inds = subsets[curr_id]
                        new_inds = curr_inds.copy()
                        new_inds[col] = new_ind
                        curr_coords = [spring.layout[i] for i in curr_inds if i != -1]
                        new_coords = [spring.layout[i] for i in new_inds if i != -1]
                        curr_e = sum(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                     for (x1, y1), (x2, y2) in itertools.combinations(curr_coords, 2))
                        new_e = sum(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    for (x1, y1), (x2, y2) in itertools.combinations(new_coords, 2))
                        if new_e < curr_e:
                            subsets[[neigh_id, row], col] = ind, new_ind
                        counts[ind] = 0
                        for neigh in G2.neighbors(ind):
                            counts[neigh] -= 1
        subsets = subsets[np.any(subsets != -1, axis=1)]

    # Merge extra subsets
    nrows = len(subsets)
    if nrows > max_individuals:
        subsets = subsets[np.argsort(np.sum(subsets == -1, axis=1))]
        ii = np.argsort(np.sum(subsets[max_individuals:] != -1, axis=1))
        subsets[max_individuals:] = subsets[ii + max_individuals]
        for nrow in range(nrows - 1, max_individuals - 1, -1):
            mask = (subsets[:max_individuals] >= 0).astype(int)
            row = subsets[nrow]
            mask2 = (row >= 0).astype(int)
            temp = mask + mask2
            free_rows = np.flatnonzero(~np.any(temp == 2, axis=1))
            if not free_rows.size:
                # Special treatment
                empty = subsets[:max_individuals] == -1
                has_value = row != -1
                mask = empty & has_value
                n_chains = mask.sum(axis=1)
                while np.any(n_chains > 0):
                    ind = n_chains.argmax()
                    mask_ = np.flatnonzero(mask[ind])
                    subsets[np.ix_([ind, nrow], mask_)] = subsets[np.ix_([nrow, ind], mask_)]
                    # Update masks
                    empty = subsets[:max_individuals] == -1
                    has_value[mask_] = False
                    mask = empty & has_value
                    n_chains = mask.sum(axis=1)
                continue
            if free_rows.size == 1:
                ind = free_rows[0]
                xy = candidates[row[row != -1].astype(int), :2].mean(axis=0)
                dists = []
                for free_row in range(max_individuals):
                    sub_ = subsets[free_row]
                    d = cdist(
                        np.asarray(xy).reshape((1, -1)),
                        candidates[sub_[sub_ != -1].astype(int), :2],
                    )
                    dists.append(d.mean())
                if ind != np.argmin(dists):
                    continue
            else:
                xy = candidates[row[row != -1].astype(int), :2].mean(axis=0)
                dists = []
                for free_row in free_rows:
                    sub_ = subsets[free_row]
                    d = cdist(
                        np.asarray(xy).reshape((1, -1)),
                        candidates[sub_[sub_ != -1].astype(int), :2],
                    )
                    dists.append(d.min())
                ind = free_rows[np.argmin(dists)]
            subsets = _merge_disjoint_subsets(subsets, ind, nrow)
        subsets = subsets[np.any(subsets != -1, axis=1)]

    # Link unconnected bodyparts
    mask_valid = ~np.isnan(candidates[:, :2]).all(axis=1)
    unconnected = candidates[np.logical_and(mask_unconnected, mask_valid)]
    n_unconnected = unconnected.shape[0]
    if n_unconnected and link_unconnected:
        unconnected = unconnected[np.argsort(unconnected[:, 2])[::-1]]
        inds = list(set(unconnected[:, -1].astype(int)))
        for ind in inds:
            bpts = unconnected[unconnected[:, -1] == ind]
            for bpt in bpts:
                xy = bpt[:2]
                n_bpt = int(bpt[-2])
                free = np.flatnonzero(subsets[:, ind] == -1)
                n_free = free.size
                if not n_free:
                    row = -1 * np.ones(n_bodyparts)
                    row[ind] = n_bpt
                    subsets = np.vstack((subsets, row))
                elif n_free == 1:
                    subsets[free[0], ind] = n_bpt
                else:
                    dists = []
                    for j in free:
                        sub_ = subsets[j]
                        d = cdist(
                            np.asarray(xy).reshape((1, -1)),
                            candidates[sub_[sub_ != -1].astype(int), :2],
                        )
                        dists.append(d.min())
                    subsets[free[np.argmin(dists)], ind] = n_bpt

    return subsets[:max_individuals], candidates


def assemble_individuals(
    inference_cfg,
    data_dict,
    n_multibodyparts,
    paf_inds,
    paf_graph,
    use_springs=False,
    dist_funcs=None
):
    all_detections = _nest_detections_in_arrays(data_dict)
    single_detections = all_detections[n_multibodyparts:]
    if len(single_detections):
        single = np.full((len(single_detections), 3), np.nan)
        for n, dets in enumerate(single_detections):
            if len(dets) > 1:
                single[n] = dets[np.argmax(dets[:, 2]), :3]
            elif len(dets) == 1:
                single[n] = dets[0, :3]
    else:
        single = None

    multi_detections = all_detections[:n_multibodyparts]
    if np.all([~np.any(dets) for dets in multi_detections]):
        return None, single

    all_connections = _extract_strong_connections(
        multi_detections,
        paf_graph,
        paf_inds,
        data_dict["costs"],
        inference_cfg["pcutoff"],
        dist_funcs,
    )
    subsets, candidates = _link_detections(
        multi_detections,
        all_connections,
        inference_cfg["topktoretain"],
        use_springs=use_springs,
    )
    ncols = 4 if inference_cfg["withid"] else 3
    animals = np.full((len(subsets), n_multibodyparts, ncols), np.nan)
    for animal, subset in zip(animals, subsets):
        inds = subset.astype(int)
        mask = inds != -1
        animal[mask] = candidates[inds[mask], :-2]

    return animals, single


def _conv_square_to_condensed_indices(ind_row, ind_col, n):
    if ind_row == ind_col:
        raise ValueError('There are no diagonal elements in condensed matrices.')

    if ind_row < ind_col:
        ind_row, ind_col = ind_col, ind_row
    return n * ind_col - ind_col * (ind_col + 1) // 2 + ind_row - 1 - ind_col


Position = Tuple[float, float]


@dataclass(frozen=True)
class Joint:
    pos: Position
    confidence: float = 1.
    label: int = None
    idx: int = None
    group: int = -1


class Link:
    def __init__(self, j1, j2, affinity=1):
        self.j1 = j1
        self.j2 = j2
        self.affinity = affinity
        self._length = sqrt((j1.pos[0] - j2.pos[0]) ** 2
                            + (j1.pos[1] - j2.pos[1]) ** 2)

    def __repr__(self):
        return f'Link {self.idx}, affinity={self.affinity:.2f}, length={self.length:.2f}'

    @property
    def confidence(self):
        return self.j1.confidence * self.j2.confidence

    @property
    def idx(self):
        return self.j1.idx, self.j2.idx

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    def to_vector(self):
        return [*self.j1.pos, *self.j2.pos]


class Assembly:
    def __init__(self, size):
        self.data = np.full((size, 4), np.nan)
        self._affinity = 0
        self._links = []
        self._visible = set()
        self._idx = set()
        self._dict = dict()

    def __len__(self):
        return len(self._visible)

    def __contains__(self, assembly):
        return bool(self._visible.intersection(assembly._visible))

    def __add__(self, other):
        if other in self:
            raise ValueError('Assemblies contain shared joints.')

        assembly = Assembly(self.data.shape[0])
        for link in self._links + other._links:
            assembly.add_link(link)
        return assembly

    @property
    def xy(self):
        return self.data[:, :2]

    @property
    def affinity(self):
        return self._affinity / self.n_links

    @property
    def n_links(self):
        return len(self._links)

    def add_joint(self, joint):
        if joint.label in self._visible:
            return False
        self.data[joint.label] = *joint.pos, joint.confidence, joint.group
        self._visible.add(joint.label)
        self._idx.add(joint.idx)
        return True

    def remove_joint(self, joint):
        if joint.label not in self._visible:
            return False
        self.data[joint.label] = np.nan
        self._visible.remove(joint.label)
        self._idx.remove(joint.idx)
        return True

    def add_link(self, link, store_dict=False):
        if store_dict:
            # Selective copy; deepcopy is >5x slower
            self._dict = {
                'data': self.data.copy(),
                '_affinity': self._affinity,
                '_links': self._links.copy(),
                '_visible': self._visible.copy(),
                '_idx': self._idx.copy(),
            }
        i1, i2 = link.idx
        if i1 in self._idx and i2 in self._idx:
            self._affinity += link.affinity
            self._links.append(link)
            return False
        if link.j1.label in self._visible and link.j2.label in self._visible:
            return False
        self.add_joint(link.j1)
        self.add_joint(link.j2)
        self._affinity += link.affinity
        self._links.append(link)
        return True

    def calc_pairwise_distances(self):
        return pdist(self.data, metric='sqeuclidean')


class Assembler:
    def __init__(
        self,
        data,
        *,
        max_n_individuals,
        n_multibodyparts,
        graph=None,
        paf_inds=None,
        greedy=False,
        safe_edge=True,
        pcutoff=0.1,
        min_affinity=0.1,
        nan_policy='little',
        force_fusion=False,
        add_discarded=False,
        window_size=0,
        method='m1',
    ):
        self.data = data
        self.metadata = self.parse_metadata(self.data)
        self.max_n_individuals = max_n_individuals
        self.n_multibodyparts = n_multibodyparts
        self.n_uniquebodyparts = self.n_keypoints - n_multibodyparts
        self.greedy = greedy
        self.safe_edge = safe_edge
        self.pcutoff = pcutoff
        self.min_affinity = min_affinity
        self.nan_policy = nan_policy
        self.force_fusion = force_fusion
        self.add_discarded = add_discarded
        self.window_size = window_size
        self.method = method
        self.graph = graph or self.metadata['paf_graph']
        self.paf_inds = paf_inds or self.metadata['paf']
        self._gamma = 0.01
        self._trees = dict()
        self._kde = None
        self.assemblies = dict()
        self.unique = dict()

    def __getitem__(self, item):
        return self.data[self.metadata['imnames'][item]]

    @property
    def n_keypoints(self):
        return self.metadata['num_joints']

    def calibrate(self, train_data_file):
        df = pd.read_hdf(train_data_file)
        try:
            df.drop('single', level='individuals', axis=1, inplace=True)
        except KeyError:
            pass
        n_bpts = len(df.columns.get_level_values('bodyparts').unique())
        xy = df.to_numpy().reshape((-1, n_bpts, 2))
        xy = xy[~np.isnan(xy).any(axis=(1, 2))]
        # TODO Normalize dists by longest length?
        # TODO Bayesian multiple imputation of missing data
        dists = np.vstack([pdist(data, 'sqeuclidean') for data in xy])
        kde = gaussian_kde(dists.T)
        kde.mean = np.mean(dists, axis=0)
        self._kde = kde

    def calc_assembly_mahalanobis_dist(
        self,
        assembly,
        return_proba=False,
        nan_policy='little',
    ):
        if self._kde is None:
            raise ValueError('Assembler should be calibrated first with training data.')

        dists = assembly.calc_pairwise_distances() - self._kde.mean
        mask = np.isnan(dists)
        if mask.any() and nan_policy == 'little':
            inds = np.flatnonzero(~mask)
            dists = dists[inds]
            inv_cov = self._kde.inv_cov[np.ix_(inds, inds)]
            # Correct distance to account for missing observations
            factor = self._kde.d / len(inds)
        else:
            # Alternatively, reduce contribution of missing values to the Mahalanobis
            # distance to zero by substituting the corresponding means.
            dists[mask] = 0
            mask.fill(False)
            inv_cov = self._kde.inv_cov
            factor = 1
        dot = dists @ inv_cov
        mahal = factor * sqrt(np.sum((dot * dists), axis=-1))
        if return_proba:
            proba = 1 - chi2.cdf(mahal, np.sum(~mask))
            return mahal, proba
        return mahal

    def calc_link_probability(
        self,
        link,
    ):
        if self._kde is None:
            raise ValueError('Assembler should be calibrated first with training data.')

        i = link.j1.label
        j = link.j2.label
        ind = _conv_square_to_condensed_indices(i, j, self.n_multibodyparts)
        mu = self._kde.mean[ind]
        sigma = self._kde.covariance[i, j]
        z = (link.length ** 2 - mu) / sigma
        return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    @staticmethod
    def _flatten_detections(data_dict):
        ind = 0
        coordinates = data_dict['coordinates'][0]
        confidence = data_dict['confidence']
        ids = data_dict.get('identity', None)
        if ids is None:
            ids = [np.ones(len(arr), dtype=int) * -1 for arr in confidence]
        else:
            ids = [arr.argmax(axis=1) for arr in ids]
        for i, (coords, conf, id_) in enumerate(zip(coordinates, confidence, ids)):
            if not np.any(coords):
                continue
            for xy, p, g in zip(coords, conf, id_):
                joint = Joint(tuple(xy), p.item(), i, ind, g)
                ind += 1
                yield joint

    def extract_best_links(
        self,
        joints_dict,
        costs,
        trees=None,
    ):
        links = []
        for (s, t), ind in zip(self.graph, self.paf_inds):
            dets_s = joints_dict.get(s, None)
            dets_t = joints_dict.get(t, None)
            if dets_s is None or dets_t is None:
                continue
            lengths = costs[ind]['distance']
            if np.isinf(lengths).all():
                continue
            aff = costs[ind][self.method].copy()
            aff[np.isnan(aff)] = 0

            if trees:
                vecs = np.vstack(
                    [[*det_s.pos, *det_t.pos] for det_s in dets_s for det_t in dets_t]
                )
                dists = []
                for n, tree in enumerate(trees, start=1):
                    d, _ = tree.query(vecs)
                    dists.append(np.exp(-self._gamma * n * d))
                w = np.mean(dists, axis=0)
                aff *= w.reshape(aff.shape)

            if self.greedy:
                conf = np.asarray([
                    [det_s.confidence * det_t.confidence for det_t in dets_t] for det_s in dets_s]
                )
                rows, cols = np.where(
                    (conf >= self.pcutoff * self.pcutoff)
                    & (aff >= self.min_affinity)
                )
                candidates = sorted(
                    zip(rows, cols, aff[rows, cols], lengths[rows, cols]),
                    key=lambda x: x[2], reverse=True,
                )
                i_seen = set()
                j_seen = set()
                for i, j, w, l in candidates:
                    if i not in i_seen and j not in j_seen:
                        i_seen.add(i)
                        j_seen.add(j)
                        links.append(Link(dets_s[i], dets_t[j], w))
                        if len(i_seen) == self.max_n_individuals:
                            break
            else:  # Optimal keypoint pairing
                inds_s = sorted(range(len(dets_s)),
                                key=lambda x: dets_s[x].confidence,
                                reverse=True)[:self.max_n_individuals]
                inds_t = sorted(range(len(dets_t)),
                                key=lambda x: dets_t[x].confidence,
                                reverse=True)[:self.max_n_individuals]
                keep_s = [ind for ind in inds_s if dets_s[ind].confidence >= self.pcutoff]
                keep_t = [ind for ind in inds_t if dets_t[ind].confidence >= self.pcutoff]
                aff = aff[np.ix_(keep_s, keep_t)]
                rows, cols = linear_sum_assignment(aff, maximize=True)
                for row, col in zip(rows, cols):
                    w = aff[row, col]
                    if w >= self.min_affinity:
                        links.append(Link(dets_s[keep_s[row]], dets_t[keep_t[col]], w))
        return links

    def _fill_assembly(
        self,
        assembly,
        lookup,
        assembled,
        safe_edge,
        nan_policy,
    ):
        stack = []
        visited = set()
        tabu = []
        counter = itertools.count()

        def push_to_stack(i):
            for j, link in lookup[i].items():
                if j in assembly._idx:
                    continue
                if link.idx in visited:
                    continue
                heapq.heappush(stack, (-link.affinity, next(counter), link))
                visited.add(link.idx)

        for idx in assembly._idx:
            push_to_stack(idx)

        while stack and len(assembly) < self.n_multibodyparts:
            _, _, best = heapq.heappop(stack)
            i, j = best.idx
            if i in assembly._idx:
                new_ind = j
            elif j in assembly._idx:
                new_ind = i
            else:
                continue
            if new_ind in assembled:
                continue
            if safe_edge:
                d_old = self.calc_assembly_mahalanobis_dist(assembly, nan_policy=nan_policy)
                success = assembly.add_link(best, store_dict=True)
                if not success:
                    assembly._dict = dict()
                    continue
                d = self.calc_assembly_mahalanobis_dist(assembly, nan_policy=nan_policy)
                if d < d_old:
                    push_to_stack(new_ind)
                    try:
                        _, _, link = heapq.heappop(tabu)
                        heapq.heappush(stack, (-link.affinity, next(counter), link))
                    except IndexError:
                        pass
                else:
                    heapq.heappush(tabu, (d - d_old, next(counter), best))
                    assembly.__dict__.update(assembly._dict)
                assembly._dict = dict()
            else:
                assembly.add_link(best)
                push_to_stack(new_ind)

    def build_assemblies(
        self,
        joints,
        links,
    ):
        safe_edge = self.safe_edge and self._kde is not None

        lookup = defaultdict(dict)
        for link in links:
            i, j = link.idx
            lookup[i][j] = link
            lookup[j][i] = link

        assemblies = []
        assembled = set()

        # Fill the subsets with unambiguous, complete individuals
        G = nx.Graph([link.idx for link in links])
        for chain in nx.connected_components(G):
            if len(chain) == self.n_multibodyparts:
                edges = [tuple(sorted(edge)) for edge in G.edges(chain)]
                assembly = Assembly(self.n_multibodyparts)
                for link in links:
                    i, j = link.idx
                    if (i, j) in edges:
                        success = assembly.add_link(link)
                        if success:
                            lookup[i].pop(j)
                            lookup[j].pop(i)
                assembled.update(assembly._idx)
                assemblies.append(assembly)

        if len(assemblies) == self.max_n_individuals:
            return assemblies

        for link in sorted(links, key=lambda x: x.affinity, reverse=True):
            if any(i in assembled for i in link.idx):
                continue
            assembly = Assembly(self.n_multibodyparts)
            assembly.add_link(link)
            self._fill_assembly(
                assembly, lookup, assembled, safe_edge, self.nan_policy,
            )
            for link in assembly._links:
                i, j = link.idx
                lookup[i].pop(j)
                lookup[j].pop(i)
            assembled.update(assembly._idx)
            assemblies.append(assembly)

        # Fuse superfluous assemblies
        n_extra = len(assemblies) - self.max_n_individuals
        if n_extra > 0:
            if safe_edge:
                ds_old = [self.calc_assembly_mahalanobis_dist(assembly)
                          for assembly in assemblies]
                while len(assemblies) > self.max_n_individuals:
                    ds = []
                    for i, j in itertools.combinations(range(len(assemblies)), 2):
                        if assemblies[j] not in assemblies[i]:
                            temp = assemblies[i] + assemblies[j]
                            d = self.calc_assembly_mahalanobis_dist(temp)
                            delta = d - max(ds_old[i], ds_old[j])
                            ds.append((i, j, delta, d, temp))
                    if not ds:
                        break
                    min_ = sorted(ds, key=lambda x: x[2])
                    i, j, delta, d, new = min_[0]
                    if delta < 0 or len(min_) == 1:
                        assemblies[i] = new
                        assemblies.pop(j)
                        ds_old[i] = d
                        ds_old.pop(j)
                    else:
                        break
            elif self.force_fusion:
                assemblies = sorted(assemblies, key=len)
                for nrow in range(n_extra):
                    assembly = assemblies[nrow]
                    candidates = [a for a in assemblies[nrow:] if assembly not in a]
                    if not candidates:
                        continue
                    if len(candidates) == 1:
                        candidate = candidates[0]
                    else:
                        dists = []
                        for cand in candidates:
                            d = cdist(assembly.xy, cand.xy)
                            dists.append(np.nanmin(d))
                        candidate = candidates[np.argmin(dists)]
                    ind = assemblies.index(candidate)
                    assemblies[ind] += assembly
            else:
                store = dict()
                for assembly in assemblies:
                    if len(assembly) != self.n_multibodyparts:
                        for i in assembly._idx:
                            store[i] = assembly
                used = [link for assembly in assemblies for link in assembly._links]
                unconnected = [link for link in links if link not in used]
                for link in unconnected:
                    i, j = link.idx
                    try:
                        if store[j] not in store[i]:
                            temp = store[i] + store[j]
                            store[i].__dict__.update(temp.__dict__)
                            assemblies.remove(store[j])
                            for idx in store[j]._idx:
                                store[idx] = store[i]
                    except KeyError:
                        pass

        # Second pass without edge safety
        for assembly in assemblies:
            if len(assembly) != self.n_multibodyparts:
                self._fill_assembly(assembly, lookup, assembled, False, '')
                assembled.update(assembly._idx)

        if self.add_discarded:
            # Last pass to fill assemblies with unconnected body parts
            discarded = set(joint for joint in joints
                            if joint.idx not in assembled
                            and np.isfinite(joint.confidence))
            if len(assemblies) > self.max_n_individuals:
                assemblies = sorted(assemblies, key=len, reverse=True)
                for assembly in assemblies[self.max_n_individuals:]:
                    for link in assembly._links:
                        discarded.update((link.j1, link.j2))
                assemblies = assemblies[:self.max_n_individuals]
            if discarded:
                for joint in sorted(discarded, key=lambda x: x.confidence, reverse=True):
                    if safe_edge:
                        for assembly in assemblies:
                            if joint.label in assembly._visible:
                                continue
                            d_old = self.calc_assembly_mahalanobis_dist(assembly)
                            assembly.add_joint(joint)
                            d = self.calc_assembly_mahalanobis_dist(assembly)
                            if d < d_old:
                                break
                            assembly.remove_joint(joint)
                    else:
                        dists = []
                        for i, assembly in enumerate(assemblies):
                            if joint.label in assembly._visible:
                                continue
                            d = cdist(assembly.xy, np.atleast_2d(joint.pos))
                            dists.append((i, np.nanmin(d)))
                        if not dists:
                            continue
                        min_ = sorted(dists, key=lambda x: x[1])
                        ind, _ = min_[0]
                        assemblies[ind].add_joint(joint)

        return assemblies

    def _assemble(
        self,
        data_dict,
        ind_frame,
        return_links=False,
    ):
        joints = list(self._flatten_detections(data_dict))
        if not joints:
            return None, None

        bag = defaultdict(list)
        for joint in joints:
            bag[joint.label].append(joint)

        if self.n_uniquebodyparts:
            unique = np.full((self.n_uniquebodyparts, 3), np.nan)
            for n, ind in enumerate(range(self.n_multibodyparts, self.n_keypoints)):
                dets = bag[ind]
                if not dets:
                    continue
                if len(dets) > 1:
                    det = max(dets, key=lambda x: x.confidence)
                else:
                    det = dets[0]
                unique[n] = *det.pos, det.confidence
            if np.isnan(unique).all():
                unique = None
        else:
            unique = None

        if not any(i in bag for i in range(self.n_multibodyparts)):
            return None, unique

        trees = []
        for j in range(1, self.window_size + 1):
            tree = self._trees.get(ind_frame - j, None)
            if tree is not None:
                trees.append(tree)

        links = self.extract_best_links(
            bag,
            data_dict["costs"],
            trees,
        )
        if self._kde:
            for link in links[::-1]:
                p = max(self.calc_link_probability(link), 0.001)
                link.affinity *= p
                if link.affinity < self.min_affinity:
                    links.remove(link)

        if self.window_size >= 1:
            # Store selected edges for subsequent frames
            vecs = np.vstack([link.to_vector() for link in links])
            self._trees[ind_frame] = cKDTree(vecs)

        assemblies = self.build_assemblies(joints, links)
        if return_links:
            return assemblies, unique, links
        return assemblies, unique

    def assemble(
        self,
        chunk_size=1,
        n_processes=None,
    ):
        if chunk_size == 0:
            for i, data_dict in enumerate(tqdm(self)):
                assemblies, unique = self._assemble(data_dict, i)
                if assemblies:
                    self.assemblies[i] = assemblies
                if unique:
                    self.unique[i] = unique
        else:
            global wrapped  # Hack to make the function pickable

            def wrapped(i):
                return i, self._assemble(self[i], i)

            n_frames = len(self.metadata['imnames'])
            with Pool(n_processes) as p:
                with tqdm(total=n_frames) as pbar:
                    for i, (assemblies, unique) in p.imap_unordered(
                            wrapped,
                            range(n_frames),
                            chunksize=chunk_size
                    ):
                        if assemblies:
                            self.assemblies[i] = assemblies
                        if unique:
                            self.unique[i] = unique
                        pbar.update()

    @staticmethod
    def parse_metadata(data):
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

    def to_h5(self, output_name):
        data = np.full(
            (len(self.metadata['imnames']),
             self.max_n_individuals,
             self.n_multibodyparts, 4),
            fill_value=np.nan,
        )
        for ind, assemblies in self.assemblies.items():
            for n, assembly in enumerate(assemblies):
                data[ind, n] = assembly.data
        index = pd.MultiIndex.from_product(
            [
                ['scorer'],
                map(str, range(self.max_n_individuals)),
                map(str, range(self.n_multibodyparts)),
                ['x', 'y', 'likelihood']
            ],
            names=['scorer', 'individuals', 'bodyparts', 'coords']
        )
        temp = data[..., :3].reshape((data.shape[0], -1))
        df = pd.DataFrame(temp, columns=index)
        df.to_hdf(output_name, key='ass')


def calc_object_keypoint_similarity(xy_pred, xy_true, sigma):
    visible = ~np.isnan(xy_pred * xy_true).all(axis=1)
    if visible.sum() < 2:  # 2 points needed
        return np.nan
    pred = xy_pred[visible]
    true = xy_true[visible]
    dist_squared = np.sum((pred - true) ** 2, axis=1)
    scale_squared = np.product(np.ptp(true, axis=0) + np.spacing(1))
    k_squared = (2 * sigma) ** 2
    oks = np.exp(-dist_squared / (2 * scale_squared * k_squared))
    return np.mean(oks)


def match_assemblies(ass_pred, ass_true, sigma):
    inds_true = list(range(len(ass_true)))
    inds_pred = np.argsort([ins.affinity for ins in ass_pred])[::-1]
    matched = []
    for ind_pred in inds_pred:
        xy_pred = ass_pred[ind_pred].xy
        oks = []
        for ind_true in inds_true:
            xy_true = ass_true[ind_true].xy
            oks.append(calc_object_keypoint_similarity(xy_pred, xy_true, sigma))
        ind_best = np.argmax(oks)
        ind_true_best = inds_true.pop(ind_best)
        matched.append((ass_pred[ind_pred], ass_true[ind_true_best], oks[ind_best]))
        if not inds_true:
            break
    unmatched = [ass_true[ind] for ind in inds_true]
    return matched, unmatched


def parse_ground_truth_data_file(h5_file):
    df = pd.read_hdf(h5_file)
    try:
        df.drop("single", axis=1, level="individuals", inplace=True)
    except KeyError:
        pass
    n_individuals = len(df.columns.get_level_values("individuals").unique())
    n_bodyparts = len(df.columns.get_level_values("bodyparts").unique())
    data = df.to_numpy().reshape((df.shape[0], n_individuals, n_bodyparts, 3))
    return _parse_ground_truth_data(data)


def _parse_ground_truth_data(data):
    gt = dict()
    for i, arr in enumerate(data):
        temp = []
        for row in arr:
            if np.isnan(row[:, :2]).all():
                continue
            ass = Assembly(row.shape[0])
            ass.data[:, :3] = row
            temp.append(ass)
        if not temp:
            continue
        gt[i] = temp
    return gt


def evaluate_assembly(
    ass_pred_dict,
    ass_true_dict,
    oks_sigma=0.072,
    oks_thresholds=np.linspace(0.5, 0.95, 10),
):
    # sigma is taken as the median of all COCO keypoint standard deviations
    all_matched = []
    all_unmatched = []
    for ind, ass_pred in tqdm(ass_pred_dict.items()):
        ass_true = ass_true_dict.get(ind)
        if ass_true is None:
            continue
        matched, unmatched = match_assemblies(ass_pred, ass_true, oks_sigma)
        all_matched.extend(matched)
        all_unmatched.extend(unmatched)
    oks = np.asarray([match[2] for match in all_matched])
    ntot = len(all_matched) + len(all_unmatched)
    recall_thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    for th in oks_thresholds:
        tp = np.cumsum(oks >= th)
        fp = np.cumsum(oks < th)
        rc = tp / ntot
        pr = tp / (fp + tp + np.spacing(1))
        recall = rc[-1]
        pr = np.sort(pr)[::-1]
        inds_rc = np.searchsorted(rc, recall_thresholds)
        precision = np.zeros(inds_rc.shape)
        valid = inds_rc < len(pr)
        precision[valid] = pr[inds_rc[valid]]
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "mAP": precisions.mean(),
        "mAR": recalls.mean(),
    }
