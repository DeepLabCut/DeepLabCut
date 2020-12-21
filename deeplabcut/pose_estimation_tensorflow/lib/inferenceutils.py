"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import combinations
from math import sqrt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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
    max_ndets,
    pcutoff=0.3,
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
        inds_a = np.argsort(dets_a[:, 2])[::-1][:max_ndets]
        inds_b = np.argsort(dets_b[:, 2])[::-1][:max_ndets]
        keep_a = inds_a[dets_a[inds_a, 2] >= pcutoff]
        keep_b = inds_b[dets_b[inds_b, 2] >= pcutoff]
        # sl = np.ix_(keep_a, keep_b)
        # w = costs[paf_inds[i]][method][sl]
        w = costs[paf_inds[i]][method].copy()
        affs = w.copy()
        w[np.isnan(w)] = 0  # FIXME Why is it even NaN??
        # dist = dist[sl]
        if dist_funcs:
            w += dist_funcs[i](dist)
        row_inds, col_inds = linear_sum_assignment(w, maximize=True)
        connections = []
        for row, col in zip(row_inds, col_inds):
            d = dist[row, col]
            if dets_a[row, 2] >= pcutoff and dets_b[col, 2] >= pcutoff:
                connections.append(
                    # [int(dets_a[keep_a[row], -1]), int(dets_b[keep_b[col], -1]), affs[row, col], d, row, col]
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
    if nrows > left:
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
                                     for (x1, y1), (x2, y2) in combinations(curr_coords, 2))
                        new_e = sum(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    for (x1, y1), (x2, y2) in combinations(new_coords, 2))
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
                                     for (x1, y1), (x2, y2) in combinations(curr_coords, 2))
                        new_e = sum(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    for (x1, y1), (x2, y2) in combinations(new_coords, 2))
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
        inference_cfg["topktoretain"],
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
