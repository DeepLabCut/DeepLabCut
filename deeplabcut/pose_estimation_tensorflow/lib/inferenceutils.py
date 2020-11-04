"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import numpy as np
from math import sqrt

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


##########################################################
#### conversion & greedy bodypart matching code
##########################################################


def convertdetectiondict2listoflist(dataimage, BPTS, withid=False, evaluation=False):
    """ Arranges data into list of list with the following entries:
    [(x, y, score, global index of detection)] (all detections per bodypart).

    Also includes id if available. [x,y,score,global id, id] """

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
    lowerbound=None,
    upperbound=None,
    evaluation=False,
):
    """ Auxiliary function;  Returns list of connections (limbs) of a particular type.

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

    """
    all_connections = []
    missing_connections = []
    costs = dataimage["prediction"]["costs"] if evaluation else dataimage["costs"]
    for edge in range(len(partaffinityfield_graph)):
        a, b = partaffinityfield_graph[edge]
        cand_a = all_detections[iBPTS[a]]  # convert bpt index to the one in all_detections!
        cand_b = all_detections[iBPTS[b]]
        n_a = len(cand_a)
        n_b = len(cand_b)
        if n_a != 0 and n_b != 0:
            scores = costs[PAF[edge]][cfg.method]
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
                                score_with_dist_prior > cfg.pafthreshold
                                and cfg.distnormalizationLOWER <= d < cfg.distnormalization
                                and si * sj > cfg.detectionthresholdsquare
                        ):
                            connection_candidate.append(
                                [
                                    i,
                                    j,
                                    score_with_dist_prior,
                                ]
                            )
                    else:
                        if (
                                score_with_dist_prior > cfg.pafthreshold
                                and lowerbound[edge] <= d < upperbound[edge]
                                and si * sj > cfg.detectionthresholdsquare
                        ):
                            connection_candidate.append(
                                [
                                    i,
                                    j,
                                    score_with_dist_prior,
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
            for i, j, score in connection_candidate:
                if i not in i_seen and j not in j_seen:
                    i_seen.add(i)
                    j_seen.add(j)
                    ii = int(cand_a[i][-1])  # global index!
                    jj = int(cand_b[j][-1])
                    connection.append([ii, jj, score, i, j])
                    if len(connection) == nrows:
                        break
            all_connections.append(connection)
        else:
            missing_connections.append(edge)
            all_connections.append([])
    return all_connections, missing_connections


def link_joints_to_individuals(
    cfg,
    all_detections,
    all_connections,
    missing_connections,
    partaffinityfield_graph,
    iBPTS,
    num_joints,
):
    candidates = np.array([item for sublist in all_detections for item in sublist])

    # Sort connections in descending order of affinity
    connections = []
    for n, (node1, node2) in enumerate(partaffinityfield_graph):
        if n not in missing_connections:
            for connection in all_connections[n]:
                connection.extend([iBPTS[node1], iBPTS[node2]])
                connections.append(connection)
    connections = sorted(connections, key=lambda x: x[2], reverse=True)

    subset = np.empty((0, num_joints + 2))
    for connection in connections:
        ind1, ind2 = connection[:2]
        node1, node2 = connection[-2:]
        mask = np.logical_or(
            subset[:, node1] == ind1, subset[:, node2] == ind2
        )
        subset_inds = np.flatnonzero(mask)[:2]
        found = subset_inds.size
        if found == 1:
            sub_ = subset[subset_inds[0]]
            if sub_[node1] != ind1:
                sub_[node1] = ind1
            elif sub_[node2] != ind2:
                sub_[node2] = ind2
            sub_[-1] += 1
            sub_[-2] += connection[2]
        elif found == 2:
            membership = np.sum(subset[subset_inds, :-2] >= 0, axis=0)
            if not np.any(membership == 2):  # Merge disjoint subsets
                s1, s2 = subset_inds
                subset[s1, :-2] += subset[s2, :-2] + 1
                subset[s1, -2:] += subset[s2, -2:]
                subset = np.delete(subset, s2, axis=0)
        elif not found:
            row = -1 * np.ones(num_joints + 2)
            row[[node1, node2]] = ind1, ind2
            row[-1] = 1
            row[-2] = connection[2]
            subset = np.vstack((subset, row))

    to_keep = np.logical_and(subset[:, -1] >= cfg.minimalnumberofconnections,
                             subset[:, -2] / subset[:, -1] >= cfg.averagescore)
    subset = subset[to_keep]
    return subset, candidates


def assemble_individuals(
    inference_cfg,
    data,
    numjoints,
    BPTS,
    iBPTS,
    PAF,
    paf_graph,
    paf_links,
    lowerbound=None,
    upperbound=None,
    evaluation=False,
    print_intermediate=False,
):

    # filter detections according to inferencecfg parameters
    all_detections = convertdetectiondict2listoflist(
        data, BPTS, withid=inference_cfg.withid, evaluation=evaluation
    )

    # filter connections according to inferencecfg parameters
    connection_all, missing_connections = extract_strong_connections(inference_cfg, data, all_detections, iBPTS,
                                                                     paf_graph, PAF, lowerbound, upperbound,
                                                                     evaluation=evaluation)
    # assemble putative subsets
    subset, candidate = link_joints_to_individuals(inference_cfg, all_detections, connection_all, missing_connections,
                                                   paf_links, iBPTS, numjoints)
    if print_intermediate:
        print(all_detections)
        print(connection_all)
        print(subset)

    sortedindividuals = np.argsort(-subset[:, -2])  # sort by top score!
    if len(sortedindividuals) > inference_cfg.topktoretain:
        sortedindividuals = sortedindividuals[: inference_cfg.topktoretain]

    animals = []
    for n in sortedindividuals:  # number of individuals
        individual = np.zeros(3 * numjoints) * np.nan
        for i in range(numjoints):  # number of limbs
            ind = int(subset[n][i])
            if -1 == ind:  # bodypart not assigned
                continue
            else:
                individual[3 * i : 3 * i + 3] = candidate[ind, :3]
        animals.append(individual)
    return animals
