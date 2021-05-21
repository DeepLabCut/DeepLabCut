"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Extract_detections with C++ code & nms adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

import numpy as np
import tensorflow as tf

vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf

from skimage.feature import peak_local_max
from scipy.ndimage import measurements


def extract_cnn_output(outputs_np, cfg):
    """ extract locref, scmap and partaffinityfield from network """
    scmap = outputs_np[0]
    scmap = np.squeeze(scmap)
    if cfg["location_refinement"]:
        locref = np.squeeze(outputs_np[1])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg["locref_stdev"]
    else:
        locref = None
    if cfg["partaffinityfield_predict"] and ("multi-animal" in cfg["dataset_type"]):
        paf = np.squeeze(outputs_np[2])
    else:
        paf = None

    if len(scmap.shape) == 2:  # for single body part!
        scmap = np.expand_dims(scmap, axis=2)
    return scmap, locref, paf


def extract_cnn_outputmulti(outputs_np, cfg):
    """ extract locref + scmap from network
    Dimensions: image batch x imagedim1 x imagedim2 x bodypart"""
    scmap = outputs_np[0]
    if cfg["location_refinement"]:
        locref = outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], shape[2], -1, 2))
        locref *= cfg["locref_stdev"]
    else:
        locref = None
    if cfg["partaffinityfield_predict"] and ("multi-animal" in cfg["dataset_type"]):
        paf = outputs_np[2]
    else:
        paf = None

    if len(scmap.shape) == 2:  # for single body part!
        scmap = np.expand_dims(scmap, axis=2)
    return scmap, locref, paf


def compute_edge_costs(
    pafs,
    peak_inds_in_batch,
    graph,
    paf_inds,
    n_points=10,
    n_decimals=3,
):
    n_samples = pafs.shape[0]
    n_bodyparts = np.max(peak_inds_in_batch[:, 3]) + 1
    sample_inds = []
    edge_inds = []
    all_edges = []
    all_peaks = []
    for i in range(n_samples):
        samples_i = peak_inds_in_batch[:, 0] == i
        peak_inds = peak_inds_in_batch[samples_i, 1:]
        peaks = peak_inds[:, :2]
        bpt_inds = peak_inds[:, 2]
        idx = np.arange(peaks.shape[0])
        idx_per_bpt = {j: idx[bpt_inds == j].tolist() for j in range(n_bodyparts)}
        edges = []
        for k, (s, t) in enumerate(graph):
            inds_s = idx_per_bpt[s]
            inds_t = idx_per_bpt[t]
            candidate_edges = ((i, j) for i in inds_s for j in inds_t)
            edges.extend(candidate_edges)
            edge_inds.extend([k] * len(inds_s) * len(inds_t))
        sample_inds.extend([i] * len(edges))
        all_edges.extend(edges)
        all_peaks.append(peaks[np.asarray(edges)])
    sample_inds = np.asarray(sample_inds, dtype=np.int32)
    edge_inds = np.asarray(edge_inds, dtype=np.int32)
    all_edges = np.asarray(all_edges, dtype=np.int32)
    all_peaks = np.concatenate(all_peaks)
    vecs_s = all_peaks[:, 0]
    vecs_t = all_peaks[:, 1]
    vecs = vecs_t - vecs_s
    lengths = np.linalg.norm(vecs, axis=1) + np.spacing(1)
    xy = np.linspace(vecs_s, vecs_t, n_points, axis=1, dtype=np.int32)
    y = pafs[
        sample_inds.reshape((-1, 1)),
        xy[..., 0],
        xy[..., 1],
        edge_inds.reshape((-1, 1)),
    ]
    integ = np.trapz(y, xy[..., ::-1], axis=1)
    affinities = np.linalg.norm(integ, axis=1) / lengths
    np.round(affinities, decimals=n_decimals, out=affinities)
    np.round(lengths, decimals=n_decimals, out=lengths)

    # Form cost matrices
    all_costs = []
    for i in range(n_samples):
        samples_i_mask = sample_inds == i
        costs = dict()
        for paf_ind, k in zip(paf_inds, range(len(graph))):
            edges_k_mask = edge_inds == k
            idx = np.flatnonzero(samples_i_mask & edges_k_mask)
            s, t = all_edges[idx].T
            n_sources = np.unique(s).size
            n_targets = np.unique(t).size
            costs[paf_ind] = dict()
            costs[paf_ind]["m1"] = affinities[idx].reshape((n_sources, n_targets))
            costs[paf_ind]["distance"] = lengths[idx].reshape((n_sources, n_targets))
        all_costs.append(costs)

    return all_costs


def compute_peaks_and_costs(
    scmaps,
    locrefs,
    pafs,
    graph,
    paf_inds,
    stride,
    n_id_channels,
    nms_radius=5,
    min_confidence=0.01,
    n_points=10,
    n_decimals=3,
    session=None,
):
    n_samples, _, _, n_channels = np.shape(scmaps)
    n_bodyparts = n_channels - n_id_channels
    peak_inds_in_batch = find_local_peak_indices(
        tf.convert_to_tensor(scmaps[..., :n_bodyparts], dtype=tf.float32),
        nms_radius,
        min_confidence,
    )
    if session is None:
        with tf.Session() as session:
            peak_inds_in_batch = session.run(peak_inds_in_batch)
    else:
        peak_inds_in_batch = session.run(peak_inds_in_batch)
    pos = calc_peak_locations(locrefs, peak_inds_in_batch, stride, n_decimals)
    costs = compute_edge_costs(
        pafs, peak_inds_in_batch, graph, paf_inds, n_points, n_decimals,
    )
    s, r, c, b = peak_inds_in_batch.T
    prob = np.round(scmaps[s, r, c, b], n_decimals).reshape((-1, 1))
    if n_id_channels:
        ids = np.round(scmaps[s, r, c, -n_id_channels:], n_decimals)

    peaks_and_costs = []
    for i in range(n_samples):
        xy = []
        p = []
        id_ = []
        samples_i_mask = peak_inds_in_batch[:, 0] == i
        for j in range(n_bodyparts):
            bpts_j_mask = peak_inds_in_batch[:, 3] == j
            idx = np.flatnonzero(samples_i_mask & bpts_j_mask)
            xy.append(pos[idx])
            p.append(prob[idx])
            if n_id_channels:
                id_.append(ids[idx])
        dict_ = {"coordinates": (xy,), "confidence": p, "costs": costs[i]}
        if n_id_channels:
            dict_["identity"] = id_
        peaks_and_costs.append(dict_)

    return peaks_and_costs


def predict_batched_peaks_and_costs(
    pose_cfg,
    images_batch,
    sess,
    inputs,
    outputs,
    peaks_gt=None,
    n_points=10,
    n_decimals=3,
):
    scmaps, locrefs, pafs = sess.run(outputs, feed_dict={inputs: images_batch})
    locrefs = np.reshape(locrefs, (*locrefs.shape[:3], -1, 2))
    locrefs *= pose_cfg["locref_stdev"]
    pafs = np.reshape(pafs, (*pafs.shape[:3], -1, 2))
    graph = pose_cfg["partaffinityfield_graph"]
    limbs = pose_cfg.get("paf_best", np.arange(len(graph)))
    if len(graph) != len(limbs):
        limbs = np.arange(len(graph))
    preds = compute_peaks_and_costs(
        scmaps,
        locrefs,
        pafs,
        graph,
        limbs,
        pose_cfg["stride"],
        pose_cfg.get("num_idchannel", 0),
        int(pose_cfg.get("nmsradius", 5)),
        pose_cfg.get("minconfidence", 0.01),
        n_points,
        n_decimals,
        session=sess,
    )
    if peaks_gt is not None:
        costs_gt = compute_edge_costs(
            pafs, peaks_gt, graph, limbs, n_points, n_decimals,
        )
        for i, costs in enumerate(costs_gt):
            preds[i]["groundtruth_costs"] = costs

    return preds


def find_local_maxima(scmap, radius, threshold):
    peak_idx = peak_local_max(
        scmap, min_distance=radius, threshold_abs=threshold, exclude_border=False
    )
    grid = np.zeros_like(scmap, dtype=bool)
    grid[tuple(peak_idx.T)] = True
    labels = measurements.label(grid)[0]
    xy = measurements.center_of_mass(grid, labels, range(1, np.max(labels) + 1))
    return np.asarray(xy, dtype=np.int).reshape((-1, 2))


def find_local_peak_indices(scmaps, radius, threshold):
    pooled = TF.nn.max_pool2d(scmaps, [radius, radius], strides=1, padding='SAME')
    mask = TF.logical_and(tf.equal(scmaps, pooled), scmaps >= threshold)
    return TF.cast(TF.where(mask), TF.int32)


def calc_peak_locations(
    locrefs,
    peak_inds_in_batch,
    stride,
    n_decimals=3,
):
    s, r, c, b = peak_inds_in_batch.T
    off = locrefs[s, r, c, b]
    loc = stride * peak_inds_in_batch[:, [2, 1]] + stride // 2 + off
    return np.round(loc, decimals=n_decimals)
