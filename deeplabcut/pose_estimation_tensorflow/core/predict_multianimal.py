#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#


import numpy as np
import tensorflow as tf
from skimage.feature import peak_local_max
from scipy.ndimage import measurements


def extract_cnn_output(outputs_np, cfg):
    """extract locref, scmap and partaffinityfield from network"""
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
    """extract locref + scmap from network
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
    n_bodyparts,
    n_points=10,
    n_decimals=3,
):
    # Clip peak locations to PAFs dimensions
    h, w = pafs.shape[1:3]
    peak_inds_in_batch[:, 1] = np.clip(peak_inds_in_batch[:, 1], 0, h - 1)
    peak_inds_in_batch[:, 2] = np.clip(peak_inds_in_batch[:, 2], 0, w - 1)

    n_samples = pafs.shape[0]
    sample_inds = []
    edge_inds = []
    all_edges = []
    all_peaks = []
    for i in range(n_samples):
        samples_i = peak_inds_in_batch[:, 0] == i
        peak_inds = peak_inds_in_batch[samples_i, 1:]
        if not np.any(peak_inds):
            continue
        peaks = peak_inds[:, :2]
        bpt_inds = peak_inds[:, 2]
        idx = np.arange(peaks.shape[0])
        idx_per_bpt = {j: idx[bpt_inds == j].tolist() for j in range(n_bodyparts)}
        edges = []
        for k, (s, t) in zip(paf_inds, graph):
            inds_s = idx_per_bpt[s]
            inds_t = idx_per_bpt[t]
            if not (inds_s and inds_t):
                continue
            candidate_edges = ((i, j) for i in inds_s for j in inds_t)
            edges.extend(candidate_edges)
            edge_inds.extend([k] * len(inds_s) * len(inds_t))
        if not edges:
            continue
        sample_inds.extend([i] * len(edges))
        all_edges.extend(edges)
        all_peaks.append(peaks[np.asarray(edges)])
    if not all_peaks:
        return [dict() for _ in range(n_samples)]

    sample_inds = np.asarray(sample_inds, dtype=np.int32)
    edge_inds = np.asarray(edge_inds, dtype=np.int32)
    all_edges = np.asarray(all_edges, dtype=np.int32)
    all_peaks = np.concatenate(all_peaks)
    vecs_s = all_peaks[:, 0]
    vecs_t = all_peaks[:, 1]
    vecs = vecs_t - vecs_s
    lengths = np.linalg.norm(vecs, axis=1).astype(np.float32)
    lengths += np.spacing(1, dtype=np.float32)
    xy = np.linspace(vecs_s, vecs_t, n_points, axis=1, dtype=np.int32)
    y = pafs[
        sample_inds.reshape((-1, 1)),
        xy[..., 0],
        xy[..., 1],
        edge_inds.reshape((-1, 1)),
    ]
    integ = np.trapz(y, xy[..., ::-1], axis=1)
    affinities = np.linalg.norm(integ, axis=1).astype(np.float32)
    # unit_vecs = vecs / lengths[:, np.newaxis]
    # affinities = np.squeeze(y @ np.expand_dims(unit_vecs, axis=2)).sum(axis=1)
    affinities /= lengths
    np.round(affinities, decimals=n_decimals, out=affinities)
    np.round(lengths, decimals=n_decimals, out=lengths)

    # Form cost matrices
    all_costs = []
    for i in range(n_samples):
        samples_i_mask = sample_inds == i
        costs = dict()
        for k in paf_inds:
            edges_k_mask = edge_inds == k
            idx = np.flatnonzero(samples_i_mask & edges_k_mask)
            s, t = all_edges[idx].T
            n_sources = np.unique(s).size
            n_targets = np.unique(t).size
            costs[k] = dict()
            costs[k]["m1"] = affinities[idx].reshape((n_sources, n_targets))
            costs[k]["distance"] = lengths[idx].reshape((n_sources, n_targets))
        all_costs.append(costs)

    return all_costs


def compute_peaks_and_costs(
    scmaps,
    locrefs,
    pafs,
    peak_inds_in_batch,
    graph,
    paf_inds,
    stride,
    n_id_channels,
    n_points=10,
    n_decimals=3,
):
    n_samples, _, _, n_channels = np.shape(scmaps)
    n_bodyparts = n_channels - n_id_channels
    pos = calc_peak_locations(locrefs, peak_inds_in_batch, stride, n_decimals)
    if graph:
        costs = compute_edge_costs(
            pafs,
            peak_inds_in_batch,
            graph,
            paf_inds,
            n_bodyparts,
            n_points,
            n_decimals,
        )
    else:
        costs = None
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
        dict_ = {"coordinates": (xy,), "confidence": p}
        if costs is not None:
            dict_["costs"] = costs[i]
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
    extra_dict=None,
):

    if extra_dict:
        features = sess.run(extra_dict["features"], feed_dict={inputs: images_batch})

    scmaps, locrefs, *pafs, peaks = sess.run(outputs, feed_dict={inputs: images_batch})
    if ~np.any(peaks):
        return []

    locrefs = np.reshape(locrefs, (*locrefs.shape[:3], -1, 2))
    locrefs *= pose_cfg["locref_stdev"]
    if pafs:
        pafs = np.reshape(pafs[0], (*pafs[0].shape[:3], -1, 2))
    else:
        pafs = None
    graph = pose_cfg["partaffinityfield_graph"]
    limbs = pose_cfg.get("paf_best", np.arange(len(graph)))
    graph = [graph[l] for l in limbs]
    preds = compute_peaks_and_costs(
        scmaps,
        locrefs,
        pafs,
        peaks,
        graph,
        limbs,
        pose_cfg["stride"],
        pose_cfg.get("num_idchannel", 0),
        n_points,
        n_decimals,
    )
    if peaks_gt is not None and graph:
        costs_gt = compute_edge_costs(
            pafs,
            peaks_gt,
            graph,
            limbs,
            pose_cfg["num_joints"],
            n_points,
            n_decimals,
        )
        for i, costs in enumerate(costs_gt):
            preds[i]["groundtruth_costs"] = costs
    if extra_dict:
        return preds, features
    else:
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


def find_local_peak_indices_maxpool_nms(scmaps, radius, threshold):
    pooled = tf.nn.max_pool2d(scmaps, [radius, radius], strides=1, padding="SAME")
    maxima = scmaps * tf.cast(tf.equal(scmaps, pooled), tf.float32)
    return tf.cast(tf.where(maxima >= threshold), tf.int32)


def find_local_peak_indices_dilation(scmaps, radius, threshold):
    kernel = np.zeros((radius, radius, 1))
    mid = (radius - 1) // 2
    kernel[mid, mid] = -1
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)

    height = tf.shape(scmaps)[1]
    width = tf.shape(scmaps)[2]
    depth = tf.shape(scmaps)[3]
    scmaps_flat = tf.reshape(
        tf.transpose(scmaps, [0, 3, 1, 2]),
        [-1, height, width, 1],
    )
    scmaps_dil = tf.nn.dilation2d(
        scmaps_flat,
        kernel,
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME",
    )
    scmaps_dil = tf.transpose(
        tf.reshape(scmaps_dil, [-1, depth, height, width]),
        [0, 2, 3, 1],
    )
    argmax_and_thresh_img = (scmaps > scmaps_dil) & (scmaps > threshold)
    return tf.cast(tf.where(argmax_and_thresh_img), tf.int32)


def find_local_peak_indices_skimage(scmaps, radius, threshold):
    inds_gt = []
    for i in range(scmaps.shape[0]):
        for j in range(scmaps.shape[3]):
            scmap = scmaps[i, ..., j]
            peaks = find_local_maxima(scmap, radius, threshold)
            samples_i = np.ones(len(peaks), dtype=np.int).reshape((-1, 1)) * i
            bpts_j = np.ones_like(samples_i) * j
            inds_gt.append(np.c_[samples_i, peaks, bpts_j])
    return np.concatenate(inds_gt)


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
