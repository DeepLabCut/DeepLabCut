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


def AssociationCosts(
    cfg, coordinates, partaffinitymaps, stride, half_stride, numsteps=50
):
    """ Association costs for detections based on PAFs """
    Distances = {}
    ny, nx, nlimbs = np.shape(partaffinitymaps)
    graph = cfg["partaffinityfield_graph"]
    limbs = cfg.get("paf_best", np.arange(len(graph)))
    if len(graph) != len(limbs):
        limbs = np.arange(len(graph))

    for l, (bp1, bp2) in zip(limbs, graph):
        # get coordinates for bp1 and bp2
        C1 = coordinates[bp1]
        C2 = coordinates[bp2]

        dist = np.zeros((len(C1), len(C2))) * np.nan
        L2distance = np.zeros((len(C1), len(C2))) * np.nan
        # 'm2'
        # distscalarproduct=np.zeros((len(C1),len(C2)))*np.nan
        for c1i, c1 in enumerate(C1):
            for c2i, c2 in enumerate(C2):
                if np.prod(np.isfinite(c1)) * np.prod(np.isfinite(c2)):
                    c1s = (c1 - half_stride) / stride
                    c2s = (c2 - half_stride) / stride

                    c1s[0] = np.clip(int(c1s[0]), 0, nx - 1)
                    c1s[1] = np.clip(int(c1s[1]), 0, ny - 1)
                    c2s[0] = np.clip(int(c2s[0]), 0, nx - 1)
                    c2s[1] = np.clip(int(c2s[1]), 0, ny - 1)

                    Lx = np.array(np.linspace(c1s[0], c2s[0], numsteps), dtype=int)
                    Ly = np.array(np.linspace(c1s[1], c2s[1], numsteps), dtype=int)

                    length = np.sqrt(np.sum((c1s - c2s) ** 2))

                    L2distance[c1i, c2i] = length  # storing length (used in inference)
                    if length > 0:
                        v = (c1s - c2s) * 1.0 / length

                        if c1s[0] != c2s[0]:
                            dx = np.trapz(
                                [
                                    partaffinitymaps[Ly[i], Lx[i], 2 * l]
                                    for i in range(numsteps)
                                ],
                                dx=(c1s[0] - c2s[0]) * 1.0 / (length * numsteps),
                            )
                        else:
                            dx = 0

                        if c1s[1] != c2s[1]:
                            dy = np.trapz(
                                [
                                    partaffinitymaps[Ly[i], Lx[i], 2 * l + 1]
                                    for i in range(numsteps)
                                ],
                                dx=(c1s[1] - c2s[1]) * 1.0 / (length * numsteps),
                            )
                        else:
                            dy = 0

                        # distscalarproduct[c1i,c2i]=dy*v[1]+dx*v[0] #scalar product [v unit vector dx,dy in pixel coordinats from partaffinitymap]
                        dist[c1i, c2i] = np.sqrt(dy ** 2 + dx ** 2)

            Distances[l] = {}
            Distances[l]["m1"] = dist
            # Distances[l]['m2'] = distscalarproduct
            Distances[l]["distance"] = L2distance

    return Distances


def compute_edge_costs(
    pafs,
    peak_inds_in_batch,
    graph,
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
        all_peaks.append(peaks[edges])
    sample_inds = np.asarray(sample_inds, dtype=np.int32)
    edge_inds = np.asarray(edge_inds, dtype=np.int32)
    all_edges = np.asarray(all_edges, dtype=np.int32)
    all_peaks = np.concatenate(all_peaks)
    vecs_s = all_peaks[:, 0]
    vecs_t = all_peaks[:, 1]
    vecs = vecs_t - vecs_s
    lengths = np.linalg.norm(vecs, axis=1)
    xy = np.linspace(vecs_s, vecs_t, n_points, axis=1, dtype=np.int32)
    y = pafs[
        sample_inds.reshape((-1, 1)),
        xy[..., 0],
        xy[..., 1],
        edge_inds.reshape((-1, 1)),
    ]
    integ = np.trapz(y, xy[..., ::-1], axis=1)
    aff = np.linalg.norm(integ, axis=1) / lengths
    return (
        np.round(aff, decimals=n_decimals),
        np.round(lengths, decimals=n_decimals),
        all_edges,
        sample_inds,
        edge_inds,
    )


# TODO Add identity indexing
def compute_peaks_and_costs(
    scmaps,
    locrefs,
    pafs,
    graph,
    paf_inds,
    nms_radius=5,
    min_confidence=0.01,
    stride=8,
    n_points=10,
    n_decimals=3,
    session=None,
):
    peak_inds_in_batch = find_local_peak_indices(scmaps, nms_radius, min_confidence)
    if session is None:
        with tf.Session() as session:
            peak_inds_in_batch = session.run(peak_inds_in_batch)
    else:
        peak_inds_in_batch = session.run(peak_inds_in_batch)
    pos = calc_peak_locations(locrefs, peak_inds_in_batch, stride, n_decimals)
    prob = calc_peak_probabilities(scmaps, peak_inds_in_batch, n_decimals)
    costs = compute_edge_costs(
        pafs, peak_inds_in_batch, graph, n_points, n_decimals
    )

    # Reshape to nested arrays and cost matrices
    peaks_and_costs = []
    affinities, lengths, all_edges, sample_inds, edge_inds = costs
    n_samples, _, _, n_bodyparts = np.shape(scmaps)
    n_graph_edges = len(graph)
    for i in range(n_samples):
        # Form nested arrays
        xy = []
        p = []
        samples_i_mask = peak_inds_in_batch[:, 0] == i
        for j in range(n_bodyparts):
            bpts_j_mask = peak_inds_in_batch[:, 3] == j
            idx = np.flatnonzero(samples_i_mask & bpts_j_mask)
            xy.append(pos[idx])
            p.append(prob[idx])

        # Form cost matrices
        samples_i_mask2 = sample_inds == i
        costs = dict()
        for paf_ind, k in zip(paf_inds, range(n_graph_edges)):
            edges_k_mask = edge_inds == k
            idx = np.flatnonzero(samples_i_mask2 & edges_k_mask)
            s, t = all_edges[idx].T
            n_source_peaks = np.unique(s).size
            n_target_peaks = np.unique(t).size
            costs[paf_ind] = dict()
            costs[paf_ind]["m1"] = affinities[idx].reshape((n_source_peaks, n_target_peaks))
            costs[paf_ind]["distance"] = lengths[idx].reshape((n_source_peaks, n_target_peaks))

        dict_ = {"coordinates": (xy,), "confidence": p, "costs": costs}
        peaks_and_costs.append(dict_)

    return peaks_and_costs


def predict_batched_peaks_and_costs(
    pose_cfg,
    images_batch,
    sess,
    inputs,
    outputs,
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
        int(pose_cfg.get("nmsradius", 5)),
        pose_cfg.get("minconfidence", 0.01),
        pose_cfg["stride"],
        session=sess,
    )
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


def calc_peak_probabilities(
    scmaps,
    peak_inds_in_batch,
    n_decimals=3,
):
    s, r, c, b = peak_inds_in_batch.T
    prob = scmaps[s, r, c, b]
    return np.round(prob, decimals=n_decimals).reshape((-1, 1))


def extract_detections_python(cfg, scmap, locref, pafs, radius, threshold):
    Detections = {}
    stride = cfg["stride"]
    halfstride = stride * 0.5
    num_joints = cfg["num_joints"]
    unProb = [None] * num_joints
    unPos = [None] * num_joints

    for p_idx in range(num_joints):
        map_ = scmap[:, :, p_idx]
        xy = find_local_maxima(map_, radius, threshold)
        prob = map_[xy[:, 0], xy[:, 1]][:, np.newaxis]
        pos = xy[:, ::-1] * stride + halfstride + locref[xy[:, 0], xy[:, 1], p_idx]
        unProb[p_idx] = np.round(prob, 5)
        unPos[p_idx] = np.round(pos, 3)

    Detections["coordinates"] = (unPos,)
    Detections["confidence"] = unProb

    if pafs is not None:
        Detections["costs"] = AssociationCosts(cfg, unPos, pafs, stride, halfstride)
    else:
        Detections["costs"] = {}

    return Detections


def get_detectionswithcosts(
    image,
    cfg,
    sess,
    inputs,
    outputs,
    outall=False,
    nms_radius=5.0,
    det_min_score=0.1,
):
    """ Extract pose and association costs from PAFs """
    im = np.expand_dims(image, axis=0).astype(float)

    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref, paf = extract_cnn_output(outputs_np, cfg)
    detections = extract_detections_python(
        cfg, scmap, locref, paf, int(nms_radius), det_min_score
    )
    if outall:
        return scmap, locref, paf, detections
    else:
        return detections


def extract_detection_withgroundtruth_python(
    cfg, groundtruthcoordinates, scmap, locref, pafs, radius, threshold
):
    Detections = {}
    stride = cfg["stride"]
    halfstride = stride * 0.5
    num_joints = cfg["num_joints"]
    num_idchannel = cfg.get("num_idchannel", 0)
    unProb = [None] * num_joints
    unPos = [None] * num_joints
    unID = [None] * num_joints

    for p_idx in range(num_joints):
        map_ = scmap[:, :, p_idx]
        xy = find_local_maxima(map_, radius, threshold)
        prob = map_[xy[:, 0], xy[:, 1]][:, np.newaxis]
        pos = xy[:, ::-1] * stride + halfstride + locref[xy[:, 0], xy[:, 1], p_idx]
        unProb[p_idx] = np.round(prob, 5)
        unPos[p_idx] = np.round(pos, 3)
        if num_idchannel > 0:
            inds = [num_joints + id for id in range(num_idchannel)]
            cur_id = scmap[xy[:, 0], xy[:, 1]][:, inds]
            unID[p_idx] = np.round(cur_id, 5)

    Detections["coordinates"] = (unPos,)
    Detections["confidence"] = unProb
    if num_idchannel > 0:
        Detections["identity"] = unID

    if pafs is not None:
        Detections["costs"] = AssociationCosts(cfg, unPos, pafs, stride, halfstride)
        Detections["groundtruth_costs"] = AssociationCosts(
            cfg, groundtruthcoordinates, pafs, stride, halfstride
        )
    else:
        Detections["costs"] = {}
        Detections["groundtruth_costs"] = {}
    return Detections


def get_detectionswithcostsandGT(
    image,
    groundtruthcoordinates,
    cfg,
    sess,
    inputs,
    outputs,
    outall=False,
    nms_radius=5.0,
    det_min_score=0.1,
):
    """ Extract pose and association costs from PAFs """
    im = np.expand_dims(image, axis=0).astype(float)

    # if 'eval_scale' in cfg.keys():
    #     import imgaug.augmenters as iaa
    #     im = iaa.Resize(float(cfg['eval_scale']))(images=im)

    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref, paf = extract_cnn_output(outputs_np, cfg)
    detections = extract_detection_withgroundtruth_python(
        cfg,
        groundtruthcoordinates,
        scmap,
        locref,
        paf,
        int(nms_radius),
        det_min_score,
    )
    if outall:
        return scmap, locref, paf, detections
    else:
        return detections


## Functions below implement are for batch sizes > 1:
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


def extract_batchdetections_python(cfg, scmap, locref, pafs, threshold):
    Detections = {}
    stride = cfg["stride"]
    radius = int(cfg["nmsradius"])
    halfstride = stride * 0.5
    num_joints = cfg["num_joints"]
    num_idchannel = cfg.get("num_idchannel", 0)
    unProb = [None] * num_joints
    unPos = [None] * num_joints
    unID = [None] * num_joints

    for p_idx in range(num_joints):
        map_ = scmap[:, :, p_idx]
        xy = find_local_maxima(map_, radius, threshold)
        prob = map_[xy[:, 0], xy[:, 1]][:, np.newaxis]
        pos = xy[:, ::-1] * stride + halfstride + locref[xy[:, 0], xy[:, 1], p_idx]
        unProb[p_idx] = np.round(prob, 5)
        unPos[p_idx] = np.round(pos, 3)
        if num_idchannel > 0:
            inds = [num_joints + id for id in range(num_idchannel)]
            cur_id = scmap[xy[:, 0], xy[:, 1]][:, inds]
            unID[p_idx] = np.round(cur_id, 5)

    Detections["coordinates"] = (unPos,)
    Detections["confidence"] = unProb
    if num_idchannel > 0:
        Detections["identity"] = unID
    if pafs is not None:
        Detections["costs"] = AssociationCosts(cfg, unPos, pafs, stride, halfstride)
    else:
        Detections["costs"] = {}
    return Detections


def get_batchdetectionswithcosts(
    image,
    dlc_cfg,
    batchsize,
    det_min_score,
    sess,
    inputs,
    outputs,
    outall=False,
):
    outputs_np = sess.run(outputs, feed_dict={inputs: image})
    scmap, locref, pafs = extract_cnn_outputmulti(
        outputs_np, dlc_cfg
    )
    detections = []
    for l in range(batchsize):
        if pafs is None:
            paf = None
        else:
            paf = pafs[l]
        dets = extract_batchdetections_python(
            dlc_cfg, scmap[l], locref[l], paf, det_min_score
        )
        detections.append(dets)

    if outall:
        return scmap, locref, pafs, detections
    else:
        return detections
