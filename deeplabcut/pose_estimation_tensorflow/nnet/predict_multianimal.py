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


# TODO: compute all the grids etc. once for the video analysis method
# (and then just pass on the variables)
def pos_from_grid_raw(gridpos, stride, halfstride):
    return gridpos * stride + halfstride


def make_nms_grid(nms_radius):
    nms_radius = int(np.ceil(nms_radius))
    size = np.arange(2 * nms_radius + 1)
    xx, yy = np.meshgrid(size, size)
    dist_grid = np.where(
        (xx - nms_radius) ** 2 + (yy - nms_radius) ** 2 <= nms_radius ** 2, 1, 0
    )
    return np.array(dist_grid, dtype=np.uint8)


def extract_detections(cfg, scmap, locref, pafs, nms_radius, det_min_score):
    """ Extract detections correcting by locref and estimating association costs based on PAFs """
    from nms_grid import nms_grid  # this needs to be installed (C-code)

    Detections = {}
    stride = cfg["stride"]
    halfstride = stride * 0.5
    num_joints = cfg["num_joints"]
    dist_grid = make_nms_grid(nms_radius)
    unProb = [None] * num_joints
    unPos = [None] * num_joints

    # apply nms
    for p_idx in range(num_joints):
        # IMPORTANT, as C++ function expects row-major
        prob_map = np.ascontiguousarray(scmap[:, :, p_idx])
        dets = nms_grid(prob_map, dist_grid, det_min_score)
        cur_prob = np.zeros([len(dets), 1], dtype=np.float64)
        cur_pos = np.zeros([len(dets), 2], dtype=np.float64)

        for idx, didx in enumerate(dets):
            ix = didx % scmap.shape[1]
            iy = didx // scmap.shape[1]
            cur_prob[idx, 0] = scmap[iy, ix, p_idx]  # prob
            cur_pos[idx, :] = (
                pos_from_grid_raw(np.array([ix, iy]), stride, halfstride)
                + locref[iy, ix, p_idx, :]
            )  # scmap + locrefinment!

        unProb[p_idx] = np.round(cur_prob, 5)
        unPos[p_idx] = np.round(cur_pos, 3)

    Detections["coordinates"] = (unPos,)
    Detections["confidence"] = unProb
    if pafs is not None:
        Detections["costs"] = AssociationCosts(cfg, unPos, pafs, stride, halfstride)
    else:
        Detections["costs"] = {}

    return Detections


def find_local_maxima(scmap, radius, threshold):
    peak_idx = peak_local_max(
        scmap, min_distance=radius, threshold_abs=threshold, exclude_border=False
    )
    grid = np.zeros_like(scmap, dtype=bool)
    grid[tuple(peak_idx.T)] = True
    labels = measurements.label(grid)[0]
    xy = measurements.center_of_mass(grid, labels, range(1, np.max(labels) + 1))
    return np.asarray(xy, dtype=np.int).reshape((-1, 2))


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
    c_engine=False,
):
    """ Extract pose and association costs from PAFs """
    im = np.expand_dims(image, axis=0).astype(float)

    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref, paf = extract_cnn_output(outputs_np, cfg)

    if c_engine:
        detections = extract_detections(
            cfg, scmap, locref, paf, nms_radius=nms_radius, det_min_score=det_min_score
        )
    else:
        detections = extract_detections_python(
            cfg, scmap, locref, paf, int(nms_radius), det_min_score
        )
    if outall:
        return scmap, locref, paf, detections
    else:
        return detections


# These two functions are for evaluation specifically (one also calculates integral between gt points)
def extract_detection_withgroundtruth(
    cfg, groundtruthcoordinates, scmap, locref, pafs, nms_radius, det_min_score
):
    """ Extract detections correcting by locref and estimating association costs based on PAFs """
    from nms_grid import nms_grid  # this needs to be installed (C-code)

    Detections = {}
    num_idchannel = cfg.get("num_idchannel", 0)
    stride = cfg["stride"]
    halfstride = stride * 0.5
    num_joints = cfg["num_joints"]
    # get dist_grid
    dist_grid = make_nms_grid(nms_radius)
    unProb = [None] * num_joints
    unPos = [None] * num_joints
    unID = [None] * num_joints
    # apply nms
    for p_idx in range(num_joints):
        # IMPORTANT, as C++ function expects row-major
        prob_map = np.ascontiguousarray(scmap[:, :, p_idx])
        dets = nms_grid(prob_map, dist_grid, det_min_score)
        cur_prob = np.zeros([len(dets), 1], dtype=np.float64)
        cur_pos = np.zeros([len(dets), 2], dtype=np.float64)
        if num_idchannel > 0:
            cur_id = np.zeros([len(dets), num_idchannel], dtype=np.float64)

        for idx, didx in enumerate(dets):
            ix = didx % scmap.shape[1]
            iy = didx // scmap.shape[1]
            cur_prob[idx, 0] = scmap[iy, ix, p_idx]  # prob
            cur_pos[idx, :] = (
                pos_from_grid_raw(np.array([ix, iy]), stride, halfstride)
                + locref[iy, ix, p_idx, :]
            )  # scmap + locrefinment!
            for id in range(num_idchannel):
                cur_id[idx, id] = np.amax(scmap[iy, ix, num_joints + id])

        if num_idchannel > 0:
            unID[p_idx] = np.round(cur_id, 5)
        unProb[p_idx] = np.round(cur_prob, 5)
        unPos[p_idx] = np.round(cur_pos, 3)

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
    c_engine=False,
):
    """ Extract pose and association costs from PAFs """
    im = np.expand_dims(image, axis=0).astype(float)

    # if 'eval_scale' in cfg.keys():
    #     import imgaug.augmenters as iaa
    #     im = iaa.Resize(float(cfg['eval_scale']))(images=im)

    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref, paf = extract_cnn_output(outputs_np, cfg)

    if c_engine:
        detections = extract_detection_withgroundtruth(
            cfg, groundtruthcoordinates, scmap, locref, paf, nms_radius, det_min_score
        )
    else:
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


def extract_batchdetections(
    scmap,
    locref,
    pafs,
    cfg,
    dist_grid,
    num_joints,
    num_idchannel,
    stride,
    halfstride,
    det_min_score,
):
    """ Extract detections correcting by locref and estimating association costs based on PAFs """
    from nms_grid import nms_grid  # this needs to be installed (C-code)

    Detections = {}
    # get dist_grid
    unProb = [None] * num_joints
    unPos = [None] * num_joints
    unID = [None] * num_joints
    # apply nms
    for p_idx in range(num_joints):
        # IMPORTANT, as C++ function expects row-major
        prob_map = np.ascontiguousarray(scmap[:, :, p_idx])
        dets = nms_grid(prob_map, dist_grid, det_min_score)
        cur_prob = np.zeros([len(dets), 1], dtype=np.float64)
        cur_pos = np.zeros([len(dets), 2], dtype=np.float64)
        if num_idchannel > 0:
            cur_id = np.zeros([len(dets), num_idchannel], dtype=np.float64)

        for idx, didx in enumerate(dets):
            ix = didx % scmap.shape[1]
            iy = didx // scmap.shape[1]
            cur_prob[idx, 0] = scmap[iy, ix, p_idx]  # prob
            cur_pos[idx, :] = (
                pos_from_grid_raw(np.array([ix, iy]), stride, halfstride)
                + locref[iy, ix, p_idx, :]
            )  # scmap + locrefinment!
            for id in range(num_idchannel):
                cur_id[idx, id] = np.amax(scmap[iy, ix, num_joints + id])

        if num_idchannel > 0:
            unID[p_idx] = np.round(cur_id, 5)
        unProb[p_idx] = np.round(cur_prob, 5)
        unPos[p_idx] = np.round(cur_pos, 3)

    Detections["coordinates"] = (unPos,)
    Detections["confidence"] = unProb
    if num_idchannel > 0:
        Detections["identity"] = unID
    if pafs is not None:
        Detections["costs"] = AssociationCosts(cfg, unPos, pafs, stride, halfstride)
    else:
        Detections["costs"] = {}
    return Detections


def extract_batchdetections_python(cfg, scmap, locref, pafs, radius, threshold):
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
    else:
        Detections["costs"] = {}
    return Detections


def get_batchdetectionswithcosts(
    image,
    dlc_cfg,
    dist_grid,
    batchsize,
    num_joints,
    num_idchannel,
    stride,
    halfstride,
    det_min_score,
    sess,
    inputs,
    outputs,
    outall=False,
    c_engine=False,
):

    outputs_np = sess.run(outputs, feed_dict={inputs: image})
    scmap, locref, pafs = extract_cnn_outputmulti(
        outputs_np, dlc_cfg
    )  # processes image batch.
    # batchsize,ny,nx,num_joints = scmap.shape
    detections = []
    for l in range(batchsize):
        if pafs is None:
            paf = None
        else:
            paf = pafs[l]
        if c_engine:
            dets = extract_batchdetections(
                scmap[l],
                locref[l],
                paf,
                dlc_cfg,
                dist_grid,
                num_joints,
                num_idchannel,
                stride,
                halfstride,
                det_min_score,
            )
        else:
            radius = len(dist_grid - 1) // 2
            dets = extract_batchdetections_python(
                dlc_cfg, scmap[l], locref[l], paf, radius, det_min_score
            )
        detections.append(dets)

    if outall:
        return scmap, locref, pafs, detections
    else:
        return detections
