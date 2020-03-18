import motmetrics as mm
import numpy as np
import os
os.environ['DLClight'] = 'True'
import pickle
import deeplabcut
import pandas as pd
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
from easydict import EasyDict as edict
from itertools import product
from tqdm import tqdm


def reconstruct_bbox_from_bodyparts(data, margin, to_xywh=False):
    bbox = np.full((data.shape[0], 5), np.nan)
    x = data.xs('x', axis=1, level=-1)
    y = data.xs('y', axis=1, level=-1)
    bbox[:, 0] = np.nanmin(x, axis=1) - margin
    bbox[:, 1] = np.nanmin(y, axis=1) - margin
    bbox[:, 2] = np.nanmax(x, axis=1) + margin
    bbox[:, 3] = np.nanmax(y, axis=1) + margin
    bbox[:, -1] = np.nanmean(data.xs('likelihood', axis=1, level=-1), axis=1)
    if to_xywh:
        convert_bbox_to_xywh(bbox, inplace=True)
    return bbox


def reconstruct_all_bboxes(data, margin, to_xywh):
    animals = data.columns.get_level_values('individuals').unique()
    bboxes = np.full((len(animals), data.shape[0], 5), np.nan)
    for n, animal in enumerate(animals):
        bboxes[n] = reconstruct_bbox_from_bodyparts(data.xs(animal, axis=1, level=1), margin, to_xywh)
    return bboxes


def convert_bbox_to_xywh(bbox, inplace=False):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    if not inplace:
        new_bbox = bbox.copy()
        new_bbox[:, 2] = w
        new_bbox[:, 3] = h
        return new_bbox
    bbox[:, 2] = w
    bbox[:, 3] = h


def compute_mot_metrics(inference_cfg, data, bboxes_ground_truth):
    mot_tracker = trackingutils.Sort(inference_cfg)
    all_jointnames = data['metadata']['all_joints_names']
    numjoints = len(all_jointnames)
    partaffinityfield_graph = data['metadata']['PAFgraph']
    PAF = np.arange(len(partaffinityfield_graph))
    partaffinityfield_graph = [partaffinityfield_graph[l] for l in PAF]
    linkingpartaffinityfield_graph = partaffinityfield_graph
    BPTS = iBPTS = range(numjoints)
    ids = np.array(list(range(bboxes_ground_truth.shape[0])))
    imnames = [fn for fn in list(data) if fn != 'metadata']
    acc = mm.MOTAccumulator(auto_id=True)
    for i, imname in enumerate(tqdm(imnames)):
        animals = inferenceutils.assemble_individuals(inference_cfg, data[imname], numjoints, BPTS, iBPTS,
                                                      PAF, partaffinityfield_graph, linkingpartaffinityfield_graph)
        bb = inferenceutils.individual2boundingbox(inference_cfg, animals)
        trackers = mot_tracker.update(bb)
        bboxes_hyp = convert_bbox_to_xywh(trackers[:, :4])
        bboxes_gt = bboxes_ground_truth[:, i, :4]
        ids_gt = ids.copy()
        empty = np.isnan(bboxes_gt).any(axis=1)
        if empty.any():
            bboxes_gt = bboxes_gt[~empty]
            ids_gt = ids_gt[~empty]
        dist = mm.distances.iou_matrix(bboxes_gt, bboxes_hyp, max_iou=inference_cfg['iou_threshold'])
        acc.update(ids_gt, trackers[:, 4], dist)
    return acc


def print_all_metrics(accumulators, all_params=None):
    if not all_params:
        names = [f'iter{i + 1}' for i in range(len(accumulators))]
    else:
        names = ['{:.1f}_{}_{}'.format(*params) for params in all_params]
    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, metrics=mm.metrics.motchallenge_metrics, names=names)
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)


config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/dlc-models/iteration-0/silversideschoolingJul14-trainset95shuffle1/test/inference_cfg.yaml'
ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle0_30000tracks.h5'
full_data_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle1_30000_full.pickle'
inference_cfg = edict(deeplabcut.auxiliaryfunctions.read_plainconfig(config_inference))
testing_cfg = edict(inference_cfg.copy())
ground_truth = pd.read_hdf(ground_truth_file)
bboxes_ground_truth = reconstruct_all_bboxes(ground_truth, inference_cfg['boundingboxslack'], to_xywh=True)
with open(full_data_file, 'rb') as file:
    data = pickle.load(file)

accumulators = []
thresholds = np.linspace(0.1, 0.9, 5, endpoint=True)
max_ages = [1, 3, 5, 10]
min_hits = [1, 3, 5, 10]
combinations = list(product(thresholds, max_ages, min_hits))
for threshold, max_age, min_hit in combinations:
    testing_cfg['iou_threshold'] = threshold
    testing_cfg['max_age'] = max_age
    testing_cfg['min_hits'] = min_hit
    accumulators.append(compute_mot_metrics(testing_cfg, data, bboxes_ground_truth))
print_all_metrics(accumulators, combinations)
