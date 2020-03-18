import motmetrics as mm
import numpy as np
import os
os.environ['DLClight'] = 'True'
import pickle
import deeplabcut
import pandas as pd
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
from easydict import EasyDict as edict
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


config = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/config.yaml'
config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/dlc-models/iteration-0/silversideschoolingJul14-trainset95shuffle1/test/inference_cfg.yaml'
video = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.short.avi'
ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle0_30000tracks.h5'
full_data_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle1_30000_full.pickle'

inference_cfg = edict(deeplabcut.auxiliaryfunctions.read_plainconfig(config_inference))
testing_cfg = inference_cfg.copy()
ground_truth = pd.read_hdf(ground_truth_file)
bboxes_ground_truth = reconstruct_all_bboxes(ground_truth, inference_cfg['boundingboxslack'], to_xywh=True)
ids = np.array(list(range(bboxes_ground_truth.shape[0])))
with open(full_data_file, 'rb') as file:
    data = pickle.load(file)

# data_val = []
# thresholds = np.linspace(0.1, 0.9, 9, endpoint=True)
# inferencecfg.max_age=100
# inferencecfg.min_hits=3
# for threshold in thresholds:
#     testing['iou_threshold'] = threshold
# for min_hit in min_hits:
#     testing['min_hits'] = min_hit
#     deeplabcut.convert_detections2tracklets(config, [video], inferencecfg=testing)
#     manager = TrackletManager(config, 0, 0)
#     manager.load_tracklets_from_pickle(trackname)
#     data_val.append(manager.xy.swapaxes(0, 1).reshape((manager.nframes, -1)))
# for data in data_val:
#     print(np.nanmean(data - ground_truth))


mot_tracker = trackingutils.Sort(testing_cfg)
all_jointnames=data['metadata']['all_joints_names']
numjoints=len(all_jointnames)
partaffinityfield_graph = data['metadata']['PAFgraph']
PAF=np.arange(len(partaffinityfield_graph))
partaffinityfield_graph=[partaffinityfield_graph[l] for l in PAF]
linkingpartaffinityfield_graph=partaffinityfield_graph
BPTS=iBPTS=range(numjoints)
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

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
