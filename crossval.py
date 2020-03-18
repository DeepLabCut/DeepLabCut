import motmetrics as mm
import numpy as np
import os
os.environ['DLClight'] = 'True'
import pickle
import deeplabcut
import pandas as pd
from deeplabcut.refine_training_dataset.tracklets import TrackletManager
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


def reconstruct_all_bboxes(data, margin):
    animals = data.columns.get_level_values('individuals').unique()
    bboxes = np.full((len(animals), data.shape[0], 5), np.nan)
    for n, animal in enumerate(animals):
        bboxes[n] = reconstruct_bbox_from_bodyparts(data.xs(animal, axis=1, level=1), margin)
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
video = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.short.avi'
ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle0_30000tracks.h5'
ground_truth = pd.read_hdf(ground_truth_file)
trackname = ground_truth_file.replace('h5', 'pickle')

a = ground_truth.xs('f12', axis=1, level=1)


config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/dlc-models/iteration-0/silversideschoolingJul14-trainset95shuffle1/test/inference_cfg.yaml'
default_inference = edict(deeplabcut.auxiliaryfunctions.read_plainconfig(config_inference))
testing = default_inference.copy()
data_val = []
thresholds = np.linspace(0.1, 0.9, 9, endpoint=True)
min_hits = [100, 200, 500]
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


with open('/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle1_30000_full.pickle', 'rb') as file:
    data = pickle.load(file)
mot_tracker = trackingutils.Sort(default_inference)
all_jointnames=data['metadata']['all_joints_names']
numjoints=len(all_jointnames)
partaffinityfield_graph = data['metadata']['PAFgraph']
PAF=np.arange(len(partaffinityfield_graph))
partaffinityfield_graph=[partaffinityfield_graph[l] for l in PAF]
linkingpartaffinityfield_graph=partaffinityfield_graph
BPTS=iBPTS=range(numjoints)
imnames = [fn for fn in list(data) if fn != 'metadata']
Tracks = {}
for index, imname in tqdm(enumerate(imnames)):
    # filter detections according to inferencecfg parameters
    all_detections = inferenceutils.convertdetectiondict2listoflist(data[imname], imname, BPTS,
                                                                    withid=default_inference.withid)
    connection_all, special_k = inferenceutils.matchconnections(default_inference, data[imname],
                                                                all_detections, iBPTS, partaffinityfield_graph, PAF)
    subset, candidate = inferenceutils.linkjoints2individuals(default_inference, all_detections, connection_all, special_k,
                                                              linkingpartaffinityfield_graph, iBPTS,
                                                              numjoints=numjoints)
    sortedindividuals = np.argsort(-subset[:, -2])  # sort by top score!
    if len(sortedindividuals) > default_inference.topktoplot:
        sortedindividuals = sortedindividuals[:default_inference.topktoplot]

    animals = []
    for n in sortedindividuals:  # range(len(subset)): #number of individuals
        individual = np.zeros(3 * numjoints) * np.nan
        for i in range(numjoints):  # number of limbs
            ind = int(subset[n][i])  # bpt index in global coordinates
            if -1 == ind:  # reached the end!
                continue
            else:  # xyl=np.ones(3)*np.nan
                # else:
                # xyl = candidate[ind, :3]
                individual[3 * i:3 * i + 3] = candidate[ind, :3]
                # >> turn into bounding box :)
        animals.append(individual)
    bb = inferenceutils.individual2boundingbox(default_inference, animals, 0)  # TODO: get cropping parameters and utilize!
    trackers = mot_tracker.update(bb)
    for ind, content in enumerate(trackers):
        tracklet_id = content[4].astype(np.int)
        if tracklet_id in Tracks.keys():
            Tracks[tracklet_id][imname] = animals[content[5].astype(np.int)]
        else:
            Tracks[tracklet_id] = {}
            Tracks[tracklet_id][imname] = animals[content[5].astype(np.int)]  # retrieve coordinates


def compute_mot_metrics(ground_truth):
    acc = mm.MOTAccumulator(auto_id=True)
    a = np.array([
        [0, 0, 1, 2],  # Format X, Y, Width, Height
        [0, 0, 0.8, 1.5],
    ])

    b = np.array([
        [0, 0, 1, 2],
        [0, 0, 1, 1],
        [0.1, 0.2, 2, 2],
    ])
    dist = mm.distances.iou_matrix(a, b, max_iou=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc)
    print(summary)

