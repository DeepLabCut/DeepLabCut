import motmetrics as mm
import numpy as np
import os
os.environ['DLClight'] = 'True'
import pickle
import deeplabcut
import pandas as pd
import warnings
from bayes_opt import BayesianOptimization
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
from deeplabcut import return_evaluate_network_data
from deeplabcut.utils import auxfun_multianimal
from deeplabcut.refine_training_dataset.tracklets import TrackletManager
from easydict import EasyDict as edict
from itertools import product
from scipy.optimize import linear_sum_assignment
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


def kalman_to_bbox(arr, margin):
    length = arr.shape[0]
    arr_reshaped = arr.reshape((length, -1, 2))
    bboxes = np.zeros((length, 4))
    bboxes[:, :2] = np.min(arr_reshaped, axis=1) - margin
    bboxes[:, 2:] = np.max(arr_reshaped, axis=1) + margin
    return convert_bbox_to_xywh(bboxes)


def set_up_evaluation(data):
    params = dict()
    params['joint_names'] = data['metadata']['all_joints_names']
    params['num_joints'] = len(params['joint_names'])
    partaffinityfield_graph = data['metadata']['PAFgraph']
    params['paf'] = np.arange(len(partaffinityfield_graph))
    params['paf_graph'] = params['paf_links'] = [partaffinityfield_graph[l] for l in params['paf']]
    params['bpts'] = params['ibpts'] = range(params['num_joints'])
    params['imnames'] = [fn for fn in list(data) if fn != 'metadata']
    return params


def compute_mot_metrics(inference_cfg, data, bboxes_ground_truth):
    params = set_up_evaluation(data)
    ids = np.array(list(range(bboxes_ground_truth.shape[0])))
    tracklets = dict()
    tracklets['header'] = pd.MultiIndex.from_product([[''], params['joint_names'], ['x', 'y', 'likelihood']],
                                                     names=['scorer', 'bodyparts', 'coords'])
    acc = mm.MOTAccumulator(auto_id=True)
    mot_tracker = trackingutils.Sort(inference_cfg)
    for i, imname in enumerate(tqdm(params['imnames'])):
        animals = inferenceutils.assemble_individuals(inference_cfg, data[imname], params['num_joints'],
                                                      params['bpts'], params['ibpts'], params['paf'],
                                                      params['paf_graph'], params['paf_links'])
        bb = inferenceutils.individual2boundingbox(inference_cfg, animals)
        trackers = mot_tracker.update(bb)
        trackingutils.fill_tracklets(tracklets, trackers, animals, imname)
        bboxes_hyp = convert_bbox_to_xywh(trackers[:, :4])
        bboxes_gt = bboxes_ground_truth[:, i, :4]
        ids_gt = ids.copy()
        empty = np.isnan(bboxes_gt).any(axis=1)
        if empty.any():
            bboxes_gt = bboxes_gt[~empty]
            ids_gt = ids_gt[~empty]
        dist = mm.distances.iou_matrix(bboxes_gt, bboxes_hyp, max_iou=inference_cfg['iou_threshold'])
        acc.update(ids_gt, trackers[:, 4], dist)
    return acc, tracklets


def compute_mot_metrics_new_tracker(inference_cfg, data, bboxes_ground_truth):
    params = set_up_evaluation(data)
    ids = np.array(list(range(bboxes_ground_truth.shape[0])))
    tracklets = dict()
    tracklets['header'] = pd.MultiIndex.from_product([[''], params['joint_names'], ['x', 'y', 'likelihood']],
                                                     names=['scorer', 'bodyparts', 'coords'])
    acc = mm.MOTAccumulator(auto_id=True)
    sort = trackingutils.SORT(params['num_joints'], inference_cfg['max_age'], inference_cfg['min_hits'])
    for i, imname in enumerate(tqdm(params['imnames'])):
        animals = inferenceutils.assemble_individuals(inference_cfg, data[imname], params['num_joints'],
                                                      params['bpts'], params['ibpts'], params['paf'],
                                                      params['paf_graph'], params['paf_links'])
        temp = [arr.reshape((-1, 3))[:, :2] for arr in animals]
        trackers = sort.track(temp)
        trackingutils.fill_tracklets(tracklets, trackers, animals, imname)
        nonempty = ~np.isnan(trackers).any(axis=1)
        if not nonempty.any():
            bboxes_hyp = []
            ids_hyp = []
        else:
            arr = trackers[nonempty]
            bboxes_hyp = kalman_to_bbox(arr[:, :-2], inference_cfg['boundingboxslack'])
            ids_hyp = arr[:, -2]
        bboxes_gt = bboxes_ground_truth[:, i, :4]
        ids_gt = ids.copy()
        empty = np.isnan(bboxes_gt).any(axis=1)
        if empty.any():
            bboxes_gt = bboxes_gt[~empty]
            ids_gt = ids_gt[~empty]
        dist = mm.distances.iou_matrix(bboxes_gt, bboxes_hyp)
        acc.update(ids_gt, ids_hyp, dist)
    return acc, tracklets


def compute_crossval_metrics(config_path, inference_cfg, shuffle=1, trainingsetindex=0,
                                modelprefix='',snapshotindex=-1,dcorr=5):
    fns = return_evaluate_network_data(config_path, shuffle=shuffle,
                                       trainingsetindex=trainingsetindex, modelprefix=modelprefix)

    predictionsfn = fns[snapshotindex]
    data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(predictionsfn)

    params = set_up_evaluation(data)

    n_images = len(params['imnames'])
    stats = np.full((n_images, 7), np.nan)  # RMSE, hits, misses, false_pos, num_detections, pck
    columns = ['train_iter', 'train_frac', 'shuffle']
    columns += ['_'.join((b, a)) for a in ('train', 'test') for b in ('rmse',  'hits', 'misses', 'falsepos', 'ndetects', 'pck', 'rpck')]
    for n, imname in enumerate(params['imnames']):
        animals = inferenceutils.assemble_individuals(inference_cfg, data[imname], params['num_joints'],
                                                        params['bpts'], params['ibpts'], params['paf'],
                                                        params['paf_graph'], params['paf_links'], evaluation=True)
        n_animals = len(animals)
        if n_animals:
            _, _, GT = data[imname]['groundtruth']
            GT = GT.droplevel('scorer').unstack(level=['bodyparts', 'coords'])
            gt = GT.values.reshape((GT.shape[0], -1, 2))
            ani = np.stack(animals).reshape((n_animals, -1, 3))[:, :gt.shape[1], :2]
            mat = np.full((gt.shape[0], n_animals), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for i in range(len(gt)):
                    for j in range(len(animals)):
                        mat[i, j] = np.sqrt(np.nanmean(np.sum((gt[i] - ani[j, :, :2]) ** 2, axis=1)))

            mat[np.isnan(mat)] = np.nanmax(mat) + 1
            row_indices, col_indices = linear_sum_assignment(mat)
            stats[n, 0] = mat[row_indices, col_indices].mean() #rmse

            gt_annot = np.any(~np.isnan(gt), axis=2)
            gt_matched = gt_annot[row_indices].flatten()

            dlc_annot = np.any(~np.isnan(ani), axis=2) #DLC assemblies
            dlc_matched = dlc_annot[col_indices].flatten()

            stats[n, 1] = np.logical_and(gt_matched, dlc_matched).sum() #hits
            stats[n, 2] = gt_annot.sum() - stats[n, 1] #misses
            stats[n, 3] = np.logical_and(~gt_matched, dlc_matched).sum() #additional detections
            stats[n, 4] = n_animals

            numgtpts=gt_annot.sum()
            #animal & bpt-wise distance!
            if numgtpts>0:
                #corrkps=np.sum((gt[row_indices]-ani[col_indices])**2,axis=2)<dcorr**2
                dists=np.sum((gt[row_indices]-ani[col_indices])**2,axis=2)
                corrkps=dists[np.isfinite(dists)]<dcorr**2
                pck = corrkps.sum()*1./numgtpts  #weigh by actually annotated ones!
                rpck=np.sum(np.exp(-dists[np.isfinite(dists)]*1./(2*dcorr**2)))*1./numgtpts

            else:
                pck = 1. #does that make sense? As a convention fully correct...
                rpck= 1.

            stats[n, 5] = pck
            stats[n, 6] = rpck

    train_iter = int(predictionsfn.split('-')[-1].split('.')[0])
    train_frac = int(predictionsfn.split('trainset')[1].split('shuffle')[0])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        res = np.r_[train_iter, train_frac, shuffle,
                    np.nanmean(stats[metadata['data']['trainIndices']], axis=0),
                    np.nanmean(stats[metadata['data']['testIndices']], axis=0)]

    return pd.DataFrame(res.reshape((1, -1)), columns=columns)


def bayesian_search(config_path, shuffle=1, trainingsetindex=0, target='rmse_test',
                    maximize=False, init_points=20, n_iter=50, acq='ei',dcorr=5):
    inferencecfg = edict()
    inferencecfg.withid = False
    inferencecfg.method = 'm1'
    inferencecfg.slack = 10
    inferencecfg.variant = 0
    inferencecfg.topktoplot = np.inf
    inferencecfg.distnormalization=400
    inferencecfg.distnormalizationLOWER=5
    inferencecfg.addlikelihoods=.1
    inferencecfg.averagescore=.1 #least score of detected animal

    pbounds = {
        'pafthreshold': (0.05, 0.5),
        'detectionthresholdsquare': (0.01, 0.9),
        'minimalnumberofconnections': (4, 15),
    }

    '''
    pbounds = {
        'distnormalization': (100, 500),
        'distnormalizationLOWER': (0, 50),
        'pafthreshold': (0.05, 0.5),
        'addlikelihoods': (0, 1),
        'detectionthresholdsquare': (0.01, 0.9),
        'minimalnumberofconnections': (4, 12),
        'averagescore': (0.1, 1)
    }
    def dlc_hyperparams(distnormalization,
                        distnormalizationLOWER,
                        pafthreshold,
                        addlikelihoods,
                        detectionthresholdsquare,
                        minimalnumberofconnections,
                        averagescore):

        inferencecfg.minimalnumberofconnections = int(minimalnumberofconnections)
        inferencecfg.averagescore = averagescore

        inferencecfg.distnormalization = distnormalization
        inferencecfg.distnormalizationLOWER = distnormalizationLOWER
        inferencecfg.detectionthresholdsquare = detectionthresholdsquare
        inferencecfg.addlikelihoods = addlikelihoods
        inferencecfg.pafthreshold = pafthreshold

    '''
    def dlc_hyperparams(pafthreshold,
                        detectionthresholdsquare,
                        minimalnumberofconnections):

        inferencecfg.minimalnumberofconnections = int(minimalnumberofconnections)
        inferencecfg.detectionthresholdsquare = detectionthresholdsquare
        inferencecfg.pafthreshold = pafthreshold

        stats = compute_crossval_metrics(config_path, inferencecfg, shuffle,trainingsetindex,dcorr=dcorr)
        val = stats[target].values[0]
        print("rmse", stats['rmse_test'].values[0], "miss", stats['misses_test'].values[0], "hit", stats['hits_test'].values[0])
        print("pck", stats['pck_test'].values[0], "pck", stats['pck_train'].values[0])
        print("rpck", stats['rpck_test'].values[0], "rpck", stats['rpck_train'].values[0])
        #val = stats['rmse_test'].values[0]*(1+stats['misses_test'].values[0]*1./stats['hits_test'].values[0])
        if np.isnan(val):
            val = 1e9
        if not maximize:
            val = -val
        return val

    opt = BayesianOptimization(f=dlc_hyperparams, pbounds=pbounds, random_state=42)
    opt.maximize(init_points=init_points, n_iter=n_iter, acq=acq)

    inferencecfg.update(opt.max['params'])
    inferencecfg.minimalnumberofconnections = int(inferencecfg.minimalnumberofconnections)

    return inferencecfg, opt


def evaluate_skeleton_tracker():
    mouse = ground_truth.xs('mus1', level='individuals', axis=1).drop('likelihood', axis=1, level=-1).values
    all_jointnames = data['metadata']['all_joints_names']
    numjoints = len(all_jointnames)

    tracker = trackingutils.SkeletonTracker(numjoints)
    tracker.state = mouse[0]
    mouse_pred = np.zeros_like(mouse)
    mouse_pred2 = np.zeros_like(mouse)
    for i in range(1, len(mouse)):
        mouse_pred[i] = tracker.predict()
        tracker.update(mouse[i])
        mouse_pred2[i] = tracker.state

    # TODO UKF not worth the cost, no remarkable diff vs KF
    # from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    #
    # def hx(x):
    #     return x[:numjoints * 2]
    #
    # def fx(x, _):
    #     F = tracker.kf.F
    #     return np.dot(F, x)
    #
    # points = MerweScaledSigmaPoints(numjoints * 4, alpha=.1, beta=2., kappa=-1)
    # ukf = UnscentedKalmanFilter(numjoints * 4, numjoints * 2, 1, hx, fx, points)
    # ukf.x[:numjoints * 2] = mouse[0]
    # mouse_pred2 = np.zeros_like(mouse)
    # for i in range(1, len(mouse)):
    #     ukf.predict()
    #     meas = mouse[i]
    #     if np.isnan(meas).any():
    #         ukf.update(None)
    #     else:
    #         ukf.update(meas)
    #     mouse_pred2[i] = ukf.x[:numjoints * 2]


def compute_mot_metrics_bboxes(inference_cfg, bboxes, bboxes_ground_truth):
    if bboxes.shape != bboxes_ground_truth.shape:
        raise ValueError('Dimension mismatch. Check the inputs.')

    ids = np.array(list(range(bboxes_ground_truth.shape[0])))
    acc = mm.MOTAccumulator(auto_id=True)
    for i in range(bboxes_ground_truth.shape[1]):
        bboxes_hyp = bboxes[:, i, :4]
        bboxes_gt = bboxes_ground_truth[:, i, :4]
        empty_hyp = np.isnan(bboxes_hyp).any(axis=1)
        empty_gt = np.isnan(bboxes_gt).any(axis=1)
        bboxes_hyp = bboxes_gt[~empty_hyp]
        bboxes_gt = bboxes_gt[~empty_gt]
        dist = mm.distances.iou_matrix(bboxes_gt, bboxes_hyp, max_iou=inference_cfg['iou_threshold'])
        acc.update(ids[~empty_hyp], ids[~empty_gt], dist)
    return acc


def print_all_metrics(accumulators, all_params=None):
    if not all_params:
        names = [f'iter{i + 1}' for i in range(len(accumulators))]
    else:
        s = '_'.join('{}' for _ in range(len(all_params[0])))
        names = [s.format(*params) for params in all_params]
    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, metrics=mm.metrics.motchallenge_metrics, names=names)
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)


def example_run():
    # config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/dlc-models/iteration-0/silversideschoolingJul14-trainset95shuffle1/test/inference_cfg.yaml'
    # ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle0_30000tracks.h5'
    # full_data_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.shortDLC_resnet50_silversideschoolingJul14shuffle1_30000_full.pickle'

    config = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/config.yaml'
    config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/inference_cfg.yaml'
    ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/videocompressed3DLC_resnet50_MultiMouseDec16shuffle2_20000tracks1.h5'
    full_data_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/videocompressed3DLC_resnet50_MultiMouseDec16shuffle2_20000_full.pickle'

    # config = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/Marmoset-Mackenzie-2019-05-29/config.yaml'
    # config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/Marmoset-Mackenzie-2019-05-29/inference_cfg.yaml'
    # ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/Marmoset-Mackenzie-2019-05-29/ultrashort_croppedDLC_resnet50_MarmosetMay29shuffle0_20000tracks.h5'
    # full_data_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/Marmoset-Mackenzie-2019-05-29/ultrashort_croppedDLC_resnet50_MarmosetMay29shuffle0_20000_full.pickle'

    projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'
    config =os.path.join(projectpath,'config.yaml')

    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data = pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')


    config = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/config.yaml'
    config_inference = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/inference_cfg.yaml'

    ground_truth_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/videocompressed3DLC_resnet50_MultiMouseDec16shuffle2_20000tracks1.h5'
    full_data_file = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/MultiMouse-Daniel-2019-12-16/videocompressed3DLC_resnet50_MultiMouseDec16shuffle2_20000_full.pickle'


    inference_cfg = edict(deeplabcut.auxiliaryfunctions.read_plainconfig(config_inference))
    testing_cfg = edict(inference_cfg.copy())
    ground_truth = pd.read_hdf(ground_truth_file)

    bboxes_ground_truth = reconstruct_all_bboxes(ground_truth, inference_cfg['boundingboxslack'], to_xywh=True)
    with open(full_data_file, 'rb') as file:
        data = pickle.load(file)

    # manager = TrackletManager(config, 0, 0)
    accumulators = []
    # accumulators_with_loader = []
    # thresholds = np.linspace(0.1, 0.9, 5, endpoint=True)
    max_ages = [1, 5, 20, 50]
    min_hits = [1, 5, 10]
    combinations = list(product(max_ages, min_hits))
    ntot = len(combinations)
    # combinations = list(product(thresholds, max_ages, min_hits))
    for n, (max_age, min_hit) in enumerate(combinations, start=1):
        print(f'Combination {n}/{ntot}')
        testing_cfg['max_age'] = max_age
        testing_cfg['min_hits'] = min_hit
        acc, tracklets = compute_mot_metrics_new_tracker(testing_cfg, data, bboxes_ground_truth)
        # acc, tracklets = compute_mot_metrics(testing_cfg, data, bboxes_ground_truth)
        accumulators.append(acc)
        # Evaluate the effect of the tracklet loader
        # manager._load_tracklets(tracklets, auto_fill=True)
        # df = manager.format_data()
        # bboxes_with_loader = reconstruct_all_bboxes(df, testing_cfg['boundingboxslack'], to_xywh=True)
        # accumulators_with_loader.append(compute_mot_metrics_bboxes(testing_cfg, bboxes_with_loader, bboxes_ground_truth))
    print_all_metrics(accumulators, combinations)
    # print_all_metrics(accumulators_with_loader, combinations)
