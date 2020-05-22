import numpy as np
import os

os.environ['DLClight'] = 'True'

import pickle
import deeplabcut
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm

import deeplabcut, pickle, os, sys
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
import numpy as np
import pandas as pd

#sys.path.append(os.path.join('/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/deeplabcut/pose_estimation_tensorflow/lib'))
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
import crossval

projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'

projectpath = '/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/MultiMouse-Daniel-2019-12-16'
modelprefix = 'simplebaseline'

configfile=os.path.join(projectpath,'config.yaml')

cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)

inferencecfg=edict()
inferencecfg.minimalnumberofconnections=3
inferencecfg.averagescore=.2
inferencecfg.distnormalization=400
inferencecfg.distnormalizationLOWER=0
inferencecfg.detectionthresholdsquare=.1
inferencecfg.addlikelihoods=0.
inferencecfg.pafthreshold=.1
inferencecfg.method='m1'
inferencecfg.withid=False
inferencecfg.slack=10
inferencecfg.variant=0 #
inferencecfg.topktoretain=3 #THIS SHOULD BE Larger than # animals!

#example use case for running
#data=crossval.compute_crossval_metrics(configfile, inferencecfg, shuffle=2, trainingsetindex=0)

#inferencecfg, opt = crossval.bayesian_search(configfile, shuffle=2, trainingsetindex=0, target='rmse_test', init_points=20, n_iter=50, acq='ei')


inferencecfg, opt = crossval.bayesian_search(configfile, shuffle=0, trainingsetindex=0, target='pck_test', 
                                                init_points=7, n_iter=50, acq='ei',maximize=True, 
                                                dcorr=6,leastbpts=3,modelprefix=modelprefix)

print(inferencecfg)
data=crossval.compute_crossval_metrics(configfile, inferencecfg, shuffle=0, trainingsetindex=0,modelprefix=modelprefix)
print(data)




'''
dcorr=5 #pixel distance cutoff
leastbpts=3 #at least 3 bpts

config_path=configfile
inference_cfg=inferencecfg
shuffle=2
trainingsetindex=0
modelprefix=''
snapshotindex=-1

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
fns = return_evaluate_network_data(config_path, shuffle=shuffle,
                                   trainingsetindex=trainingsetindex, modelprefix=modelprefix)

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

predictionsfn = fns[snapshotindex]
data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(predictionsfn)

params = set_up_evaluation(data)

n_images = len(params['imnames'])
stats = np.full((n_images, 6), np.nan)  # RMSE, hits, misses, false_pos, num_detections, pck
columns = ['train_iter', 'train_frac', 'shuffle']
columns += ['_'.join((b, a)) for a in ('train', 'test') for b in ('rmse', 'misses', 'hits', 'falsepos', 'ndetects', 'pck')]
for n, imname in enumerate(params['imnames']):
    animals = inferenceutils.assemble_individuals(inference_cfg, data[imname], params['num_joints'],
                                                    params['bpts'], params['ibpts'], params['paf'],
                                                    params['paf_graph'], params['paf_links'], evaluation=True)
    n_animals = len(animals)
    if n_animals:
        _, _, GT = data[imname]['groundtruth']
        GT = GT.droplevel('scorer').unstack(level=['bodyparts', 'coords'])
        gt = GT.values.reshape((GT.shape[0], -1, 2))
        
        if leastbpts>0: #ONLY KEEP animals with at least as many bpts (to get rid of crops that cannot be assembled)
            gt = gt[np.nansum(gt,axis=(1,2))>leastbpts]

        ani = np.stack(animals).reshape((n_animals, -1, 3))[:, :gt.shape[1], :2]
        mat = np.full((gt.shape[0], n_animals), np.nan)
        mat2 = np.full((gt.shape[0], n_animals), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            
            for i in range(len(gt)):
                numpts=np.nansum(gt[i]>0)//2
                for j in range(len(animals)):
                    #RMSE loss gt to detection
                    mat[i, j] = np.sqrt(np.nanmean(np.sum((gt[i] - ani[j, :, :2]) ** 2, axis=1))) #average over bpts
                    # rpck loss:
                    if numpts > 0:
                        dists=np.nanmean((gt[i]-ani[j])**2,axis=1) #squared dist per bpt!
                        mat2[i, j] = -np.sum(np.exp(-dists[np.isfinite(dists)]*1./(2*dcorr**2)))*1./numpts
                    else:
                        mat2[i, j] = -1.
                    
        mat[np.isnan(mat)] = np.nanmax(mat) + 1

        row_indices2, col_indices2 = linear_sum_assignment(mat2)
        row_indices, col_indices = linear_sum_assignment(mat)
        for i,r in enumerate(row_indices2):

        print('RMSE:', row_indices,row_indices2)
        print("rpck:", col_indices,col_indices2)

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
        
        stats[n, 5] = pck

train_iter = int(predictionsfn.split('-')[-1].split('.')[0])
train_frac = int(predictionsfn.split('trainset')[1].split('shuffle')[0])

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    res = np.r_[train_iter, train_frac, shuffle,
                np.nanmean(stats[metadata['data']['trainIndices']], axis=0),
                np.nanmean(stats[metadata['data']['testIndices']], axis=0)]


'''