#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:25:28 2019

@author: alex
"""

import numpy as np

from pathlib import Path
import os, sys
os.environ['DLClight']='True'

import deeplabcut
import shutil

c_engine=False
saveiters=5000
displayiters=100

modelprefix='simplebaseline'


#REQUIRED FOR INFERENCE:
#nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
#sys.path.append(os.path.join('/home/alex/Dropbox/InterestingCode/sDeepLabCut',nmspath))
#sys.path.append(os.path.join('/mnt/md0/alex/multianimal/sDeepLabCut',nmspath))

nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))

#projectpath='/home/alex/Dropbox/InterestingCode/social_datasets/Marmoset-Mackenzie-2019-05-29'
projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/MultiMouse-Daniel-2019-12-16'
config=os.path.join(projectpath,'config.yaml')

#deeplabcut.check_labels(config)

#print("Crop images...")
#deeplabcut.cropimagesandlabels(config,userfeedback=False)
#deeplabcut.create_multianimaltraining_dataset(config,Shuffles=[0,1,2])
'''
import shutil
projectpath=Path(config).parents[0]
print("Copying datasets")
try:
    shutil.copytree(os.path.join(projectpath,'dlc-models'),os.path.join(projectpath,modelprefix,'dlc-models'))
except FileExistsError:
    print("Folder exists already...")
'''

trainingsetindex=0
videopath=os.path.join(projectpath,'videos')
destfolder=os.path.join(projectpath,modelprefix)

for shuffle in [0, 1,2]:
    trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(config,shuffle=shuffle,trainingsetindex=trainingsetindex,modelprefix=modelprefix)
    cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)

    cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
    #cfg_dlc['bank3']=128
    #cfg_dlc['bank5']=128
    #cfg_dlc['smfactor']=4
    #cfg_dlc['stride']=4
    cfg_dlc['augmentationprobability']=.6
    cfg_dlc['weigh_only_present_joints']=False
    cfg_dlc['cropratio']=.8
    cfg_dlc['cropfactor']=.2
    cfg_dlc['rotation']=180 #can also be an integer def. -10,10 if true.
    cfg_dlc['covering']=True
    cfg_dlc['motion_blur'] = True #[["k", 7],["angle", [-90, 90]]]
    deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

    print("Creating multianimal training set...")
    #deeplabcut.train_network(path_config_file, shuffle=shuffle,trainingsetindex=trainingsetindex,saveiters=saveiters,displayiters=displayiters)
    deeplabcut.train_network(config, shuffle=shuffle,trainingsetindex=trainingsetindex,modelprefix=modelprefix,
                            saveiters=saveiters,displayiters=displayiters,max_snapshots_to_keep=4,maxiters=15000)

    cfg_dlc = deeplabcut.auxiliaryfunctions.edit_config(testposeconfigfile, {'nmsradius': 5, 'minconfidence': .01})

    print("Evaluating", shuffle, trainingsetindex)
    deeplabcut.evaluate_network(config,Shuffles=[shuffle],trainingsetindex=trainingsetindex,c_engine=True,modelprefix=modelprefix)

    #deeplabcut.pose_estimation_tensorflow.calculatepafdistancebounds(config, shuffle=1)
    print("Evaluating", shuffle, trainingsetindex)
    deeplabcut.evaluate_network(config,Shuffles=[shuffle],trainingsetindex=trainingsetindex,c_engine=False,plotting=True,modelprefix=modelprefix)
    deeplabcut.extract_save_all_maps(config,shuffle=shuffle,trainingsetindex=trainingsetindex,modelprefix=modelprefix)

    print("Analyze video", shuffle, trainingsetindex)
    deeplabcut.convert_detections2tracklets(config,[videopath],videotype='.mp4', destfolder=destfolder,modelprefix=modelprefix)
    deeplabcut.create_labeled_video(config,[videopath],videotype='.mp4', destfolder=destfolder,modelprefix=modelprefix)
    