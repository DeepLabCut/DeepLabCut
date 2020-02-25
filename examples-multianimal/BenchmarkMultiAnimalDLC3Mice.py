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

#REQUIRED FOR INFERENCE:
#nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
#sys.path.append(os.path.join('/home/alex/Dropbox/InterestingCode/sDeepLabCut',nmspath))
#sys.path.append(os.path.join('/mnt/md0/alex/multianimal/sDeepLabCut',nmspath))

nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))



#projectpath='/home/alex/Dropbox/InterestingCode/social_datasets/Marmoset-Mackenzie-2019-05-29'
projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'

config=os.path.join(projectpath,'config.yaml')

trainingsetindex=0

destfolder=None #os.path.join(projectpath,modelprefix)
for shuffle in [2]:

    trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.train_network_path(config,shuffle=shuffle,trainingsetindex=trainingsetindex)
    #cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)

    cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(testposeconfigfile)
    cfg_dlc['nmsradius']=5.
    cfg_dlc['minconfidence']=.01
    deeplabcut.auxiliaryfunctions.write_plainconfig(testposeconfigfile,cfg_dlc)

    print("Evaluating", shuffle, trainingsetindex)
    deeplabcut.evaluate_network(config,Shuffles=[shuffle],trainingsetindex=trainingsetindex)
