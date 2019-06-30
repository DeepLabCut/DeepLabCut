#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

Note that the same ResNet 152 is trained 4 times (twice with the standard loader).
The sequence of losses is different....

Then twice with the 'deterministic' loader. The losses are identical when this is done twice. 

I.e. I get twice: ;)
iteration: 1 loss: 1.6505 lr: 0.001
iteration: 2 loss: 0.6929 lr: 0.001
iteration: 3 loss: 0.6420 lr: 0.001
iteration: 4 loss: 0.5579 lr: 0.001
iteration: 5 loss: 0.4746 lr: 0.001
iteration: 6 loss: 0.3366 lr: 0.001
iteration: 7 loss: 0.3194 lr: 0.001
iteration: 8 loss: 0.2561 lr: 0.001
iteration: 9 loss: 0.1964 lr: 0.001
iteration: 10 loss: 0.1220 lr: 0.001


It produces nothing of interest scientifically.
"""

task='TEST_deterministic' # Enter the name of your experiment Task
scorer='Alex' # Enter the name of the experimenter/labeler


import os,  subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

print("Imported DLC!")
basepath=os.path.dirname(os.path.abspath('testscript.py'))
videoname='reachingvideo1'
video=[os.path.join(basepath,'Reaching-Mackenzie-2018-08-30','videos',videoname+'.avi')]

#to test destination folder:
#dfolder=basepath
dfolder=None

print("CREATING PROJECT")
path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True)

cfg=deeplabcut.auxiliaryfunctions.read_config(path_config_file)
cfg['numframes2pick']=5
cfg['pcutoff']=0.01
cfg['TrainingFraction']=[.8]
cfg['resnet']=152

deeplabcut.auxiliaryfunctions.write_config(path_config_file,cfg)

print("EXTRACTING FRAMES")
deeplabcut.extract_frames(path_config_file,mode='automatic',userfeedback=False)

print("CREATING-SOME LABELS FOR THE FRAMES")
frames=os.listdir(os.path.join(cfg['project_path'],'labeled-data',videoname))
#As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
for index,bodypart in enumerate(cfg['bodyparts']):
        columnindex = pd.MultiIndex.from_product([[scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(100+np.ones((len(frames),2))*50*index, columns = columnindex, index = [os.path.join('labeled-data',videoname,fn) for fn in frames])
        if index==0:
            dataFrame=frame
        else:
            dataFrame = pd.concat([dataFrame, frame],axis=1)

dataFrame.to_csv(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + scorer + ".csv"))
dataFrame.to_hdf(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

print("Plot labels...")

deeplabcut.check_labels(path_config_file)

print("CREATING TRAININGSET")
deeplabcut.create_training_dataset(path_config_file)

posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(1),'train/pose_cfg.yaml')

DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=10
DLC_config['display_iters']=1
DLC_config['multi_step']=[[0.001,10]]

print("CHANGING training parameters to end quickly!")
deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)

print("TRAIN again... different loss?")
deeplabcut.train_network(path_config_file)


DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['dataset_type']='deterministic'
DLC_config['deterministic']=True
deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)

print("TRAIN again... the same losses!")
deeplabcut.train_network(path_config_file)

print("ALL DONE!!! - default cases are functional.")
