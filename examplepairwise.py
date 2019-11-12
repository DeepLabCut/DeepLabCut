#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:51:45 2019

@author: alex
"""
import os, yaml, deeplabcut
from pathlib import Path
import sys
import os
from pathlib import Path

# Loading example data set [supressed!]
#path_config_file = os.path.join(os.getcwd(),'Reaching-Mackenzie-2018-08-30/config.yaml')

base='/home/alex/Hacking/DLCreleases/DLC4alphatesters/DeepLabCutpairwise/examples/Reaching-Mackenzie-2018-08-30/'
config=os.path.join(base,'config.yaml')

#deeplabcut.load_demo_data(config)

##########################################
######### STEP 1 -- collect pairwise stats:
##########################################

#getting paths!
trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(config,shuffle=1,trainingsetindex=0)

cfg=deeplabcut.utils.read_plainconfig(trainposeconfigfile)
#setting path:
cfg['pairwise_stats_fn']=str(os.path.join(Path(trainposeconfigfile).parent,"pwstats.mat"))
cfg['pairwise_stats_collect']=True
cfg['pairwise_predict']=False #otherwise they are loaded, but don't exist yet...
cfg['dataset_type']='pairwise'
#for pairwise stats collection we need scale 1!
cfg['global_scale']=1.
cfg['scale_jitter_lo']=1.
cfg['scale_jitter_up']=1.

cfg['set_pairwise_stats_collect']=True
deeplabcut.utils.write_plainconfig(trainposeconfigfile,cfg)

deeplabcut.pairwise_stats(trainposeconfigfile)


######################################
########## SWITCH STATES >> to prepare for training
#####################################
cfg=deeplabcut.utils.read_plainconfig(trainposeconfigfile)
cfg['location_refinement']=True

cfg['pairwise_predict']=True #True
cfg['pairwise_loss_weight']=.1 #relative weight of loss

########### NOW CHANGE PARAMETERS TO YOUR USUAL training params
cfg['global_scale']=1.
cfg['scale_jitter_lo']=.25
cfg['scale_jitter_up']=1.1
cfg['dataset_type']='pairwise'
cfg['pairwise_stats_collect']=False
cfg['net_type']='resnet_50'
deeplabcut.utils.write_plainconfig(trainposeconfigfile,cfg)

deeplabcut.train_network(config,shuffle=1,maxiters=1001,saveiters=50,displayiters=5)

######################################
########## Evaluate
#####################################

cfg=deeplabcut.utils.read_plainconfig(testposeconfigfile)
cfg['pairwise_predict']=True #True
deeplabcut.utils.write_plainconfig(testposeconfigfile,cfg)
deeplabcut.evaluate_network(config,Shuffles=[1]) #,plotting=True)


import matplotlib.pyplot as plt
import numpy as np
import pickle


##########################
####@change that path:
#############
resultsfilename='/home/alex/Hacking/DLCreleases/DLC4alphatesters/DeepLabCutpairwise/examples/Reaching-Mackenzie-2018-08-30/evaluation-results/iteration-0/ReachingAug30-trainset95shuffle1/DLC_resnet50_ReachingAug30shuffle1_1001-snapshot-1001.h5'
with open(resultsfilename.split('.h5')[0] + 'pairwise.pickle', 'rb') as f:
            data=pickle.load(f)

fn=22

im=deeplabcut.auxfun_videos.imread(os.path.join(base,data[fn]['name']))

plt.imshow(im)

# x & y bodypart detections:
x=data[fn]['pose'][0::3]
y=data[fn]['pose'][1::3]

xpw=data[fn]['pws'][:,:,:,0]
ypw=data[fn]['pws'][:,:,:,1]

#plot bodyparts
plt.plot(x,y,'r.')

#and vector predictions!
numbodyparts=len(x)
for base in range(numbodyparts):
    for i in range(numbodyparts):
        plt.plot([x[base],x[base]+xpw[0,base,i]],
                    [y[base],y[base]+ypw[0,base,i]])

plt.show()
