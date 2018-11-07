#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

This script tests various functionalities in an automatic way (mostly for developers).
It does not require ffmpeg in terminal and also tests kmeans frame extraction.
It should take about 4 minutes to run this in a CPU. 

It produces nothing of interesting scientifically. 
"""

task='TEST2' # Enter the name of your experiment Task
scorer='Alex' # Enter the name of the experimenter/labeler

import deeplabcut, os, yaml, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

def read_config(configname):
    """    Reads config file     """
    with open(str(configname), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return(cfg)

def write_config(configname,cfg):
    with open(str(configname), 'w') as ymlfile:
                yaml.dump(cfg, ymlfile,default_flow_style=False)

print("Imported DLC!")
basepath=os.path.dirname(os.path.abspath('testscript.py'))
videoname='reachingvideo1'
video=[os.path.join(basepath,'Reaching-Mackenzie-2018-08-30/videos/'+videoname+'.avi')]

print("CREATING PROJECT")
path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True)
cfg=read_config(path_config_file)
cfg['numframes2pick']=5
cfg['pcutoff']=0.01
cfg['TrainingFraction']=[.8]

write_config(path_config_file,cfg)

print("EXTRACTING FRAMES")
deeplabcut.extract_frames(path_config_file,mode='automatic',algo='kmeans')

print("CREATING-SOME LABELS FOR THE FRAMES")
frames=os.listdir(os.path.join(cfg['project_path'],'labeled-data',videoname))
#As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
for index,bodypart in enumerate(cfg['bodyparts']): 
        columnindex = pd.MultiIndex.from_product([[scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(np.ones((len(frames),2))*50*index, columns = columnindex, index = [os.path.join('labeled-data',videoname,fn) for fn in frames])
        if index==0:
            dataFrame=frame
        else:
            dataFrame = pd.concat([dataFrame, frame],axis=1)

dataFrame.to_csv(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + scorer + ".csv"))
dataFrame.to_hdf(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

print("CREATING TRAININGSET")
deeplabcut.create_training_dataset(path_config_file)

posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(1),'train/pose_cfg.yaml')
DLC_config=read_config(posefile)
DLC_config['save_iters']=10
DLC_config['display_iters']=2
DLC_config['multi_step']=[[0.001,10]]

print("CHANGING training parameters to end quickly!")
write_config(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)

#this is much easier now: deeplabcut.train_network(path_config_file,gputouse=0,max_snapshots_to_keep=None,saveiters=1)

print("EVALUATE")
deeplabcut.evaluate_network(path_config_file,plotting=True)

print("CUT SHORT VIDEO AND ANALYZE")

# Make super short video (so the analysis is quick!)
vname='brief'
newvideo=os.path.join(cfg['project_path'],'videos',vname+'.avi')

subprocess.call(['ffmpeg','-i',video[0],'-ss','00:00:00','-to','00:00:00.4','-c','copy',newvideo])
deeplabcut.analyze_videos(path_config_file,[newvideo])

print("CREATE VIDEO")
deeplabcut.create_labeled_video(path_config_file,[newvideo])


print("EXTRACT OUTLIERS")
deeplabcut.extract_outlier_frames(path_config_file,[newvideo],outlieralgorithm='jump',epsilon=0,automatic=True)

file=os.path.join(cfg['project_path'],'labeled-data',vname,"machinelabels-iter"+ str(cfg['iteration']) + '.h5')

print("RELABELING")
DF=pd.read_hdf(file,'df_with_missing')
DLCscorer=np.unique(DF.columns.get_level_values(0))[0]
DF.columns.set_levels([scorer.replace(DLCscorer,scorer)],level=0,inplace=True)
DF =DF.drop('likelihood',axis=1,level=2)
DF.to_csv(os.path.join(cfg['project_path'],'labeled-data',vname,"CollectedData_" + scorer + ".csv"))
DF.to_hdf(os.path.join(cfg['project_path'],'labeled-data',vname,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

print("MERGING")
deeplabcut.merge_datasets(path_config_file)

print("CREATING TRAININGSET")
deeplabcut.create_training_dataset(path_config_file)

cfg=read_config(path_config_file)
posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(1),'train/pose_cfg.yaml')
DLC_config=read_config(posefile)
DLC_config['save_iters']=5
DLC_config['display_iters']=1
DLC_config['multi_step']=[[0.05,5]]

print("CHANGING training parameters to end quickly!")
write_config(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)


print("ALL DONE!!! - default cases are functional.")
