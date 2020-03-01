#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

It should take about 3:30 minutes to run this in a CPU.
It should take about 1:30 minutes on a GPU (incl. downloading the ResNet weights)

It produces nothing of interest scientifically.
"""

task='TEST' # Enter the name of your experiment Task
scorer='Alex' # Enter the name of the experimenter/labeler

import os,  subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

print("Imported DLC!")
basepath=os.path.dirname(os.path.abspath('testscript.py'))
videoname='reachingvideo1'

video=[os.path.join(basepath,'../examples/Reaching-Mackenzie-2018-08-30','videos',videoname+'.avi')]

# For testing a color video:
#videoname='baby4hin2min'
#video=[os.path.join('/home/alex/Desktop/Data',videoname+'.mp4')]
#to test destination folder:
#dfolder=basepath

dfolder=None
net_type='resnet_50' #'mobilenet_v2_0.35' #'resnet_50'
augmenter_type='default' #'tensorpack'
augmenter_type2='imgaug'
augmenter_type3='tensorpack'
numiter=5

print("CREATING PROJECT")
path_config_file=deeplabcut.create_new_project(task,scorer,video,copy_videos=True)

cfg=deeplabcut.auxiliaryfunctions.read_config(path_config_file)
cfg['numframes2pick']=5
cfg['pcutoff']=0.01
cfg['TrainingFraction']=[.8]
cfg['skeleton']=[['bodypart1','bodypart2'],['bodypart1','bodypart3']]

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
deeplabcut.create_training_dataset(path_config_file,net_type=net_type,augmenter_type=augmenter_type)

posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(1),'train/pose_cfg.yaml')

DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=numiter
DLC_config['display_iters']=2
DLC_config['multi_step']=[[0.001,numiter]]

print("CHANGING training parameters to end quickly!")
deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)

print("EVALUATE")
deeplabcut.evaluate_network(path_config_file,plotting=True)
#deeplabcut.evaluate_network(path_config_file,plotting=True,trainingsetindex=33)
print("CUT SHORT VIDEO AND ANALYZE (with dynamic cropping!)")

# Make super short video (so the analysis is quick!)

try: #you need ffmpeg command line interface
    #subprocess.call(['ffmpeg','-i',video[0],'-ss','00:00:00','-to','00:00:00.4','-c','copy',newvideo])
    newvideo=deeplabcut.ShortenVideo(video[0],start='00:00:00',stop='00:00:00.4',outsuffix='short',outpath=os.path.join(cfg['project_path'],'videos'))
    vname=Path(newvideo).stem
except: # if ffmpeg is broken
    vname='brief'
    newvideo=os.path.join(cfg['project_path'],'videos',vname+'.mp4')
    from moviepy.editor import VideoFileClip,VideoClip
    clip = VideoFileClip(video[0])
    clip.reader.initialize()
    def make_frame(t):
        return clip.get_frame(1)

    newclip = VideoClip(make_frame, duration=1)
    newclip.write_videofile(newvideo,fps=30)


deeplabcut.analyze_videos(path_config_file,[newvideo],save_as_csv=True, destfolder=dfolder, dynamic=(True,.1,5))

#print("CREATE VIDEO")
#deeplabcut.create_labeled_video(path_config_file,[newvideo], destfolder=dfolder,save_frames=True)

print("Making plots")
deeplabcut.plot_trajectories(path_config_file, [newvideo], destfolder=dfolder)



##############################
########## PREFIX TEST!
##############################
import shutil
projectpath=Path(path_config_file).parents[0]
modelprefix='prefixtest'
print("Copying datasets")
try:
    shutil.copytree(os.path.join(projectpath,'dlc-models'),os.path.join(projectpath,modelprefix,'dlc-models'))
except FileExistsError:
    print("Folder exists already...")

trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,modelprefix=modelprefix)

DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
DLC_config['save_iters']=numiter
DLC_config['display_iters']=2
DLC_config['multi_step']=[[0.001,numiter]]
print("CHANGING training parameters to end quickly!")
deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file,modelprefix=modelprefix)

print("EVALUATE")
deeplabcut.evaluate_network(path_config_file,plotting=True,modelprefix=modelprefix)

destfolder=os.path.join(projectpath,modelprefix)
print("ANALYZE")
deeplabcut.analyze_videos(path_config_file,[newvideo],save_as_csv=True, destfolder=destfolder, dynamic=(True,.1,5),modelprefix=modelprefix)

print("Making plots")
deeplabcut.plot_trajectories(path_config_file, [newvideo], destfolder=destfolder,modelprefix=modelprefix)
