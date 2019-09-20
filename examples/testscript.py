#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

It should take about 4:00 minutes to run this in a CPU.
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
deeplabcut.create_training_dataset(path_config_file)

posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(1),'train/pose_cfg.yaml')

DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=10
DLC_config['display_iters']=2
DLC_config['multi_step']=[[0.001,10]]

print("CHANGING training parameters to end quickly!")
deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)

print("EVALUATE")
deeplabcut.evaluate_network(path_config_file,plotting=True)

print("CUT SHORT VIDEO AND ANALYZE")

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

deeplabcut.analyze_videos(path_config_file,[newvideo],save_as_csv=True, destfolder=dfolder)

print("CREATE VIDEO")
deeplabcut.create_labeled_video(path_config_file,[newvideo], destfolder=dfolder)

print("Making plots")
deeplabcut.plot_trajectories(path_config_file,[newvideo], destfolder=dfolder)


print("EXTRACT OUTLIERS")
deeplabcut.extract_outlier_frames(path_config_file,[newvideo],outlieralgorithm='jump',epsilon=0,automatic=True, destfolder=dfolder)
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

cfg=deeplabcut.auxiliaryfunctions.read_config(path_config_file)
posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(1),'train/pose_cfg.yaml')
DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=5
DLC_config['display_iters']=1
DLC_config['multi_step']=[[0.001,5]]

print("CHANGING training parameters to end quickly!")
deeplabcut.auxiliaryfunctions.write_config(posefile,DLC_config)

print("TRAIN")
deeplabcut.train_network(path_config_file)

print("Inference with new direct cropping")
deeplabcut.analyze_videos(path_config_file,[newvideo],destfolder=dfolder,cropping=[0,50,0,50],save_as_csv=True)

print("Extracting skeleton distances, filter and plot filtered output")
deeplabcut.analyzeskeleton(path_config_file, [newvideo], save_as_csv=True, destfolder=dfolder)

deeplabcut.filterpredictions(path_config_file,[newvideo])


#deeplabcut.create_labeled_video(path_config_file,[newvideo], destfolder=dfolder,filtered=True)
deeplabcut.create_labeled_video(path_config_file,[newvideo], destfolder=dfolder,displaycropped=True,filtered=True)

deeplabcut.plot_trajectories(path_config_file,[newvideo], destfolder=dfolder,filtered=True)

print("CREATING TRAININGSET for shuffle 2")
print("will be used for 3D testscript...")
deeplabcut.create_training_dataset(path_config_file,Shuffles=[2])

posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(2),'train/pose_cfg.yaml')

DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=10
DLC_config['display_iters']=2
DLC_config['multi_step']=[[0.001,10]]

print("CHANGING training parameters to end quickly!")
deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

print("TRAINING shuffle 2")
deeplabcut.train_network(path_config_file,shuffle=2)

print("ANALYZING some individual frames")
deeplabcut.analyze_time_lapse_frames(path_config_file,os.path.join(cfg['project_path'],'labeled-data/reachingvideo1/'))



print("ALL DONE!!! - default cases are functional.")
print("Re-import DLC with env. variable set to test DLC light mode.")
os.environ['DLClight']='True'
subprocess.call(['python3',"-c","import deeplabcut"])
