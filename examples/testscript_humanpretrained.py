'''
Testscript human network

'''

import os,  subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np


Task = 'human_dancing'
YourName = 'teamDLC'

basepath=os.path.dirname(os.path.abspath('testscript.py'))
videoname='reachingvideo1'
video=[os.path.join(basepath,'Reaching-Mackenzie-2018-08-30','videos',videoname+'.avi')]


configfile, path_train_config=deeplabcut.create_pretrained_human_project(Task, YourName,video, videotype='avi', analyzevideo=False, createlabeledvideo=False, copy_videos=False) #must leave copy_videos=True

lastvalue=5
DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(path_train_config)
pretrainedDeeperCutweights=DLC_config['init_weights']

print("EXTRACTING FRAMES")
deeplabcut.extract_frames(configfile,mode='automatic',userfeedback=False)

print("CREATING-SOME LABELS FOR THE FRAMES")
cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
frames=os.listdir(os.path.join(cfg['project_path'],'labeled-data',videoname))
#As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
for index,bodypart in enumerate(cfg['bodyparts']):
        columnindex = pd.MultiIndex.from_product([[cfg['scorer']], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(100+np.ones((len(frames),2))*50*index, columns = columnindex, index = [os.path.join('labeled-data',videoname,fn) for fn in frames])
        if index==0:
            dataFrame=frame
        else:
            dataFrame = pd.concat([dataFrame, frame],axis=1)

dataFrame.to_csv(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + cfg['scorer'] + ".csv"))
dataFrame.to_hdf(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + cfg['scorer'] + '.h5'),'df_with_missing',format='table', mode='w')

deeplabcut.create_training_dataset(configfile,Shuffles=[1])
DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(path_train_config)
DLC_config['save_iters']=lastvalue
DLC_config['display_iters']=1
DLC_config['multi_step']=[[0.001,lastvalue]]
DLC_config['init_weights']=pretrainedDeeperCutweights.split('.index')[0]
deeplabcut.auxiliaryfunctions.write_plainconfig(path_train_config,DLC_config)

deeplabcut.train_network(configfile,shuffle=1)

print("Adding bodypart!")
cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
cfg['bodyparts'] = ['ankle1','knee1','hip1','hip2','knee2','ankle2','wrist1','elbow1','shoulder1','shoulder2','elbow2','wrist2','chin','forehead','plus1more']
deeplabcut.auxiliaryfunctions.write_config(configfile,cfg)

print("CREATING-SOME More LABELS FOR THE FRAMES (including the new bodypart!)")
cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
frames=[f for f in os.listdir(os.path.join(cfg['project_path'],'labeled-data',videoname)) if '.png' in f]

#As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
for index,bodypart in enumerate(cfg['bodyparts']):
        columnindex = pd.MultiIndex.from_product([[cfg['scorer']], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(100+np.ones((len(frames),2))*50*index, columns = columnindex, index = [os.path.join('labeled-data',videoname,fn) for fn in frames])
        if index==0:
            dataFrame=frame
        else:
            dataFrame = pd.concat([dataFrame, frame],axis=1)

dataFrame.to_csv(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + cfg['scorer'] + ".csv"))
dataFrame.to_hdf(os.path.join(cfg['project_path'],'labeled-data',videoname,"CollectedData_" + cfg['scorer'] + '.h5'),'df_with_missing',format='table', mode='w')


deeplabcut.create_training_dataset(configfile,Shuffles=[1])
DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(path_train_config)
DLC_config['save_iters']=lastvalue
DLC_config['display_iters']=1
DLC_config['multi_step']=[[0.001,lastvalue]]
DLC_config['init_weights']=pretrainedDeeperCutweights.split('.index')[0]
deeplabcut.auxiliaryfunctions.write_plainconfig(path_train_config,DLC_config)

#deeplabcut.train_network(configfile,shuffle=1) #>> fails one body part too much!
deeplabcut.train_network(configfile,shuffle=1,keepdeconvweights=False)
