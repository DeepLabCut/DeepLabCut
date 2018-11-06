"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
R Warren, raw2163@cumc.columbia.edu
M Mathis, mackenzie@post.harvard.edu

This script analyzes videos based on a trained network (as specified in myconfig_analysis.py)
You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

Updated to allow batchprocessing. See preprint on how to tune this optimally for your use-case:

set batchsize in myconfig_analysis.py

On the inference speed and video-compression robustness of DeepLabCut
by Alexander Mathis and Rick Warren
https://www.biorxiv.org/content/early/2018/10/30/457242
"""


####################################################
# Dependencies
####################################################

import os.path
import sys
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow")
sys.path.append(subfolder + "Generating_a_Training_Set")

from myconfig_analysis import videofolder, cropping, Task, date, \
    trainingsFraction, resnet, snapshotindex, shuffle,x1, x2, y1, y2, videotype, storedata_as_csv, batchsize

print("Starting evaluation")

# Deep-cut dependencies
from config import load_config
from nnet import predict

# Dependencies for video:
import pickle
# import matplotlib.pyplot as plt
import imageio
imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import skimage.color
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

####################################################
# Loading data, and defining model folder
####################################################

basefolder = os.path.join('..','pose-tensorflow','models')
modelfolder = os.path.join(basefolder, Task + str(date) + '-trainset' +
               str(int(trainingsFraction * 100)) + 'shuffle' + str(shuffle))

cfg = load_config(os.path.join(modelfolder , 'test' ,"pose_cfg.yaml"))

##################################################
# Load and setup CNN part detector
##################################################

# Check which snapshots are available and sort them by # iterations
Snapshots = np.array([
    fn.split('.')[0]
    for fn in os.listdir(os.path.join(modelfolder , 'train'))
    if "index" in fn
])
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print(modelfolder)
print(Snapshots)

##################################################
# Compute predictions over images
##################################################

# Check if data already was generated:
cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])

# Name for scorer:
trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)


cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])

pdindex = pd.MultiIndex.from_product(
    [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
    names=['scorer', 'bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

#Auxiliary functions:
def CrankVideo(cfg, sess, inputs, outputs,clip,nframes_approx,batchsize):
    PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at
    ny, nx = clip.size  # dimensions of frame (height, width)
    frames = np.empty((batchsize, nx, ny, 3), dtype='ubyte') # this keeps all frames in a batch
    for index in tqdm(range(nframes_approx)):
        image = img_as_ubyte(clip.get_frame(index*1./clip.fps)) #clip.reader.read_frame())        
        if index==int(nframes_approx-frame_buffer*2):
            last_image = image
        elif index>int(nframes_approx-frame_buffer*2):
            if (image==last_image).all():
                nframes = index
                print("Detected frames: ", nframes)
                if batch_ind>0:
                    pose = predict.getposeNP(frames, cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                    PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
                    
                    
                break
            else:
                last_image = image
        
        frames[batch_ind] = image
        if batch_ind==batchsize-1:
            pose = predict.getposeNP(frames,cfg, sess, inputs, outputs)
            PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
            batch_ind = 0
            batch_num += 1
        else:
            batch_ind+=1
    return PredicteData,nframes

def StoicVideo(cfg, sess, inputs, outputs,clip,nframes_approx):
    PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))
    for index in tqdm(range(nframes_approx)):
        image = img_as_ubyte(clip.get_frame(index*1./clip.fps))
        if index==int(nframes_approx-frame_buffer*2):
            last_image = image
        elif index>int(nframes_approx-frame_buffer*2):
            if (image==last_image).all():
                nframes = index
                print("Detected frames: ", nframes)
                break
            else:
                last_image = image
        pose = predict.getpose(image, cfg, sess, inputs, outputs)
        PredicteData[index, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
    
    return PredicteData,nframes

# Intitalize net & change batch size
cfg['batch_size']=batchsize
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
start_path=os.getcwd()

frame_buffer=10
# Change to video folder.
# videofolder='../videos/' #where your folder with videos is.
os.chdir(videofolder)
videos = np.random.permutation([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
for video in videos:
    vname=video.split('.')[0]
    dataname = vname+ scorer +'.h5'
    try:
        # Attempt to load data.
        pd.read_hdf(dataname)
        print("Video already analyzed!", dataname)
    except FileNotFoundError:
        print("Loading ", video)
        start0=time.time()
        clip = VideoFileClip(video)
        ny, nx = clip.size  # dimensions of frame (height, width)
        fps = clip.fps
        if cropping:
            print("Cropped video according to dimensions defined in config_analysis.py",x1,x2,y1,y2)
            #clip = clip.crop(y1=y1, y2=y2, x1=x1, x2=x2)  # one might want to adjust
            clip=clip.crop(y1=y1, y2=y2, x1=x1, x2=x2)  # one might want to adjust
        
        nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)
        print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
              "fps!")
        print("Overall # of frames: ", nframes_approx,"with cropped frame dimensions: ", clip.size)

        start = time.time()
        print("Starting to extract posture")
        if batchsize>1:
            PredicteData,nframes=CrankVideo(cfg, sess, inputs, outputs,clip, nframes_approx,batchsize)
        else:
            PredicteData,nframes=StoicVideo(cfg, sess, inputs, outputs,clip, nframes_approx)

        stop = time.time()

        dictionary = {
            "start-loading": start0,
            "start-inference": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": scorer,
            "config file": cfg,
            "fps": fps,
            "batch_size": batchsize,
            "frame_dimensions": (ny, nx),
            "nframes": nframes
        }
        metadata = {'data': dictionary}

        print("Saving results...")
        DataMachine = pd.DataFrame(PredicteData[:nframes,:], columns=pdindex, index=range(nframes)) #slice pose data to have same # as # of frames.
        DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
        
        if storedata_as_csv:
            DataMachine.to_csv(video.split('.')[0] + scorer+'.csv')
        
        print("Evaluation took", round(stop-start,2), "seconds.")
        with open(dataname.split('.h5')[0] + 'includingmetadata.pickle',
                  'wb') as f:
            pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
        
        # Delete clip + reader object https://github.com/Zulko/moviepy/issues/57
        # https://github.com/Zulko/moviepy/issues/518 (can slow down on Windows ...)
        clip.close()
        del clip
        #reader.close() 
        #del clip.reader

#return to start path.
os.chdir(str(start_path))