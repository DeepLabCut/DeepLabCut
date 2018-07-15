"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script analyzes videos based on a trained network (as specified in myconfig_analysis.py)

You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

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
    trainingsFraction, resnet, snapshotindex, shuffle,x1, x2, y1, y2, videotype, storedata_as_csv

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

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


def getpose(image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


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
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
pdindex = pd.MultiIndex.from_product(
    [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
    names=['scorer', 'bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

# videofolder='../videos/' #where your folder with videos is.
frame_buffer = 10

os.chdir(videofolder)
videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
print("Starting ", videofolder, videos)
for video in videos:
    dataname = video.split('.')[0] + scorer + '.h5'
    try:
        # Attempt to load data...
        pd.read_hdf(dataname)
        print("Video already analyzed!", dataname)
    except FileNotFoundError:
        print("Loading ", video)
        clip = VideoFileClip(video)
        ny, nx = clip.size  # dimensions of frame (height, width)
        fps = clip.fps
        #nframes = np.sum(1 for j in clip.iter_frames()) #this is slow (but accurate)
        nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)
        # this will overestimage number of frames (see https://github.com/AlexEMG/DeepLabCut/issues/9) This is especially a problem
        # for high frame rates and long durations due to rounding errors (as Rich Warren found). Later we crop the result (line 187)
        
        if cropping:
            clip = clip.crop(
                y1=y1, y2=y2, x1=x1, x2=x2)  # one might want to adjust

        print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
              "fps!")
        print("Overall # of frames: ", nframes_approx,"with cropped frame dimensions: ", clip.size)

        start = time.time()
        PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))
        clip.reader.initialize()
        print("Starting to extract posture")
        for index in tqdm(range(nframes_approx)):
            #image = img_as_ubyte(clip.get_frame(index * 1. / fps))
            image = img_as_ubyte(clip.reader.read_frame())
            # Thanks to Rick Warren for the  following snipplet:
            # if close to end of video, start checking whether two adjacent frames are identical
            # this should only happen when moviepy has reached the final frame
            # if two adjacent frames are identical, terminate the loop
            if index==int(nframes_approx-frame_buffer*2):
                last_image = image
            elif index>int(nframes_approx-frame_buffer*2):
                if (image==last_image).all():
                    nframes = index
                    print("Detected frames: ", nframes)
                    break
                else:
                    last_image = image
            pose = getpose(image, cfg, outputs)
            PredicteData[index, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!

        stop = time.time()

        dictionary = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": scorer,
            "config file": cfg,
            "fps": fps,
            "frame_dimensions": (ny, nx),
            "nframes": nframes
        }
        metadata = {'data': dictionary}

        print("Saving results...")
        DataMachine = pd.DataFrame(PredicteData[:nframes,:], columns=pdindex, index=range(nframes)) #slice pose data to have same # as # of frames.
        DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
        
        if storedata_as_csv:
            DataMachine.to_csv(video.split('.')[0] + scorer+'.csv')
        
        with open(dataname.split('.')[0] + 'includingmetadata.pickle',
                  'wb') as f:
            pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
