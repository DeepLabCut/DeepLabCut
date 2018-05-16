"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script analyzes videos based on a trained network.
You need tensorflow for evaluation. Run by:
    
CUDA_VISIBLE_DEVICES=0 python3 AnalyzeABunchofPictures.py

This script loops over a folder containing tiff stacks and analyses them stack by stack. By popular demand of the Woolf lab at HMS.
This can also be adapted to load different types of images. 
"""

####################################################
# Dependencies
####################################################

import os.path
import sys

subfolder = os.getcwd().split('Evaluation-Tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow/")
sys.path.append(subfolder + "Generating_a_Training_Set")

from myconfig_analysis import videofolder, cropping, Task, date, \
    trainingsFraction, resnet, snapshotindex, shuffle

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

# Dependencies for video:
import pickle
# import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_ubyte
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

basefolder = '../pose-tensorflow/models/'  # for cfg file & ckpt!
modelfolder = (basefolder + Task + str(date) + '-trainset' +
               str(int(trainingsFraction * 100)) + 'shuffle' + str(shuffle))
cfg = load_config(modelfolder + '/test/' + "pose_cfg.yaml")

##################################################
# Load and setup CNN part detector
##################################################

# Check which snap shots are available and sort them by # iterations
Snapshots = np.array([
    fn.split('.')[0]
    for fn in os.listdir(modelfolder + '/train/')
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
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

# Name for scorer:
trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
pdindex = pd.MultiIndex.from_product(
    [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
    names=['scorer', 'bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

#Folder where your tiffstacks are:
os.chdir(videofolder)
videos = np.sort([fn for fn in os.listdir(os.curdir) if (".tiff" in fn)])

print("Starting ", videofolder, videos)
for tiffstack in videos:
    dataname = tiffstack.split('.')[0] + scorer + '.h5'
    try:
        # Attempt to load data...
        pd.read_hdf(dataname)
        print("tiffstack already analyzed!", dataname)
    except:
        print("Loading ", tiffstack)
        
        im = io.imread(tiffstack)
        nframes=np.shape(im)[0] # Assuming: numframes x width x height [otherwise consider changing this!]
        
        start = time.time()
        PredicteData = np.zeros((nframes, 3 * len(cfg['all_joints_names'])))

        print("Starting to extract posture")
        for index in tqdm(range(nframes)):
            image = img_as_ubyte(im[index])
            pose = getpose(image, cfg, outputs)
            PredicteData[index, :] = pose.flatten(
            )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!

        stop = time.time()

        dictionary = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": scorer,
            "config file": cfg,
            "nframes": nframes
        }
        metadata = {'data': dictionary}

        print("Saving results...")
        DataMachine = pd.DataFrame(
            PredicteData, columns=pdindex, index=range(nframes))
        DataMachine.to_hdf(
            dataname, 'df_with_missing', format='table', mode='w')
        with open(dataname.split('.')[0] + 'includingmetadata.pickle',
                  'wb') as f:
            pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
