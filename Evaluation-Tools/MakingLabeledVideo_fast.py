"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

Edited by:
Hao Wu, hwu01@g.harvard.edu

This script labels the bodyparts in videos as analzyed by "AnalyzeVideos.py". 
This code does not store any frames and substantially faster! 

python3 MakingLabeledVideo_fast.py

Videos are not inteded to be created in the docker container (due to missing ffmpeg bindings)
You can find the directory by: "find / | grep ffmpeg" and then setting it. 
"""

####################################################
# Dependencies
####################################################
import os.path
import sys
subfolder = os.getcwd().split('Evaluation-Tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")
# Dependencies for video:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage.draw import circle_perimeter, circle

from VideoProcessor import VideoProcessorSK as vp
import auxiliaryfunctions

from myconfig_analysis import videofolder, cropping, scorer, Task, date, \
    resnet, shuffle, trainingsiterations, pcutoff, deleteindividualframes, x1, x2, y1, y2, videotype, alphavalue, dotsize, colormap

# loading meta data / i.e. training & test files
basefolder = os.path.join('..','pose-tensorflow','models')

datafolder = os.path.join(basefolder , "UnaugmentedDataSet_" + Task + date)
Data = pd.read_hdf(os.path.join(datafolder , 'data-' + Task , 'CollectedData_' + scorer + '.h5'),
    'df_with_missing')

bodyparts2plot = list(np.unique(Data.columns.get_level_values(1)))
# creating colors:
colorclass=plt.cm.ScalarMappable(cmap=colormap)
C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts2plot)))
colors=(C[:,:3]*255).astype(np.uint8)

####################################################
# Loading descriptors of model
####################################################

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)


##################################################
# Datafolder
##################################################
os.chdir(videofolder)
videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn) and ("labeled" not in fn)])
print("Starting ", videofolder, videos)

for video in videos:
    
    vname = video.split('.')[0]
    if os.path.isfile(os.path.join(vname + '_DeepLabCutlabeled.mp4')):
        print("Labeled video already created.")
    else:
        clip =VideoFileClip(video) #vp(video)
        print("Loading ", video, "and data.")
        dataname = video.split('.')[0] + scorer + '.h5'
        try:
            Dataframe = pd.read_hdf(dataname)
            clip = vp(fname = video,sname = os.path.join(vname + '_DeepLabCutlabeled.mp4'))
        except FileNotFoundError:
            print("Data was not analyzed (run AnalysisVideos.py first).")

        ny = clip.height()
        nx = clip.width()
        fps = clip.fps()
        nframes = clip.frame_count()
        duration = nframes/fps
        print("Duration of video [s]: ", duration, ", recorded with ", fps,
              "fps!")
        print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",
              ny,nx)
    
        print("Generating frames")
        for index in tqdm(range(nframes)):
            image = clip.load_frame()
            for bpindex, bp in enumerate(bodyparts2plot):
                if Dataframe[scorer][bp]['likelihood'].values[index] > pcutoff:
                    xc = int(Dataframe[scorer][bp]['x'].values[index])
                    yc = int(Dataframe[scorer][bp]['y'].values[index])
                    #rr, cc = circle_perimeter(yc,xc,radius)
                    rr, cc = circle(yc,xc,dotsize,shape=(ny,nx))
                    image[rr, cc, :] = colors[bpindex]

            frame = image
            clip.save_frame(frame)

        clip.close()
