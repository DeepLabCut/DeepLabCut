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
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")
# Dependencies for video:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage.draw import circle_perimeter, circle
from VideoProcessor import VideoProcessorSK as vp

####################################################
# Loading descriptors of model
####################################################

from myconfig_analysis import videofolder, cropping, Task, date, \
    resnet, shuffle, trainingsiterations, pcutoff, deleteindividualframes, x1, x2, y1, y2, videotype, alphavalue, dotsize, colormap
from myconfig_analysis import scorer as humanscorer

# loading meta data / i.e. training & test files
#basefolder = os.path.join('..','pose-tensorflow','models')
#datafolder = os.path.join(basefolder , "UnaugmentedDataSet_" + Task + date)
#Data = pd.read_hdf(os.path.join(datafolder , 'data-' + Task , 'CollectedData_' + humanscorer + '.h5'),'df_with_missing')

# Name for scorer based on passed on parameters from myconfig_analysis. Make sure they refer to the network of interest.
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)

####################################################
# Auxiliary function
####################################################

def CreateVideo(clip,Dataframe):
        ''' Creating individual frames with labeled body parts and making a video''' 
        scorer=np.unique(Dataframe.columns.get_level_values(0))[0]
        bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
        colorclass=plt.cm.ScalarMappable(cmap=colormap)
        C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts2plot)))
        colors=(C[:,:3]*255).astype(np.uint8)
        ny,nx,fps = clip.height(),clip.width(), clip.fps()
        nframes = len(Dataframe.index)
        duration = nframes/fps
        
        print("Duration of video [s]: ", duration, ", recorded with ", fps,"fps!")
        print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",ny,nx)
        
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
        clip =vp(video)
        print("Loading ", video, "and data.")
        dataname = video.split('.')[0] + scorer + '.h5'
        try:
            Dataframe = pd.read_hdf(dataname)
            clip = vp(fname = video,sname = os.path.join(vname + '_DeepLabCutlabeled.mp4'))
            CreateVideo(clip,Dataframe)
        except FileNotFoundError:
            datanames=[fn for fn in os.listdir(os.curdir) if (vname in fn) and (".h5" in fn) and "resnet" in fn]
            if len(datanames)==0:
                print("The video was not analyzed with this scorer:", scorer)
                print("No other scorers were found, please run AnalysisVideos.py first.")
            elif len(datanames)>0:
                print("The video was not analyzed with this scorer:", scorer)
                print("Other scorers were found, however:", datanames)
                print("Creating labeled video for:", datanames[0]," instead.")

                Dataframe = pd.read_hdf(datanames[0])
                clip = vp(fname = video,sname = os.path.join(vname + '_DeepLabCutlabeled.mp4'))
                CreateVideo(clip,Dataframe)