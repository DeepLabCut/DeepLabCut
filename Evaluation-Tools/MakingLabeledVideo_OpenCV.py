"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

Edited by:
Hao Wu, hwu01@g.harvard.edu

This script labels the bodyparts in videos as analzyed by "AnalyzeVideos.py". This code is relatively slow as 
it stores all individual frames. Should be reworked at some point. Contributions are welcome. 
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
from matplotlib import cm
import imageio
imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import auxiliaryfunctions
from myconfig_analysis import videofolder, cropping, scorer, Task, date, \
    resnet, shuffle, trainingsiterations, pcutoff, deleteindividualframes, x1, x2, y1, y2, videotype, alphavalue
from VideoProcessor import VideoProcessorCV as vp

# loading meta data / i.e. training & test files
basefolder = os.path.join('..','pose-tensorflow','models')

datafolder = os.path.join(basefolder , "UnaugmentedDataSet_" + Task + date)
Data = pd.read_hdf(os.path.join(datafolder , 'data-' + Task , 'CollectedData_' + scorer + '.h5'),
    'df_with_missing')

bodyparts2plot = list(np.unique(Data.columns.get_level_values(1)))


# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


####################################################
# Loading descriptors of model
####################################################

colors = get_cmap(len(bodyparts2plot))

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
# scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
#     date) + 'shuffle' + str(shuffle) + '_' + str(13000)

##################################################
# Datafolder
##################################################
# videofolder='../videos/' #where your folder with videos is.

os.chdir(videofolder)

videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
print("Starting ", videofolder, videos)
for video in videos:
    vname = video.split('.')[0]
    tmpfolder = 'temp' + vname
    
    auxiliaryfunctions.attempttomakefolder(tmpfolder)
    if os.path.isfile(os.path.join(tmpfolder, vname + '.avi')):
        print("Labeled video already created.")
    else:
        print("Loading ", video, "and data.")
        dataname = video.split('.')[0] + scorer + '.h5'
        try:
            Dataframe = pd.read_hdf(dataname)
            clip = vp(fname = video,sname = 'labeled_' + vname + '.avi')
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

                    xc = Dataframe[scorer][bp]['x'].values[index]
                    yc = Dataframe[scorer][bp]['y'].values[index]
                    d = 3
                    xl = int(max(0,xc- d))
                    yl = int(max(0,yc -d))
                    xr = int(min(nx,xc + d))
                    yr = int(min(ny,yc + d))

                    image[yl:yr,xl:xr,bpindex] = (image[yl:yr,xl:xr,bpindex]*(1-alphavalue)).astype(np.uint8)


                    crgb = colors(bpindex)

                    for cind in range(3):
                        image[yl:yr,xl:xr,cind] += int(253.0 * alphavalue * crgb[cind] * crgb[3])

            frame = image
            clip.save_frame(frame)

        clip.close()