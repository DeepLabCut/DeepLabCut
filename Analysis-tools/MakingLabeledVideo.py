"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script labels the bodyparts in videos as analzyed by "AnalyzeVideos.py". This code is relatively slow as 
it stores all individual frames. Use  MakingLabeledVideo_fast.py instead for faster (and slightly different) version (frames are not stored).

python3 MakingLabeledVideo.py

Note: run python3 AnalyzeVideos.py first.
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

# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name=colormap):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def CreateVideo(clip,Dataframe):
    ''' Creating individual frames with labeled body parts and making a video''' 
    scorer=np.unique(Dataframe.columns.get_level_values(0))[0]
    bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    colors = get_cmap(len(bodyparts2plot))

    ny, nx = clip.size  # dimensions of frame (height, width)
    fps = clip.fps
    nframes = len(Dataframe.index)
    if cropping:
        # one might want to adjust
        clip = clip.crop(y1=y1, y2=y2, x1=x1, x2=x2)
    clip.reader.initialize()
    print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
          "fps!")
    print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",
          clip.size)
    print("Generating frames")
    for index in tqdm(range(nframes)):
        
        imagename = tmpfolder + "/file%04d.png" % index
        if os.path.isfile(tmpfolder + "/file%04d.png" % index):
            pass
        else:
            plt.axis('off')
            image = img_as_ubyte(clip.reader.read_frame())
            #image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))

            if np.ndim(image) > 2:
                h, w, nc = np.shape(image)
            else:
                h, w = np.shape(image)

            plt.figure(frameon=False, figsize=(w * 1. / 100, h * 1. / 100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.imshow(image)

            for bpindex, bp in enumerate(bodyparts2plot):
                if Dataframe[scorer][bp]['likelihood'].values[index] > pcutoff:
                    plt.scatter(
                        Dataframe[scorer][bp]['x'].values[index],
                        Dataframe[scorer][bp]['y'].values[index],
                        s=dotsize**2,
                        color=colors(bpindex),
                        alpha=alphavalue)

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(imagename)

            plt.close("all")

    os.chdir(tmpfolder)

    print("Generating video")
    subprocess.call([
        'ffmpeg', '-framerate',
        str(clip.fps), '-i', 'file%04d.png', '-r', '30', '../'+vname + '_DeepLabCutlabeled.mp4'])
    if deleteindividualframes:
        for file_name in glob.glob("*.png"):
            os.remove(file_name)

    os.chdir("../")

##################################################
# Datafolder
##################################################
# videofolder='../videos/' #where your folder with videos is.

os.chdir(videofolder)
videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn) and ("labeled" not in fn)])

print("Starting ", videofolder, videos)
for video in videos:
    vname = video.split('.')[0]
    tmpfolder = 'temp' + vname
    auxiliaryfunctions.attempttomakefolder(tmpfolder)
    
    if os.path.isfile(os.path.join(tmpfolder, vname + '_DeepLabCutlabeled.mp4')):
        print("Labeled video already created.")
    else:
        print("Loading ", video, "and data.")
        dataname = video.split('.')[0] + scorer + '.h5'
        try: # to load data for this video + scorer
            Dataframe = pd.read_hdf(dataname)
            clip = VideoFileClip(video)
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
                clip = VideoFileClip(video)
                CreateVideo(clip,Dataframe)