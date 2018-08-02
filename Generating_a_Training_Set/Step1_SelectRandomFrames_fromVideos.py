"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

A key point for a successful feature detector is to select diverse frames, which are typical for the behavior
you study that should be labeled. 

This helper script selects frames either uniformly sampled from a particular video (or folder) (selectionalgorithm=='uniform'). 
Note: this might not yield diverse frames, if the behavior is sparsely distributed (consider using kmeans), and/or select frames manually etc.

Also make sure to get select data from different (behavioral) sessions and different animals if those vary substantially (to train an invariant feature detector).

Individual images should not be too big (i.e. < 850 x 850 pixel). Although this
can be taken care of later as well, it is advisable to crop the frames, to
remove unnecessary parts of the frame as much as possible.
"""

import imageio
imageio.plugins.ffmpeg.download()
import matplotlib
matplotlib.use('Agg')
from moviepy.editor import VideoFileClip
from skimage import io
from skimage.util import img_as_ubyte
import numpy as np
import os
import math
import sys
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])
from myconfig import Task, videopath, videotype, filename, x1, x2, y1, y2, start, stop, \
    date, cropping, numframes2pick, selectionalgorithm, checkcropping
import auxiliaryfunctions, frameselectiontools

def CheckCropping(videopath,filename,x1,x2,y1,y2,cropping, videotype, time=start):
    ''' Display frame at time "time" for video to check if cropping is fine. 
    Select ROI of interest by adjusting values in myconfig.py
    
    USAGE for cropping:
    clip.crop(x1=None, y1=None, x2=None, y2=None, width=None, height=None, x_center=None, y_center=None)
    
    Returns a new clip in which just a rectangular subregion of the
    original clip is conserved. x1,y1 indicates the top left corner and
    x2,y2 is the lower right corner of the croped region.
    
    All coordinates are in pixels. Float numbers are accepted.
    '''
    videos=auxiliaryfunctions.GetVideoList(filename,videopath,videotype)
    if filename!='all':
        videotype=filename.split('.')[1]

    for vindex,video in enumerate(videos):
        clip = VideoFileClip(os.path.join(videopath,video))
        print("Extracting ", video)
        
        ny, nx = clip.size  # dimensions of frame (width, height)
        if cropping==True:
            # Select ROI of interest by adjusting values in myconfig.py
            clip=clip.crop(y1=y1,y2=y2,x1 = x1,x2=x2)
        
        image = clip.get_frame(time*clip.duration) #frame is accessed by index *1./clip.fps (fps cancels)
        io.imsave("IsCroppingOK"+video.split('.')[0]+".png", image)
        
        if vindex==len(videos)-1:
            print("--> Open the CroppingOK-videofilename-.png file(s) to set the output range! <---")
            print("--> Adjust shiftx, shifty, fx and fy accordingly! <---")
    return image

def SelectFrames(videopath,filename,x1,x2,y1,y2,cropping, videotype,start,stop,Task,selectionalgorithm):
    ''' Selecting frames from videos for labeling.'''
    if start>1.0 or stop>1.0 or start<0 or stop<0 or start>=stop:
            print("Please change start & stop, they should form a normalized interval with 0<= start < stop<=1.")
    else:
        basefolder = 'data-' + Task + '/'
        auxiliaryfunctions.attempttomakefolder(basefolder)
        videos=auxiliaryfunctions.GetVideoList(filename,videopath,videotype)
        for vindex,video in enumerate(videos):
            print("Loading ", video)
            clip = VideoFileClip(os.path.join(videopath, video))
            print("Duration of video [s], ", clip.duration, "fps, ", clip.fps,
                  "Cropped frame dimensions: ", clip.size)

            ####################################################
            # Creating folder with name of experiment and extract random frames
            ####################################################
            folder = video.split('.')[0]
            auxiliaryfunctions.attempttomakefolder(os.path.join(basefolder,folder))
            indexlength = int(np.ceil(np.log10(clip.duration * clip.fps)))
            # Extract the first frame (not cropped!) - useful for data augmentation
            index = 0
            image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
            io.imsave(os.path.join(basefolder,folder,"img" + str(index).zfill(indexlength) + ".png"),image)
            
            if cropping==True:
                # Select ROI of interest by adjusting values in myconfig.py
                clip=clip.crop(y1=y1,y2=y2,x1 = x1,x2=x2)
            print("Extracting frames ...")
            if selectionalgorithm=='uniform':
                frames2pick=frameselectiontools.UniformFrames(clip,numframes2pick,start,stop)
            elif selectionalgorithm=='kmeans':
                frames2pick=frameselectiontools.KmeansbasedFrameselection(clip,numframes2pick,start,stop)
            else:
                print("Please implement this method yourself and send us a pull request!")
                frames2pick=[]
            
            indexlength = int(np.ceil(np.log10(clip.duration * clip.fps))) 
            for index in frames2pick:
                try:
                    image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
                    io.imsave(os.path.join(basefolder,folder,"img" + str(index).zfill(indexlength) + ".png"),image)
                except FileNotFoundError:
                    print("Frame # ", index, " does not exist.")

if __name__ == "__main__":
    if checkcropping==True:
        #####################################################################
        # First load the image and crop (if necessary)/ set checkcropping = True (in myconfig to do so)
        #####################################################################
        CheckCropping(videopath,filename,x1,x2,y1,y2,cropping, videotype)
    else:
        ####################################################
        # Creating folder with name of experiment and extract random frames
        ####################################################
        SelectFrames(videopath,filename,x1,x2,y1,y2,cropping, videotype,start,stop,Task,selectionalgorithm)