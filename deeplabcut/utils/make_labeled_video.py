"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

Edited by:
Hao Wu, hwu01@g.harvard.edu
Who contributed his OpenCV class!

You can find the directory for ffmpeg bindings by: "find / | grep ffmpeg" and then setting it.
"""

####################################################
# Dependencies
####################################################
import os.path
#import sys
import argparse, glob, os
import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
from pathlib import Path

import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.config import load_config
from skimage.util import img_as_ubyte
from skimage.draw import circle_perimeter, circle
from deeplabcut.utils.video_processor import VideoProcessorCV as vp # used to CreateVideo


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def CreateVideo(clip,Dataframe,pcutoff,dotsize,colormap,DLCscorer,bodyparts2plot,cropping,x1,x2,y1,y2):
        ''' Creating individual frames with labeled body parts and making a video'''
        colorclass=plt.cm.ScalarMappable(cmap=colormap)
        C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts2plot)))
        colors=(C[:,:3]*255).astype(np.uint8)
        if cropping:
            ny, nx= y2-y1,x2-x1
        else:
            ny, nx= clip.height(), clip.width()
        fps=clip.fps()
        nframes = len(Dataframe.index)
        duration = nframes/fps

        print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
        print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",nx,ny)

        print("Generating frames and creating video.")
        for index in tqdm(range(nframes)):
            image = clip.load_frame()
            if cropping:
                    image=image[y1:y2,x1:x2]
            else:
                pass
            for bpindex, bp in enumerate(bodyparts2plot):
                if Dataframe[DLCscorer][bp]['likelihood'].values[index] > pcutoff:
                    xc = int(Dataframe[DLCscorer][bp]['x'].values[index])
                    yc = int(Dataframe[DLCscorer][bp]['y'].values[index])
                    #rr, cc = circle_perimeter(yc,xc,radius)
                    rr, cc = circle(yc,xc,dotsize,shape=(ny,nx))
                    image[rr, cc, :] = colors[bpindex]

            frame = image
            clip.save_frame(frame)
        clip.close()

def CreateVideoSlow(clip,Dataframe,tmpfolder,dotsize,colormap,alphavalue,pcutoff,cropping,x1,x2,y1,y2,delete,DLCscorer,bodyparts2plot):
    ''' Creating individual frames with labeled body parts and making a video'''
    #scorer=np.unique(Dataframe.columns.get_level_values(0))[0]
    #bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    colors = get_cmap(len(bodyparts2plot),name=colormap)
    if cropping:
        ny, nx= y2-y1,x2-x1
    else:
        ny, nx= clip.height(), clip.width()
    fps=clip.fps()
    nframes = len(Dataframe.index)
    duration = nframes/fps

    print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
    print("Overall # of frames: ", int(nframes), "with cropped frame dimensions: ",nx,ny)
    print("Generating frames")
    for index in tqdm(range(nframes)):
        imagename = tmpfolder + "/file%04d.png" % index
        if os.path.isfile(tmpfolder + "/file%04d.png" % index):
            image = img_as_ubyte(clip.load_frame()) #still need to read (so counter advances!)
        else:
            plt.axis('off')
            
            image = img_as_ubyte(clip.load_frame())
            if cropping:
                    image=image[y1:y2,x1:x2]
            else:
                pass
            plt.figure(frameon=False, figsize=(nx * 1. / 100, ny * 1. / 100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.imshow(image)

            for bpindex, bp in enumerate(bodyparts2plot):
                if Dataframe[DLCscorer][bp]['likelihood'].values[index] > pcutoff:
                    plt.scatter(
                        Dataframe[DLCscorer][bp]['x'].values[index],
                        Dataframe[DLCscorer][bp]['y'].values[index],
                        s=dotsize**2,
                        color=colors(bpindex),
                        alpha=alphavalue)

            plt.xlim(0, nx)
            plt.ylim(0, ny)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(imagename)

            plt.close("all")

    start= os.getcwd()
    os.chdir(tmpfolder)

    print("All labeled frames were created, now generating video...")
    vname=str(Path(tmpfolder).stem).split('-')[1]
    try: ## One can change the parameters of the video creation script below:
        subprocess.call([
            'ffmpeg', '-framerate',
            str(clip.fps()), '-i', 'file%04d.png', '-r', '30','../'+vname + DLCscorer+'_labeled.mp4'])
    except FileNotFoundError:
        print("Ffmpeg not correctly installed, see https://github.com/AlexEMG/DeepLabCut/issues/45")

    if delete:
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
    os.chdir(start)

def create_labeled_video(config,videos,shuffle=1,trainingsetindex=0,videotype='avi',save_frames=False,delete=False,displayedbodyparts='all',codec='mp4v'):
    """
    Labels the bodyparts in a video. Make sure the video is already analyzed by the function 'analyze_video'

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of string containing the full paths of the videos to analyze.

    shuffle : int, optional
        Number of shuffles of training dataset. Default is set to 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
     
    videotype: string, optional
        Checks for the extension of the video in case the input is a directory.\nOnly videos with this extension are analyzed. The default is ``.avi``

    save_frames: bool
        If true creates each frame individual and then combines into a video. This variant is relatively slow as
        it stores all individual frames. However, it uses matplotlib to create the frames and is therefore much more flexible (one can set transparency of markers, crop, and easily customize).

    delete: bool
        If true then the individual frames created during the video generation will be deleted.

    displayedbodyparts: list of strings, optional
        This select the body parts that are plotted in the video. Either ``all``, then all body parts
        from config.yaml are used orr a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    codec: codec for labeled video. Options see http://www.fourcc.org/codecs.php [depends on your ffmpeg installation.]
    
    Examples
    --------
    If you want to create the labeled video for only 1 video
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'])
    --------

    If you want to create the labeled video for only 1 video and store the individual frames
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'],save_frames=True)
    --------

    If you want to create the labeled video for multiple videos
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'])
    --------

    If you want to create the labeled video for all the videos (as .avi extension) in a directory.
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/'])

    --------
    If you want to create the labeled video for all the videos (as .mp4 extension) in a directory.
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/'],videotype='mp4')

    --------

    """
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction) #automatically loads corresponding model (even training iteration based on snapshot index)

    bodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,displayedbodyparts)
    
    if [os.path.isdir(i) for i in videos] == [True]:
      print("Analyzing all the videos in the directory")
      videofolder= videos[0]
      os.chdir(videofolder)
      Videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
      print("Starting ", videofolder, Videos)
    else:
      Videos = videos

    for video in Videos:
        videofolder= Path(video).parents[0] #where your folder with videos is.
        os.chdir(str(videofolder))
        videotype = Path(video).suffix
        print("Starting % ", videofolder, videos)
        vname = str(Path(video).stem)
        if os.path.isfile(os.path.join(str(videofolder),vname + DLCscorer+'_labeled.mp4')):
            print("Labeled video already created.")
        else:
            print("Loading ", video, "and data.")
            dataname = os.path.join(str(videofolder),vname+DLCscorer + '.h5')
            try:
                Dataframe = pd.read_hdf(dataname)
                metadata=auxiliaryfunctions.LoadVideoMetadata(dataname)
                #print(metadata)
                datanames=[dataname]
            except FileNotFoundError:
                datanames=[fn for fn in os.listdir(os.curdir) if (vname in fn) and (".h5" in fn) and "resnet" in fn]
                if len(datanames)==0:
                    print("The video was not analyzed with this scorer:", DLCscorer)
                    print("No other scorers were found, please use the function 'analyze_videos' first.")
                elif len(datanames)>0:
                    print("The video was not analyzed with this scorer:", DLCscorer)
                    print("Other scorers were found, however:", datanames)
                    DLCscorer='DeepCut'+(datanames[0].split('DeepCut')[1]).split('.h5')[0]
                    print("Creating labeled video for:", DLCscorer," instead.")
                    Dataframe = pd.read_hdf(datanames[0])
                    metadata=auxiliaryfunctions.LoadVideoMetadata(datanames[0])

            if len(datanames)>0:
                #Loading cropping data used during analysis
                cropping=metadata['data']["cropping"]
                [x1,x2,y1,y2]=metadata['data']["cropping_parameters"]
                print(cropping,x1,x2,y1,y2)
                
                if save_frames==True:
                    tmpfolder = os.path.join(str(videofolder),'temp-' + vname)
                    auxiliaryfunctions.attempttomakefolder(tmpfolder)
                    clip = vp(video)
                    #CreateVideoSlow(clip,Dataframe,tmpfolder,cfg["dotsize"],cfg["colormap"],cfg["alphavalue"],cfg["pcutoff"],cfg["cropping"],cfg["x1"],cfg["x2"],cfg["y1"],cfg["y2"],delete,DLCscorer,bodyparts)
                    CreateVideoSlow(clip,Dataframe,tmpfolder,cfg["dotsize"],cfg["colormap"],cfg["alphavalue"],cfg["pcutoff"],cropping,x1,x2,y1,y2,delete,DLCscorer,bodyparts)
                else:
                    clip = vp(fname = video,sname = os.path.join(vname + DLCscorer+'_labeled.mp4'),codec=codec)
                    if cropping:
                        print("Fast video creation has currently not been implemented for cropped videos. Please use 'save_frames=True' to get the video.")
                    else:
                        CreateVideo(clip,Dataframe,cfg["pcutoff"],cfg["dotsize"],cfg["colormap"],DLCscorer,bodyparts,cropping,x1,x2,y1,y2) #NEED TO ADD CROPPING!

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('videos')
    cli_args = parser.parse_args()
