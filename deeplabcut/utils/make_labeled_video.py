"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Hao Wu, hwu01@g.harvard.edu contributed the original OpenCV class!
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
import platform

import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
elif platform.system() == 'Darwin':
    mpl.use('WxAgg') #TkAgg
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.config import load_config
from skimage.util import img_as_ubyte
from skimage.draw import circle_perimeter, circle, line,line_aa

from deeplabcut.utils.video_processor import VideoProcessorCV as vp # used to CreateVideo

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def CreateVideo(clip,Dataframe,pcutoff,dotsize,colormap,DLCscorer,bodyparts2plot,
                trailpoints,cropping,x1,x2,y1,y2,
                bodyparts2connect,skeleton_color,draw_skeleton,displaycropped):
        ''' Creating individual frames with labeled body parts and making a video'''
        colorclass=plt.cm.ScalarMappable(cmap=colormap)
        C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts2plot)))
        colors=(C[:,:3]*255).astype(np.uint8)

        if draw_skeleton:
            color_for_skeleton = (np.array(mpl.colors.to_rgba(skeleton_color))[:3]*255).astype(np.uint8)
            #recode the bodyparts2connect into indices for df_x and df_y for speed
            bpts2connect=[]
            index=np.arange(len(bodyparts2plot))
            for pair in bodyparts2connect:
                if pair[0] in bodyparts2plot and pair[1] in bodyparts2plot:
                    bpts2connect.append([index[pair[0]==np.array(bodyparts2plot)][0],index[pair[1]==np.array(bodyparts2plot)][0]])

        if displaycropped:
            ny, nx= y2-y1,x2-x1
        else:
            ny, nx= clip.height(), clip.width()

        fps=clip.fps()
        nframes = len(Dataframe.index)
        duration = nframes/fps

        print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
        print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",nx,ny)

        print("Generating frames and creating video.")
        df_likelihood = np.empty((len(bodyparts2plot),nframes))
        df_x = np.empty((len(bodyparts2plot),nframes))
        df_y = np.empty((len(bodyparts2plot),nframes))
        for bpindex, bp in enumerate(bodyparts2plot):
            df_likelihood[bpindex,:]=Dataframe[DLCscorer,bp,'likelihood'].values
            if cropping and not displaycropped:
                df_x[bpindex,:]=Dataframe[DLCscorer,bp,'x'].values+x1
                df_y[bpindex,:]=Dataframe[DLCscorer,bp,'y'].values+y1
            else:
                df_x[bpindex,:]=Dataframe[DLCscorer,bp,'x'].values
                df_y[bpindex,:]=Dataframe[DLCscorer,bp,'y'].values


        for index in tqdm(range(nframes)):
            image = clip.load_frame()
            if displaycropped:
                    image=image[y1:y2,x1:x2]

            for bpindex in range(len(bodyparts2plot)):
                # Draw the skeleton for specific bodyparts to be connected as specified in the config file
                if draw_skeleton:
                    for pair in bpts2connect:
                        if (df_likelihood[pair[0],index] > pcutoff) and (df_likelihood[pair[1],index] >pcutoff):
#                           rr, cc,val = line_aa(int(df_y[pair[0],index]),int(df_x[pair[0],index]),int(df_y[pair[1],index]), int(df_x[pair[1],index]))
                            rr, cc,val = line_aa(int(np.clip(df_y[pair[0],index],0,ny-1)),int(np.clip(df_x[pair[0],index],0,nx-1)), int(np.clip(df_y[pair[1],index],1,ny-1)), int(np.clip(df_x[pair[1],index],1,nx-1)))
                            image[rr, cc,:] = color_for_skeleton

                if df_likelihood[bpindex,index] > pcutoff:
                    if trailpoints>0: #plot history
                        for k in range(min(trailpoints,index+1)):
                            rr, cc = circle(df_y[bpindex,index-k],df_x[bpindex,index-k],dotsize,shape=(ny,nx))
                            image[rr, cc, :] = colors[bpindex]
                    else:
                        xc = int(df_x[bpindex,index])
                        yc = int(df_y[bpindex,index])
                        rr, cc = circle(yc,xc,dotsize,shape=(ny,nx))
                        image[rr, cc, :] = colors[bpindex]

            frame = image
            clip.save_frame(frame)
        clip.close()


def CreateVideoSlow(videooutname,clip,Dataframe,tmpfolder,
                    dotsize,colormap,alphavalue,pcutoff,trailpoints,cropping,x1,x2,y1,y2,
                    delete,DLCscorer,bodyparts2plot,outputframerate,Frames2plot,
                    bodyparts2connect,skeleton_color,draw_skeleton,displaycropped):
    ''' Creating individual frames with labeled body parts and making a video'''
    #scorer=np.unique(Dataframe.columns.get_level_values(0))[0]
    #bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))

    if displaycropped:
        ny, nx= y2-y1,x2-x1
    else:
        ny, nx= clip.height(), clip.width()

    fps=clip.fps()
    if  outputframerate is None: #by def. same as input rate.
        outputframerate=clip.fps()

    nframes = len(Dataframe.index)
    duration = nframes/fps

    print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
    print("Overall # of frames: ", int(nframes), "with cropped frame dimensions: ",nx,ny)
    print("Generating frames and creating video.")
    df_likelihood = np.empty((len(bodyparts2plot),nframes))
    df_x = np.empty((len(bodyparts2plot),nframes))
    df_y = np.empty((len(bodyparts2plot),nframes))

    for bpindex, bp in enumerate(bodyparts2plot):
        df_likelihood[bpindex,:]=Dataframe[DLCscorer,bp,'likelihood'].values
        if cropping and not displaycropped:
            df_x[bpindex,:]=Dataframe[DLCscorer,bp,'x'].values+x1
            df_y[bpindex,:]=Dataframe[DLCscorer,bp,'y'].values+y1
        else:
            df_x[bpindex,:]=Dataframe[DLCscorer,bp,'x'].values
            df_y[bpindex,:]=Dataframe[DLCscorer,bp,'y'].values

    colors = get_cmap(len(bodyparts2plot),name=colormap)
    if draw_skeleton:
            #recode the bodyparts2connect into indices for df_x and df_y for speed
            bpts2connect=[]
            index=np.arange(len(bodyparts2plot))
            for pair in bodyparts2connect:
                if pair[0] in bodyparts2plot and pair[1] in bodyparts2plot:
                    bpts2connect.append([index[pair[0]==np.array(bodyparts2plot)][0],index[pair[1]==np.array(bodyparts2plot)][0]])

    nframes_digits=int(np.ceil(np.log10(nframes)))
    if nframes_digits>9:
        raise Exception("Your video has more than 10**9 frames, we recommend chopping it up.")

    if Frames2plot==None:
        Index=range(nframes)
    else:
        Index=[]
        for k in Frames2plot:
            if k>=0 and k<nframes:
                Index.append(int(k))

    for index in tqdm(range(nframes)):
        imagename = tmpfolder + "/file"+str(index).zfill(nframes_digits)+".png"
        if os.path.isfile(imagename):
            image = img_as_ubyte(clip.load_frame()) #still need to read (so counter advances!)
        else:
            plt.axis('off')
            image = img_as_ubyte(clip.load_frame())
            if index in Index: #then extract the frame!
                if cropping and displaycropped:
                        image=image[y1:y2,x1:x2]
                else:
                    pass

                plt.figure(frameon=False, figsize=(nx * 1. / 100, ny * 1. / 100))
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.imshow(image)

                # Adds skeleton to the video
                ####################
                if draw_skeleton:
                    for pair in bpts2connect:
                        if (df_likelihood[pair[0],index] > pcutoff) and (df_likelihood[pair[1],index] >pcutoff):
                                plt.plot([df_x[pair[0],index],df_x[pair[1],index]],[df_y[pair[0],index],df_y[pair[1],index]],color=skeleton_color,alpha=alphavalue)

                for bpindex, bp in enumerate(bodyparts2plot):
                    if df_likelihood[bpindex,index] > pcutoff:
                        if trailpoints>0:
                            plt.scatter(
                                df_x[bpindex][max(0,index-trailpoints):index],
                                df_y[bpindex][max(0,index-trailpoints):index],
                                s=dotsize**2,
                                color=colors(bpindex),
                                alpha=alphavalue*.75)
                            #less transparent present.
                            plt.scatter(
                                df_x[bpindex,index],
                                df_y[bpindex,index],
                                s=dotsize**2,
                                color=colors(bpindex),
                                alpha=alphavalue)

                        else:
                            plt.scatter(
                                df_x[bpindex,index],
                                df_y[bpindex,index],
                                s=dotsize**2,
                                color=colors(bpindex),
                                alpha=alphavalue)

                plt.xlim(0, nx-1)
                plt.ylim(0, ny-1)

                plt.axis('off')
                plt.subplots_adjust(
                    left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.gca().invert_yaxis()
                plt.savefig(imagename)

                plt.close("all")

    start= os.getcwd()
    os.chdir(tmpfolder)
    print("All labeled frames were created, now generating video...")
    ## One can change the parameters of the video creation script below:
    # See ffmpeg user guide: http://ffmpeg.org/ffmpeg.html#Video-and-Audio-file-format-conversion
    #
    try:
        subprocess.call([
            'ffmpeg', '-framerate',
            str(clip.fps()), '-i', 'file%0'+str(nframes_digits)+'d.png', '-r', str(outputframerate),'../'+videooutname])
    except FileNotFoundError:
        print("Ffmpeg not correctly installed, see https://github.com/AlexEMG/DeepLabCut/issues/45")

    if delete:
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
    os.chdir(start)

def create_labeled_video(config,videos,videotype='avi',shuffle=1,trainingsetindex=0,filtered=False,save_frames=False,Frames2plot=None,delete=False,displayedbodyparts='all',codec='mp4v',outputframerate=None, destfolder=None,draw_skeleton=False,trailpoints = 0,displaycropped=False):
    """
    Labels the bodyparts in a video. Make sure the video is already analyzed by the function 'analyze_video'

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle : int, optional
        Number of shuffles of training dataset. Default is set to 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    filtered: bool, default false
        Boolean variable indicating if filtered output should be plotted rather than frame-by-frame predictions. Filtered version can be calculated with deeplabcut.filterpredictions

    videotype: string, optional
        Checks for the extension of the video in case the input is a directory.\nOnly videos with this extension are analyzed. The default is ``.avi``

    save_frames: bool
        If true creates each frame individual and then combines into a video. This variant is relatively slow as
        it stores all individual frames. However, it uses matplotlib to create the frames and is therefore much more flexible (one can set transparency of markers, crop, and easily customize).

    Frames2plot: List of indices
        If not None & save_frames=True then the frames corresponding to the index will be plotted. For example, Frames2plot=[0,11] will plot the first and the 12th frame.

    delete: bool
        If true then the individual frames created during the video generation will be deleted.

    displayedbodyparts: list of strings, optional
        This select the body parts that are plotted in the video. Either ``all``, then all body parts
        from config.yaml are used orr a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    codec: codec for labeled video. Options see http://www.fourcc.org/codecs.php [depends on your ffmpeg installation.]

    outputframerate: positive number, output frame rate for labeled video (only available for the mode with saving frames.) By default: None, which results in the original video rate.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    draw_skeleton: bool
        If ``True`` adds a line connecting the body parts making a skeleton on on each frame. The body parts to be connected and the color of these connecting lines are specified in the config file. By default: ``False``

    trailpoints: int
        Number of revious frames whose body parts are plotted in a frame (for displaying history). Default is set to 0.

    displaycropped: bool, optional
        Specifies whether only cropped frame is displayed (with labels analyzed therein), or the original frame with the labels analyzed in the cropped subset.

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
    DLCscorer,DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction) #automatically loads corresponding model (even training iteration based on snapshot index)

    bodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,displayedbodyparts)
    if draw_skeleton:
        bodyparts2connect = cfg['skeleton']
        skeleton_color = cfg['skeleton_color']
    else:
        bodyparts2connect = None
        skeleton_color = None

    Videos=auxiliaryfunctions.Getlistofvideos(videos,videotype)
    for video in Videos:
        if destfolder is None:
            videofolder= Path(video).parents[0] #where your folder with videos is.
        else:
            videofolder=destfolder

        os.chdir(str(videofolder))
        videotype = Path(video).suffix
        print("Starting % ", videofolder, videos)
        vname = str(Path(video).stem)


        #if notanalyzed:
        #notanalyzed,outdataname,sourcedataname,DLCscorer=auxiliaryfunctions.CheckifPostProcessing(folder,vname,DLCscorer,DLCscorerlegacy,suffix='checking')

        if filtered==True:
            videooutname1=os.path.join(vname + DLCscorer+'filtered_labeled.mp4')
            videooutname2=os.path.join(vname + DLCscorerlegacy+'filtered_labeled.mp4')
        else:
            videooutname1=os.path.join(vname + DLCscorer+'_labeled.mp4')
            videooutname2=os.path.join(vname + DLCscorerlegacy+'_labeled.mp4')

        if os.path.isfile(videooutname1) or os.path.isfile(videooutname2):
            print("Labeled video already created.")
        else:
            print("Loading ", video, "and data.")
            datafound,metadata,Dataframe,DLCscorer,suffix=auxiliaryfunctions.LoadAnalyzedData(str(videofolder),vname,DLCscorer,filtered) #returns boolean variable if data was found and metadata + pandas array
            videooutname=os.path.join(vname + DLCscorer+suffix+'_labeled.mp4')
            if datafound and not os.path.isfile(videooutname): #checking again, for this loader video could exist
                #Loading cropping data used during analysis
                cropping=metadata['data']["cropping"]
                [x1,x2,y1,y2]=metadata['data']["cropping_parameters"]
                if save_frames==True:
                    tmpfolder = os.path.join(str(videofolder),'temp-' + vname)
                    auxiliaryfunctions.attempttomakefolder(tmpfolder)
                    clip = vp(video)

                    CreateVideoSlow(videooutname,clip,Dataframe,tmpfolder,cfg["dotsize"],cfg["colormap"],cfg["alphavalue"],cfg["pcutoff"],trailpoints,cropping,x1,x2,y1,y2,delete,DLCscorer,bodyparts,outputframerate,Frames2plot,bodyparts2connect,skeleton_color,draw_skeleton,displaycropped)
                else:
                    if displaycropped: #then the cropped video + the labels is depicted
                        clip = vp(fname = video,sname = videooutname,codec=codec,sw=x2-x1,sh=y2-y1)
                        CreateVideo(clip,Dataframe,cfg["pcutoff"],cfg["dotsize"],cfg["colormap"],DLCscorer,bodyparts,trailpoints,cropping,x1,x2,y1,y2,bodyparts2connect,skeleton_color,draw_skeleton,displaycropped)
                    else: #then the full video + the (perhaps in cropped mode analyzed labels) are depicted
                        clip = vp(fname = video,sname = videooutname,codec=codec)
                        CreateVideo(clip,Dataframe,cfg["pcutoff"],cfg["dotsize"],cfg["colormap"],DLCscorer,bodyparts,trailpoints,cropping,x1,x2,y1,y2,bodyparts2connect,skeleton_color,draw_skeleton,displaycropped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('videos')
    cli_args = parser.parse_args()
