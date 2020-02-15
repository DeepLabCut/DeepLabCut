#!/usr/bin/env python3
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

from pathlib import Path
import subprocess, os
import cv2

# Historically DLC used: from scipy.misc import imread, imresize >> deprecated functions
def imread(path,mode=None):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

#https://docs.opencv.org/3.4.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
def imresize(img,size=1.0,interpolationmethod=cv2.INTER_AREA):
    if size!=1.0:
        return cv2.resize(img,None,fx=size,fy=size,interpolation=interpolationmethod) #(int(height*size),int(width*size)))
    else:
        return img

def ShortenVideo(vname,start='00:00:01',stop='00:01:00',outsuffix='short',outpath=None):
    """
    Auxiliary function to shorten video and output with outsuffix appended.
    to the same folder from start (hours:minutes:seconds) to stop (hours:minutes:seconds).

    Returns the full path to the shortened video!

    Parameter
    ----------
    videos : string
        A string containing the full paths of the video.

    start: hours:minutes:seconds
        Time formatted in hours:minutes:seconds, where shortened video shall start.

    stop: hours:minutes:seconds
        Time formatted in hours:minutes:seconds, where shortened video shall end.

    outsuffix: str
        Suffix for output videoname (see example).

    outpath: str
        Output path for saving video to (by default will be the same folder as the video)

    Linux/MacOs
    >>> deeplabcut.ShortenVideo('/data/videos/mouse1.avi')

    Extracts (sub)video from 1st second to 1st minutes (default values) and saves it in /data/videos as mouse1short.avi

    Windows:
    >>> deeplabcut.ShortenVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', start='00:17:00',stop='00:22:00',outsuffix='brief')

    Extracts (sub)video from minute 17 to 22 and and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1brief.avi
    """
    if outpath is None:
        vidpath=os.path.dirname(vname)
    else:
        vidpath=outpath

    #TODO check if those times exist...
    newfilename=os.path.join(vidpath,str(Path(vname).stem)+str(outsuffix)+str(Path(vname).suffix))
    print("Slicing and saving to name", newfilename)
    subprocess.call(['ffmpeg','-i',vname,'-ss',str(start),'-to',str(stop),'-c','copy',newfilename])
    return str(newfilename)

def DownSampleVideo(vname,width=-1,height=200,outsuffix='downsampled',outpath=None):
    """
    Auxiliary function to downsample a video and output it to the same folder with "outsuffix" appended in its name.
    Width and height will control the new dimensions. You can also pass only height or width and set the other one to -1,
    this will keep the aspect ratio identical.

    Returns the full path to the downsampled video!

    Parameter
    ----------
    vname : string
        A string containing the full path of the video.

    width: int
        width of output video

    height: int
        height of output video.

    outsuffix: str
        Suffix for output videoname (see example).

    outpath: str
        Output path for saving video to (by default will be the same folder as the video)

    Linux/MacOs
    >>> deeplabcut.DownSampleVideo('/data/videos/mouse1.avi')

    Downsamples the video using default values and saves it in /data/videos as mouse1cropped.avi

    Windows:
    >>> shortenedvideoname=deeplabcut.DownSampleVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', width=220,height=320,outsuffix='cropped')

    Downsamples the video to a width of 220 and height of 320 and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1cropped.avi
    """
    if outpath is None:
        vidpath=os.path.dirname(vname)
    else:
        vidpath=outpath

    newfilename=os.path.join(vidpath,str(Path(vname).stem)+str(outsuffix)+str(Path(vname).suffix))
    print("Downsampling and saving to name", newfilename)
    command = f"ffmpeg -i {vname} -filter:v scale={width}:{height} -c:a copy {newfilename}"
    subprocess.call(command, shell=True)
    return str(newfilename)
