#!/usr/bin/env python3
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os
import subprocess
from pathlib import Path

import cv2


# Historically DLC used: from scipy.misc import imread, imresize >> deprecated functions
def imread(path, mode=None):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# https://docs.opencv.org/3.4.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
def imresize(img, size=1.0, interpolationmethod=cv2.INTER_AREA):
    if size != 1.0:
        return cv2.resize(
            img, None, fx=size, fy=size, interpolation=interpolationmethod
        )  # (int(height*size),int(width*size)))
    else:
        return img


def ShortenVideo(
    vname, start="00:00:01", stop="00:01:00", outsuffix="short", outpath=None
):
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

    Examples
    ----------

    Linux/MacOs
    >>> deeplabcut.ShortenVideo('/data/videos/mouse1.avi')

    Extracts (sub)video from 1st second to 1st minutes (default values) and saves it in /data/videos as mouse1short.avi

    Windows:
    >>> deeplabcut.ShortenVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', start='00:17:00',stop='00:22:00',outsuffix='brief')

    Extracts (sub)video from minute 17 to 22 and and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1brief.avi
    """
    if outpath is None:
        vidpath = os.path.dirname(vname)
    else:
        vidpath = outpath

    # TODO check if those times exist...
    newfilename = os.path.join(
        vidpath, str(Path(vname).stem) + str(outsuffix) + str(Path(vname).suffix)
    )
    print("Slicing and saving to name", newfilename)
    command = f"ffmpeg -i {vname} -ss {start} -to {stop} -c:a copy {newfilename}"
    subprocess.call(command, shell=True)
    # subprocess.call(['ffmpeg','-i',vname,'-ss',str(start),'-to',str(stop),'-c:v','copy','-c:a', newfilename])
    return str(newfilename)


def CropVideo(
    vname,
    width=256,
    height=256,
    origin_x=0,
    origin_y=0,
    outsuffix="cropped",
    outpath=None,
    useGUI=False,
):
    """
    Auxiliary function to crop a video and output it to the same folder with "outsuffix" appended in its name.
    Width and height will control the new dimensions.

    Returns the full path to the downsampled video!

    ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4

    Parameter
    ----------
    vname : string
        A string containing the full path of the video.

    width: int
        width of output video

    height: int
        height of output video.

    origin_x, origin_y: int
        x- and y- axis origin of bounding box for cropping.

    outsuffix: str
        Suffix for output videoname (see example).

    outpath: str
        Output path for saving video to (by default will be the same folder as the video)

    Examples
    ----------

    Linux/MacOs
    >>> deeplabcut.CropVideo('/data/videos/mouse1.avi')

    Crops the video using default values and saves it in /data/videos as mouse1cropped.avi

    Windows:
    >>> =deeplabcut.CropVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', width=220,height=320,outsuffix='cropped')

    Crops the video to a width of 220 and height of 320 starting at the origin (top left) and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1cropped.avi
    """
    if outpath is None:
        vidpath = os.path.dirname(vname)
    else:
        vidpath = outpath

    if useGUI:
        print(
            "Please, select your coordinates (draw from top left to bottom right ...)"
        )
        coords = draw_bbox(vname)

        if not coords:
            return
        origin_x, origin_y = coords[:2]
        width = int(coords[2]) - int(coords[0])
        height = int(coords[3]) - int(coords[1])

    newfilename = os.path.join(
        vidpath, str(Path(vname).stem) + str(outsuffix) + str(Path(vname).suffix)
    )

    print("Cropping and saving to name", newfilename)
    command = f"ffmpeg -i {vname} -filter:v crop={width}:{height}:{origin_x}:{origin_y} -c:a copy {newfilename}"
    subprocess.call(command, shell=True)
    return str(newfilename)


def DownSampleVideo(
    vname, width=-1, height=200, outsuffix="downsampled", outpath=None, rotateccw=False
):
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

    rotateccw: bool
        Default false, rotates counter-clockwise if true.

    Examples
    ----------

    Linux/MacOs
    >>> deeplabcut.DownSampleVideo('/data/videos/mouse1.avi')

    Downsamples the video using default values and saves it in /data/videos as mouse1cropped.avi

    Windows:
    >>> shortenedvideoname=deeplabcut.DownSampleVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', width=220,height=320,outsuffix='cropped')

    Downsamples the video to a width of 220 and height of 320 and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1cropped.avi
    """
    if outpath is None:
        vidpath = os.path.dirname(vname)
    else:
        vidpath = outpath

    newfilename = os.path.join(
        vidpath, str(Path(vname).stem) + str(outsuffix) + str(Path(vname).suffix)
    )

    # Rotate, see: https://stackoverflow.com/questions/3937387/rotating-videos-with-ffmpeg
    # interesting option to just update metadata.

    print("Downsampling and saving to name", newfilename)
    if rotateccw:
        command = f"ffmpeg -i {vname} -filter:v scale={width}:{height} -vf 'transpose=1' -c:a copy {newfilename}"
    else:
        command = f"ffmpeg -i {vname} -filter:v scale={width}:{height} -c:a copy {newfilename}"
    subprocess.call(command, shell=True)
    return str(newfilename)


def draw_bbox(video):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector, Button

    clip = cv2.VideoCapture(video)
    if not clip.isOpened():
        print("Video could not be opened. Skipping...")
        return

    success = False
    # Read the video until a frame is successfully read
    while not success:
        success, frame = clip.read()

    bbox = [0, 0, frame.shape[1], frame.shape[0]]

    def line_select_callback(eclick, erelease):
        bbox[:2] = int(eclick.xdata), int(eclick.ydata)  # x1, y1
        bbox[2:] = int(erelease.xdata), int(erelease.ydata)  # x2, y2

    def validate_crop(*args):
        fig.canvas.stop_event_loop()

    def display_help(*args):
        print(
            "1. Use left click to select the region of interest. A red box will be drawn around the selected region. \n\n2. Use the corner points to expand the box and center to move the box around the image. \n\n3. Click "
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(frame[:, :, ::-1])
    ax_help = fig.add_axes([0.9, 0.2, 0.1, 0.1])
    ax_save = fig.add_axes([0.9, 0.1, 0.1, 0.1])
    crop_button = Button(ax_save, "Crop")
    crop_button.on_clicked(validate_crop)
    help_button = Button(ax_help, "Help")
    help_button.on_clicked(display_help)

    rs = RectangleSelector(
        ax,
        line_select_callback,
        drawtype="box",
        minspanx=5,
        minspany=5,
        interactive=True,
        spancoords="pixels",
        rectprops=dict(facecolor="red", edgecolor="black", alpha=0.3, fill=True),
    )
    plt.show()

    # import platform
    # if platform.system() == "Darwin":  # for OSX use WXAgg
    #    fig.canvas.start_event_loop(timeout=-1)
    # else:
    fig.canvas.start_event_loop(timeout=-1)  # just tested on Ubuntu I also need this.
    #    #fig.canvas.stop_event_loop()

    plt.close(fig)
    return bbox


def get_duration(videofile):
    """Get the duration of a video using ffprobe."""
    cmd = (
        f'ffprobe -i "{videofile}" -show_entries format=duration -v quiet -of csv="p=0"'
    )
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    return float(output)


def get_nframes_robust(videofile):
    cmd = f'ffprobe -i "{videofile}" -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1'
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    return int(output)
