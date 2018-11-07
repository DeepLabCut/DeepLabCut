"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""
from __future__ import print_function
import wx
import cv2
import os
import matplotlib
#matplotlib.use('TkAgg') #already set..

import matplotlib.pyplot as plt
import numpy as np


from moviepy.editor import VideoFileClip

#from skimage.util import img_as_ubyte

#import math
from pathlib import Path
import argparse
import yaml

class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent,config):

        wx.Frame.__init__(self, parent, title="DeepLabCut2.0 - Frame Extraction ToolBox", size=(1200, 980))

        self.split_win = wx.SplitterWindow(self)
        self.top_split = MatplotPanel(self.split_win,config)

        self.bottom_split = wx.Panel(self.split_win, style = wx.SUNKEN_BORDER)
        self.split_win.SplitHorizontally(self.top_split, self.bottom_split, 880)
#
        self.Button1 = wx.Button(self.bottom_split, -1, "Load Video", size=(150, 40), pos=(260, 25))
        self.Button1.Bind(wx.EVT_BUTTON, self.browseDir)
        self.Button1.Enable(True)

        self.Button2 = wx.Button(self.bottom_split, -1, "Grab a Frame", size=(150, 40), pos=(450, 25))
        self.Button2.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.Button2.Enable(False)

        self.Button3 = wx.Button(self.bottom_split, -1, "Help", size=(120, 40), pos=(640, 25))
        self.Button3.Bind(wx.EVT_BUTTON, self.helpButton)
        self.Button4 = wx.Button(self.bottom_split, -1, "Quit", size=(120, 40), pos=(800, 25))
        self.Button4.Bind(wx.EVT_BUTTON,self.quitButton)


        self.numberFrames = 0
        self.currFrame = 0
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.drs = []

        with open(str(config), 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile)
        self.Task = self.cfg['Task']
        self.start = self.cfg['start']
        self.stop = self.cfg['stop']
        self.date = self.cfg['date']
        self.videos = self.cfg['video_sets'].keys()
        self.video_names = [Path(i).stem for i in self.videos]
        self.config_path = Path(config)

    def quitButton(self, event):
        """
        Quits the GUI
        """
        self.Destroy()


    def CheckCropping(self):
        ''' Display frame at time "time" for video to check if cropping is fine.
        Select ROI of interest by adjusting values in myconfig.py

        USAGE for cropping:
        clip.crop(x1=None, y1=None, x2=None, y2=None, width=None, height=None, x_center=None, y_center=None)

        Returns a new clip in which just a rectangular subregion of the
        original clip is conserved. x1,y1 indicates the top left corner and
        x2,y2 is the lower right corner of the cropped region.

        All coordinates are in pixels. Float numbers are accepted.
        '''
        from skimage import io
        videosource = self.video_source
        filename = self.filename
        time = self.start
        self.x1 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[0])
        self.x2 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[1])
        self.y1 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[2])
        self.y2 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[3])

        clip = VideoFileClip(videosource)

        ny, nx = clip.size  # dimensions of frame (width, height)
        if self.cropping==True:
            # Select ROI of interest by adjusting values in myconfig.py
            clip=clip.crop(y1=self.y1,y2=self.y2,x1 = self.x1,x2=self.x2)
#            time = start
        image = clip.get_frame(time*clip.duration) #frame is accessed by index *1./clip.fps (fps cancels)
        fname = Path(filename)
        output_path = Path(self.config_path).parents[0] / 'labeled-data' / fname.stem
        if output_path.exists() :
            saveimg = str(Path(self.config_path).parents[0] / 'labeled-data') +'/IsCroppingOK_'+ str(fname)+".png" #str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames)))) + '.png'
            io.imsave(saveimg, image)
            print('Image cropped. Check the %s if cropping is ok, other wise change the parameters in the config file.' %saveimg)
        else:
            print('The path %s does not exist in the config file. Use add function to add this video in the config file and retry.' %output_path)
            self.Close(True)
        return image


    def OnSliderScroll(self, event):
        """
        Slider to scroll through the video
        """

        self.canvas.Destroy()
        self.ax1f1.clear()
        self.currFrame = (self.slider.GetValue())

        self.update()


    def browseDir(self, event):
        """
        Show the File Dialog and ask the user to select the video file
        """
        dlg = wx.FileDialog(self, "Choose a video", os.getcwd(), "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.video_source = dlg.GetPath()
            self.video_source = str(Path(self.video_source).resolve())
            self.Button1.Enable(False)
        else:
            dlg.Destroy()
            self.Close(True)
        dlg.Destroy()
        selectedvideo = Path(self.video_source)

        if  selectedvideo.stem in self.video_names:
            self.Button2.Enable(True)
            self.vid = cv2.VideoCapture(self.video_source)
            self.videoPath = os.path.dirname(self.video_source)
            self.filename = Path(self.video_source).name
            self.numberFrames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            if not self.vid.isOpened():
                msg = wx.MessageBox('Invalid Video file!Do you want to retry?', 'Error!', wx.YES_NO | wx.ICON_WARNING)
                if msg == 2:
                    self.Button1.Enable(True)
            self.slider = wx.Slider(self.top_split, -1, 0, self.currFrame, self.numberFrames, size=(950, -1),  pos=(140, 780),style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.update()

            cropMsg = wx.MessageBox("Do you want to crop the image?",'Want to crop?',wx.YES_NO|wx.ICON_INFORMATION)
            if cropMsg == 2:
                self.cropping = True
                self.basefolder = 'data-' + self.Task + '/'
                MainFrame.CheckCropping(self)
            else:
                self.cropping = False

        else:
            wx.MessageBox('Video file is not in config file. Use add function to add this video in the config file and retry!', 'Error!', wx.OK | wx.ICON_WARNING)
            self.Close(True)

    def update(self):
        """
        Updates the image with the current slider index
        """

        self.vid.set(1,self.currFrame)
        self.fig1, (self.ax1f1) = plt.subplots(figsize=(12, 7.8),facecolor = "None")
        ret, frame = self.vid.read()
        im_axis = self.ax1f1.imshow(frame)
        self.ax1f1.set_title(str(str(self.currFrame)+"/"+str(self.numberFrames) +" "+ self.filename))
        self.ax1f1.axis("off")
        # self.ax1f1.set_title(str(str(self.currFrame)+"/"+str(self.numberFrames) +" "+ self.filename))
        self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
        # self.canvas.Destroy()

    def grabFrame(self,event):
        """
        Extracts the frame and saves in the current directory
        """
        self.vid.set(1,self.currFrame)
        ret, frame = self.vid.read()
        fname = Path(self.filename)
        output_path = self.config_path.parents[0] / 'labeled-data' / fname.stem
        if ret and output_path.exists() :
            img_name = str(output_path) +'/img'+str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames)))) + '.png'
            if self.cropping:
                crop_img = frame[self.y1:self.y2, self.x1:self.x2]
                cv2.imwrite(img_name, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(img_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
             print('The path %s does not exist in the config file. Use add function to add this video in the config file and retry.' %output_path)
             self.Close(True)

    def helpButton(self,event):
        """
        Opens Instructions
        """
        wx.MessageBox('1. Use the slider to select a frame in the entire video. \n\n2. Click Grab a Frame button to save the specific frame.\n\n3. The frame will be saved in the current working directory. \n Click OK to continue', 'Instructions to use!', wx.OK | wx.ICON_INFORMATION)

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

class MatplotPanel(wx.Panel):
    def __init__(self, parent,config):
        wx.Panel.__init__(self, parent,-1,size=(100,100))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)

def show(config):
    import imageio
    imageio.plugins.ffmpeg.download()
    app = wx.App()
    frame = MainFrame(None,config).Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
