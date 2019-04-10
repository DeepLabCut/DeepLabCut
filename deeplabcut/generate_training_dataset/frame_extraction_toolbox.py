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
import numpy as np
from pathlib import Path
import argparse
from deeplabcut.utils import auxiliaryfunctions

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(wx.Panel):

    def __init__(self, parent,config,gui_size,**kwargs):
        h=gui_size[0]/2
        w=gui_size[1]/3
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER,size=(h,w))

        self.figure = matplotlib.figure.Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return(self.figure,self.axes,self.canvas)

    def getColorIndices(self,img,bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        im = io.imread(img)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(im))
        ticks = np.linspace(0,np.max(im),len(bodyparts))[::-1]
        return norm, ticks



class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)

class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent,config):
# Settting the GUI size and panels design
        displays = (wx.Display(i) for i in range(wx.Display.GetCount())) # Gets the number of displays
        screenSizes = [display.GetGeometry().GetSize() for display in displays] # Gets the size of each display
        index = 0 # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (screenWidth*0.7,screenHeight*0.85)

        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'DeepLabCut2.0 - Manual Frame Extraction',
                            size = wx.Size(self.gui_size), pos = wx.DefaultPosition, style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")

        self.SetSizeHints(wx.Size(self.gui_size)) #  This sets the minimum size of the GUI. It can scale now!
        
###################################################################################################################################################
# Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!
        topSplitter = wx.SplitterWindow(self)

        self.image_panel = ImagePanel(topSplitter, config,self.gui_size)
        self.widget_panel = WidgetPanel(topSplitter)
        
        topSplitter.SplitHorizontally(self.image_panel, self.widget_panel,sashPosition=self.gui_size[1]*0.83)#0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

###################################################################################################################################################
# Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)

        self.load = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Load Video")
        widgetsizer.Add(self.load , 1, wx.ALL, 15)
        self.load.Bind(wx.EVT_BUTTON, self.browseDir)
        
        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        widgetsizer.Add(self.help , 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)

        self.grab = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Grab Frames")
        widgetsizer.Add(self.grab , 1, wx.ALL, 15)
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.grab.Enable(False)

        widgetsizer.AddStretchSpacer(5)
        self.slider = wx.Slider(self.widget_panel, id=wx.ID_ANY, value = 0, minValue=0, maxValue=1,size=(200, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        widgetsizer.Add(self.slider,1, wx.ALL,5)
        self.slider.Hide()
        
        widgetsizer.AddStretchSpacer(5)
        self.start_frames_sizer = wx.BoxSizer(wx.VERTICAL)
        self.end_frames_sizer = wx.BoxSizer(wx.VERTICAL)

        self.start_frames_sizer.AddSpacer(15)
        self.startFrame = wx.SpinCtrl(self.widget_panel, value='0', size=(100, -1), min=0, max=120)
        self.startFrame.Bind(wx.EVT_SPINCTRL,self.updateSlider)
        self.startFrame.Enable(False)
        self.start_frames_sizer.Add(self.startFrame,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        start_text = wx.StaticText(self.widget_panel, label='Start Frame Index')
        self.start_frames_sizer.Add(start_text,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        self.checkBox = wx.CheckBox(self.widget_panel, id=wx.ID_ANY,label = 'Range of frames')
        self.checkBox.Bind(wx.EVT_CHECKBOX,self.activate_frame_range)
        self.start_frames_sizer.Add(self.checkBox,1, wx.EXPAND|wx.ALIGN_LEFT,15)
#        
        self.end_frames_sizer.AddSpacer(15)
        self.endFrame = wx.SpinCtrl(self.widget_panel, value='1', size=(160, -1), min=1, max=120)
        self.endFrame.Enable(False)
        self.end_frames_sizer.Add(self.endFrame,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        end_text = wx.StaticText(self.widget_panel, label='Number of Frames')
        self.end_frames_sizer.Add(end_text,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        self.updateFrame = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Update")
        self.end_frames_sizer.Add(self.updateFrame,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        self.updateFrame.Bind(wx.EVT_BUTTON, self.updateSlider)
        self.updateFrame.Enable(False)
        
        widgetsizer.Add(self.start_frames_sizer,1,wx.ALL,0)
        widgetsizer.AddStretchSpacer(5)
        widgetsizer.Add(self.end_frames_sizer,1,wx.ALL,0)
        widgetsizer.AddStretchSpacer(15)
        
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit , 1, wx.ALL, 15)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)
        self.quit.Enable(True)

# Hiding these widgets and show them once the video is loaded
        self.start_frames_sizer.ShowItems(show=False)
        self.end_frames_sizer.ShowItems(show=False)

        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        self.widget_panel.Layout()
        
# Variables initialization
        self.numberFrames = 0
        self.currFrame = 0
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.drs = []
        self.cfg = auxiliaryfunctions.read_config(config)
        self.Task = self.cfg['Task']
        self.start = self.cfg['start']
        self.stop = self.cfg['stop']
        self.date = self.cfg['date']
        self.trainFraction = self.cfg['TrainingFraction']
        self.trainFraction = self.trainFraction[0]
        self.videos = self.cfg['video_sets'].keys()
        self.bodyparts = self.cfg['bodyparts']
        self.colormap = plt.get_cmap(self.cfg['colormap'])
        self.colormap = self.colormap.reversed()
        self.markerSize = self.cfg['dotsize']
        self.alpha = self.cfg['alphavalue']
        self.video_names = [Path(i).stem for i in self.videos]
        self.config_path = Path(config)
        self.extract_range_frame = False
        self.extract_from_analyse_video = False

    def quitButton(self, event):
        """
        Quits the GUI
        """
        self.statusbar.SetStatusText("")
        dlg = wx.MessageDialog(None,"Are you sure?", "Quit!",wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            print("Quitting for now!")
            self.Destroy()

    def updateSlider(self,event):
        self.slider.SetValue(self.startFrame.GetValue())
        self.currFrame = (self.slider.GetValue())
        if self.extract_from_analyse_video == True:
            self.figure.delaxes(self.figure.axes[1])
            self.plot_labels()
        self.update()
    
    def activate_frame_range(self,event):
        """
        Activates the frame range boxes
        """
        self.checkSlider = event.GetEventObject()
        if self.checkSlider.GetValue() == True:
            self.extract_range_frame = True
            self.startFrame.Enable(True)
            self.startFrame.SetValue(self.slider.GetValue())
            self.endFrame.Enable(True)
            self.updateFrame.Enable(True)
            self.grab.Enable(False)
        else:
            self.extract_range_frame = False
            self.startFrame.Enable(False)
            self.endFrame.Enable(False)
            self.updateFrame.Enable(False)
            self.grab.Enable(True)
    
    
    def line_select_callback(self,eclick, erelease):
        'eclick and erelease are the press and release events'
        self.new_x1, self.new_y1 = eclick.xdata, eclick.ydata
        self.new_x2, self.new_y2 = erelease.xdata, erelease.ydata

    
    
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

        videosource = self.video_source
        self.x1 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[0])
        self.x2 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[1])
        self.y1 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[2])
        self.y2 = int(self.cfg['video_sets'][videosource]['crop'].split(',')[3])

        if self.cropping==True:
# Select ROI of interest by drawing a rectangle
            self.cid = RectangleSelector(self.axes, self.line_select_callback,drawtype='box', useblit=False,button=[1], minspanx=5, minspany=5,spancoords='pixels',interactive=True)
            self.canvas.mpl_connect('key_press_event', self.cid)
            
    def OnSliderScroll(self, event):
        """
        Slider to scroll through the video
        """
        self.axes.clear()
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.currFrame = (self.slider.GetValue())
        self.startFrame.SetValue(self.currFrame)
        self.update()
    
    def is_crop_ok(self,event):
        """
        Checks if the cropping is ok
        """
        
        self.grab.SetLabel("Grab Frames")
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.slider.Show()
        self.start_frames_sizer.ShowItems(show=True)
        self.end_frames_sizer.ShowItems(show=True)
        self.widget_panel.Layout()
        self.slider.SetMax(self.numberFrames)
        self.startFrame.SetMax(self.numberFrames-1)
        self.endFrame.SetMax(self.numberFrames)
        self.x1 = int(self.new_x1)
        self.x2 = int(self.new_x2)
        self.y1 = int(self.new_y1)
        self.y2 = int(self.new_y2)
        self.canvas.mpl_disconnect(self.cid)
        self.axes.clear()
        self.currFrame = (self.slider.GetValue())
        self.update()
# Update the config.yaml file
        self.cfg['video_sets'][self.video_source] = {'crop': ', '.join(map(str, [self.x1, self.x2, self.y1, self.y2]))}
        auxiliaryfunctions.write_config(self.config_path,self.cfg)

        
    def browseDir(self, event):
        """
        Show the File Dialog and ask the user to select the video file
        """

        self.statusbar.SetStatusText("Looking for a video to start extraction..")
        dlg = wx.FileDialog(self, "SELECT A VIDEO", os.getcwd(), "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.video_source_original = dlg.GetPath()
            self.video_source = str(Path(self.video_source_original).resolve())
            
            self.load.Enable(False)
        else:
            pass
            dlg.Destroy()
            self.Close(True)
        dlg.Destroy()
        selectedvideo = Path(self.video_source)
        
        self.statusbar.SetStatusText('Working on video: {}'.format(os.path.split(str(selectedvideo))[-1]))

        if  str(selectedvideo.stem) in self.video_names :
            self.grab.Enable(True)
            self.vid = cv2.VideoCapture(self.video_source)
            self.videoPath = os.path.dirname(self.video_source)
            self.filename = Path(self.video_source).name
            self.numberFrames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
# Checks if the video is corrupt.
            if not self.vid.isOpened():
                msg = wx.MessageBox('Invalid Video file!Do you want to retry?', 'Error!', wx.YES_NO | wx.ICON_WARNING)
                if msg == 2:
                    self.load.Enable(True)
                    MainFrame.browseDir(self, event)
                else:
                    self.Destroy()
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.update()

            cropMsg = wx.MessageBox("Do you want to crop the frames?",'Want to crop?',wx.YES_NO|wx.ICON_INFORMATION)
            if cropMsg == 2:
                self.cropping = True
                self.grab.SetLabel("Set cropping parameters")
                self.grab.Bind(wx.EVT_BUTTON, self.is_crop_ok)
                self.widget_panel.Layout()
                self.basefolder = 'data-' + self.Task + '/'
                MainFrame.CheckCropping(self)
            else:
                self.cropping = False
                self.slider.Show()
                self.start_frames_sizer.ShowItems(show=True)
                self.end_frames_sizer.ShowItems(show=True)
                self.widget_panel.Layout()
                self.slider.SetMax(self.numberFrames-1)
                self.startFrame.SetMax(self.numberFrames-1)
                self.endFrame.SetMax(self.numberFrames-1)
       
        else:
            wx.MessageBox('Video file is not in config file. Use add function to add this video in the config file and retry!', 'Error!', wx.OK | wx.ICON_WARNING)
            self.Close(True)

    def update(self):
        """
        Updates the image with the current slider index
        """
        self.grab.Enable(True)
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.figure,self.axes,self.canvas = self.image_panel.getfigure()
        self.vid.set(1,self.currFrame)
        ret, frame = self.vid.read()
        if ret:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax = self.axes.imshow(frame)
        self.axes.set_title(str(str(self.currFrame)+"/"+str(self.numberFrames-1) +" "+ self.filename))
        self.figure.canvas.draw()

    def chooseFrame(self):
        ret, frame = self.vid.read()
        fname = Path(self.filename)
        output_path = self.config_path.parents[0] / 'labeled-data' / fname.stem
        
        if output_path.exists() :
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame=img_as_ubyte(frame)
            img_name = str(output_path) +'/img'+str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames)))) + '.png'
            if self.cropping:
                crop_img = frame[self.y1:self.y2, self.x1:self.x2]
                cv2.imwrite(img_name, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(img_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            print("%s path not found. Please make sure that the video was added to the config file using the function 'deeplabcut.add_new_videos'." %output_path)
    
    def grabFrame(self,event):
        """
        Extracts the frame and saves in the current directory
        """
        num_frames_extract = self.endFrame.GetValue()
        for i in range(self.currFrame,self.currFrame+num_frames_extract):
            self.currFrame = i
            self.vid.set(1,self.currFrame)
            self.chooseFrame()
        self.vid.set(1,self.currFrame)
        self.chooseFrame()

    def plot_labels(self):
        """
        Plots the labels of the analyzed video
        """
        self.vid.set(1,self.currFrame)
        ret, frame = self.vid.read()
        if ret:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.norm = mcolors.Normalize(vmin=np.min(frame), vmax=np.max(frame))
            self.colorIndex = np.linspace(np.min(frame),np.max(frame),len(self.bodyparts))
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = self.figure.colorbar(self.ax, cax=cax,spacing='proportional', ticks=self.colorIndex)
            cbar.set_ticklabels(self.bodyparts)
            for bpindex, bp in enumerate(self.bodyparts):
                color = self.colormap(self.norm(self.colorIndex[bpindex]))
                self.points = [self.Dataframe[self.scorer][bp]['x'].values[self.currFrame],self.Dataframe[self.scorer][bp]['y'].values[self.currFrame],1.0]
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = color , alpha=self.alpha)]
                self.axes.add_patch(circle[0])
            self.figure.canvas.draw()
        
    def helpButton(self,event):
        """
        Opens Instructions
        """
        wx.MessageBox('1. Use the Load Video button to load a video. Use the slider to select a frame in the entire video. The number mentioned on the top of the slider represents the frame index. \n\n2. Click Grab Frames button to save the specific frame.\n\n3. In events where you need to extract a range of frames, then use the checkbox Range of frames to select the start frame index and number of frames to extract. Click the update button to see the start frame index. Click Grab Frames to select the range of frames. \n\n Click OK to continue', 'Instructions to use!', wx.OK | wx.ICON_INFORMATION)

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
