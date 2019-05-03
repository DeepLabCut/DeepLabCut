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
import pandas as pd
import argparse
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.create_project import add
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

    def __init__(self, parent,config,video,shuffle,Dataframe,gui_size,**kwargs):
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
#        im = io.imread(img)
        norm = mcolors.Normalize(vmin=np.min(img), vmax=np.max(img))
        ticks = np.linspace(np.min(img),np.max(img),len(bodyparts))[::-1]
        return norm, ticks



class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)




class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent,config,video,shuffle,Dataframe,scorer,savelabeled):
# Settting the GUI size and panels design
        displays = (wx.Display(i) for i in range(wx.Display.GetCount())) # Gets the number of displays
        screenSizes = [display.GetGeometry().GetSize() for display in displays] # Gets the size of each display
        index = 0 # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (screenWidth*0.7,screenHeight*0.85)

        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'DeepLabCut2.0 - Manual Outlier Frame Extraction',
                            size = wx.Size(self.gui_size), pos = wx.DefaultPosition, style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")

        self.SetSizeHints(wx.Size(self.gui_size)) #  This sets the minimum size of the GUI. It can scale now!
        
###################################################################################################################################################
# Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!
        topSplitter = wx.SplitterWindow(self)

        self.image_panel = ImagePanel(topSplitter, config,video,shuffle,Dataframe,self.gui_size)
        self.widget_panel = WidgetPanel(topSplitter)
        
        topSplitter.SplitHorizontally(self.image_panel, self.widget_panel,sashPosition=self.gui_size[1]*0.83)#0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

###################################################################################################################################################
# Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        
        self.load_button_sizer = wx.BoxSizer(wx.VERTICAL)
        self.help_button_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        self.help_button_sizer.Add(self.help , 1, wx.ALL, 15)
#        widgetsizer.Add(self.help , 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)

        widgetsizer.Add(self.help_button_sizer,1,wx.ALL,0)
        
        self.grab = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Grab Frames")
        widgetsizer.Add(self.grab , 1, wx.ALL, 15)
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.grab.Enable(True)

        widgetsizer.AddStretchSpacer(5)
        self.slider = wx.Slider(self.widget_panel, id=wx.ID_ANY, value = 0, minValue=0, maxValue=1,size=(200, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        widgetsizer.Add(self.slider,1, wx.ALL,5)
        self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
        
        widgetsizer.AddStretchSpacer(5)
        self.start_frames_sizer = wx.BoxSizer(wx.VERTICAL)
        self.end_frames_sizer = wx.BoxSizer(wx.VERTICAL)

        self.start_frames_sizer.AddSpacer(15)
#        self.startFrame = wx.SpinCtrl(self.widget_panel, value='0', size=(100, -1), min=0, max=120)
        self.startFrame = wx.SpinCtrl(self.widget_panel, value='0', size=(100, -1))#,style=wx.SP_VERTICAL)
        self.startFrame.Enable(False)
        self.start_frames_sizer.Add(self.startFrame,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        start_text = wx.StaticText(self.widget_panel, label='Start Frame Index')
        self.start_frames_sizer.Add(start_text,1, wx.EXPAND|wx.ALIGN_LEFT,15)
        self.checkBox = wx.CheckBox(self.widget_panel, id=wx.ID_ANY,label = 'Range of frames')
        self.checkBox.Bind(wx.EVT_CHECKBOX,self.activate_frame_range)
        self.start_frames_sizer.Add(self.checkBox,1, wx.EXPAND|wx.ALIGN_LEFT,15)
#        
        self.end_frames_sizer.AddSpacer(15)
        self.endFrame = wx.SpinCtrl(self.widget_panel, value='1', size=(160, -1))#, min=1, max=120)
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

        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        
        
# Variables initialization
        self.numberFrames = 0
        self.currFrame = 0
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.drs = []
        self.extract_range_frame = False
        self.firstFrame  = 0
        # self.cropping = False

# Read confing file
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
        self.iterationindex=self.cfg['iteration']
        self.cropping = self.cfg['cropping']
        self.video_names = [Path(i).stem for i in self.videos]
        self.config_path = Path(config)
        self.video_source = Path(video).resolve()
        self.shuffle = shuffle
        self.Dataframe = Dataframe
        self.scorer = scorer
        self.savelabeled = savelabeled
        
# Read the video file
        self.vid = cv2.VideoCapture(str(self.video_source))
        self.videoPath = os.path.dirname(self.video_source)
        self.filename = Path(self.video_source).name
        self.numberFrames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.strwidth = int(np.ceil(np.log10(self.numberFrames)))
# Set the values of slider and range of frames
        self.startFrame.SetMax(self.numberFrames-1)
        self.slider.SetMax(self.numberFrames-1)
        self.endFrame.SetMax(self.numberFrames-1)
        self.startFrame.Bind(wx.EVT_SPINCTRL,self.updateSlider)#wx.EVT_SPIN
# Set the status bar
        self.statusbar.SetStatusText('Working on video: {}'.format(os.path.split(str(self.video_source))[-1]))
# Adding the video file to the config file.
        if  not (str(self.video_source.stem) in self.video_names) :
            add.add_new_videos(self.config_path,[self.video_source])

        self.filename = Path(self.video_source).name
        self.update()
        self.plot_labels()
        self.widget_panel.Layout()
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
        self.startFrame.SetValue(self.slider.GetValue())
        self.axes.clear()
        self.figure.delaxes(self.figure.axes[1])
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.currFrame = (self.slider.GetValue())
        self.update()
        self.plot_labels()
    
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

            
    def OnSliderScroll(self, event):
        """
        Slider to scroll through the video
        """
        self.axes.clear()
        self.figure.delaxes(self.figure.axes[1])
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.currFrame = (self.slider.GetValue())
        self.startFrame.SetValue(self.currFrame)
        self.update()
        self.plot_labels()
    

    def update(self):
        """
        Updates the image with the current slider index
        """
        self.grab.Enable(True)
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.figure,self.axes,self.canvas = self.image_panel.getfigure()
        self.vid.set(1,self.currFrame)
        ret, frame = self.vid.read()
        frame=img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if ret:
            if self.cropping:
                self.coords = (self.cfg['x1'],self.cfg['x2'],self.cfg['y1'], self.cfg['y2'])
                frame=frame[int(self.coords[2]):int(self.coords[3]),int(self.coords[0]):int(self.coords[1]),:]
            else:
                self.coords = None
            self.ax = self.axes.imshow(frame,cmap = self.colormap)
            self.axes.set_title(str(str(self.currFrame)+"/"+str(self.numberFrames-1) +" "+ self.filename))
            self.figure.canvas.draw()
        else:
            print("Invalid frame")
        
    def chooseFrame(self):
        ret, frame = self.vid.read()
        frame=img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.cropping:
            self.coords = (self.cfg['x1'],self.cfg['x2'],self.cfg['y1'], self.cfg['y2'])
            frame=frame[int(self.coords[2]):int(self.coords[3]),int(self.coords[0]):int(self.coords[1]),:]
        else:
            self.coords = None
        fname = Path(self.filename)
        output_path = self.config_path.parents[0] / 'labeled-data' / fname.stem

        self.machinefile = os.path.join(str(output_path),'machinelabels-iter'+str(self.iterationindex)+'.h5')
        name = str(fname.stem)
        DF = self.Dataframe.ix[[self.currFrame]]
        DF.index=[os.path.join('labeled-data',name,"img"+str(index).zfill(self.strwidth)+".png") for index in DF.index]
        img_name = str(output_path) +'/img'+str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames)))) + '.png'
        labeled_img_name = str(output_path) +'/img'+str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames)))) + 'labeled.png'

# Check for it output path and a machine label file exist
        if output_path.exists() and Path(self.machinefile).is_file():
            cv2.imwrite(img_name, frame)
            if self.savelabeled:
                self.figure.savefig(labeled_img_name,bbox_inches='tight')
            Data = pd.read_hdf(self.machinefile,'df_with_missing')
            DataCombined = pd.concat([Data,DF])
            DataCombined = DataCombined[~DataCombined.index.duplicated(keep='first')]
            DataCombined.to_hdf(self.machinefile,key='df_with_missing',mode='w')
            DataCombined.to_csv(os.path.join(str(output_path),'machinelabels.csv'))
# If machine label file does not exist then create one
        elif output_path.exists() and not(Path(self.machinefile).is_file()):
            if self.savelabeled:
                self.figure.savefig(labeled_img_name,bbox_inches='tight')
            cv2.imwrite(img_name, frame)
            DF.to_hdf(self.machinefile,key='df_with_missing',mode='w')
            DF.to_csv(os.path.join(str(output_path), "machinelabels.csv"))
        else:
            print("%s path not found. Please make sure that the video was added to the config file using the function 'deeplabcut.add_new_videos'.Quitting for now!" %output_path)
            self.Destroy()
        
    def grabFrame(self,event):
        """
        Extracts the frame and saves in the current directory
        """

        if self.extract_range_frame == True:
            num_frames_extract = self.endFrame.GetValue()
            for i in range(self.currFrame,self.currFrame+num_frames_extract):
                self.currFrame = i
                self.vid.set(1,self.currFrame)
                self.chooseFrame()
        else:
            self.vid.set(1,self.currFrame)
            self.chooseFrame()


    def plot_labels(self):
        """
        Plots the labels of the analyzed video
        """
        self.vid.set(1,self.currFrame)
        ret, frame = self.vid.read()
        self.norm,self.colorIndex = self.image_panel.getColorIndices(frame,self.bodyparts)
        if ret:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        wx.MessageBox("1. Use the checkbox 'Crop?' at the bottom left if you need to crop the frame. In this case use the left mouse button to draw a box corresponding to the region of interest. Click the 'Set cropping parameters' button to add the video with the chosen crop parameters to the config file.\n\n2. Use the slider to select a frame in the entire video. \n\n3. Click Grab Frames button to save the specific frame.\n\n4. In events where you need to extract a range frames, then use the checkbox 'Range of frames' to select the starting frame index and the number of frames to extract. \n Click the update button to see the frame. Click Grab Frames to select the range of frames. \n Click OK to continue", 'Instructions to use!', wx.OK | wx.ICON_INFORMATION)

def show(config,video,shuffle,Dataframe,scorer,savelabeled):
    import imageio
    imageio.plugins.ffmpeg.download()
    app = wx.App()
    frame = MainFrame(None,config,video,shuffle,Dataframe,scorer,savelabeled).Show()
    app.MainLoop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config','video','shuffle','Dataframe','scorer','savelabeled')
    cli_args = parser.parse_args()