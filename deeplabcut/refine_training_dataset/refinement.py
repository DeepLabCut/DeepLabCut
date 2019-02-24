"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
T Nath, nath@rowland.harvard.edu
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
"""

import sys
import wx
import os
import pandas as pd
import numpy as np
#from skimage import io
import PIL
import glob
import platform
import wx.lib.scrolledpanel as SP

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import os.path
import argparse
import matplotlib
from deeplabcut.utils import auxiliaryfunctions
from skimage import io

from pathlib import Path
from deeplabcut.refine_training_dataset import auxfun_drag
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas


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
        return(self.figure)

    def drawplot(self,img,img_name,itr,index,threshold,bodyparts,cmap,preview):
        im = io.imread(img)
        ax = self.axes.imshow(im,cmap=cmap)
        divider = make_axes_locatable(self.axes)
        colorIndex = np.linspace(np.min(im),np.max(im),len(bodyparts))
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.figure.colorbar(ax, cax=cax,spacing='proportional', ticks=colorIndex)
        cbar.set_ticklabels(bodyparts[::-1])
        if preview == False:
            self.axes.set_title(str(str(itr)+"/"+str(len(index)-1) +" "+ str(Path(index[itr]).stem) + " "+ " Threshold chosen is: " + str("{0:.2f}".format(threshold))))
        else:
            self.axes.set_title(str(str(itr)+"/"+str(len(index)-1) +" "+ str(Path(index[itr]).stem)))
        self.figure.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas)
        return(self.figure,self.axes,self.canvas,self.toolbar)

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


class ScrollPanel(SP.ScrolledPanel):
    def __init__(self, parent):
#        SP.ScrolledPanel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER, **kwargs)
        SP.ScrolledPanel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)
#        self.parent = parent
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
#        self.SetupScrolling(scroll_x=True, scrollToTop=False)
        self.Layout()
    def on_focus(self,event):
        pass

    def addCheckBoxSlider(self,bodyparts,fileIndex,markersize):
        """
        Adds checkbox and a slider
        """
        self.choiceBox = wx.BoxSizer(wx.VERTICAL)

        self.slider = wx.Slider(self, -1, markersize, 1, markersize*3,size=(250, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        self.slider.Enable(False)
        self.checkBox = wx.CheckBox(self, id=wx.ID_ANY,label = 'Adjust marker size.')
        self.choiceBox.Add(self.slider, 0, wx.ALL, 5 )
        self.choiceBox.Add(self.checkBox, 0, wx.ALL, 5 )
        self.SetSizerAndFit(self.choiceBox)
        self.Layout()
        return(self.choiceBox,self.slider,self.checkBox)

    def clearBoxer(self):
        self.choiceBox.Clear(True)

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

        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'DeepLabCut2.0 - Refinement ToolBox',
                            size = wx.Size(self.gui_size), pos = wx.DefaultPosition, style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

        self.SetSizeHints(wx.Size(self.gui_size)) #  This sets the minimum size of the GUI. It can scale now!
###################################################################################################################################################

# Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!

        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)

        self.image_panel = ImagePanel(vSplitter, config,self.gui_size)
        self.choice_panel = ScrollPanel(vSplitter)
#        self.choice_panel.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
#        self.choice_panel.SetupScrolling(scroll_x=True, scrollToTop=False)
        vSplitter.SplitVertically(self.image_panel,self.choice_panel, sashPosition=self.gui_size[0]*0.8)
        vSplitter.SetSashGravity(1)
        self.widget_panel = WidgetPanel(topSplitter)
        topSplitter.SplitHorizontally(vSplitter, self.widget_panel,sashPosition=self.gui_size[1]*0.83)#0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

###################################################################################################################################################
# Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        self.load = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Load labels")
        widgetsizer.Add(self.load , 1, wx.ALL, 15)
        self.load.Bind(wx.EVT_BUTTON, self.browseDir)

        self.prev = wx.Button(self.widget_panel, id=wx.ID_ANY, label="<<Previous")
        widgetsizer.Add(self.prev , 1, wx.ALL, 15)
        self.prev.Bind(wx.EVT_BUTTON, self.prevImage)
        self.prev.Enable(False)

        self.next = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Next>>")
        widgetsizer.Add(self.next , 1, wx.ALL, 15)
        self.next.Bind(wx.EVT_BUTTON, self.nextImage)
        self.next.Enable(False)

        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        widgetsizer.Add(self.help , 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)
        self.help.Enable(True)
#
        self.zoom = wx.ToggleButton(self.widget_panel, label="Zoom")
        widgetsizer.Add(self.zoom , 1, wx.ALL, 15)
        self.zoom.Bind(wx.EVT_TOGGLEBUTTON, self.zoomButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.zoom.Enable(False)

        self.home = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Home")
        widgetsizer.Add(self.home , 1, wx.ALL,15)
        self.home.Bind(wx.EVT_BUTTON, self.homeButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.home.Enable(False)

        self.pan = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Pan")
        widgetsizer.Add(self.pan , 1, wx.ALL, 15)
        self.pan.Bind(wx.EVT_TOGGLEBUTTON, self.panButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.pan.Enable(False)

        self.save = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Save")
        widgetsizer.Add(self.save , 1, wx.ALL, 15)
        self.save.Bind(wx.EVT_BUTTON, self.saveDataSet)
        self.save.Enable(False)

        widgetsizer.AddStretchSpacer(15)
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit , 1, wx.ALL|wx.ALIGN_RIGHT, 15)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)

#        widgetsizer.AddStretchSpacer(15)
#        self.adjustLabelCheck = wx.CheckBox(self.widget_panel,id=wx.ID_ANY,  label = 'Adjust original labels?')
#        widgetsizer.Add(self.adjustLabelCheck , 1, wx.ALL, 15)
#        self.adjustLabelCheck.Bind(wx.EVT_CHECKBOX,self.adjustLabel)

        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        self.widget_panel.Layout()

###############################################################################################################################
# Variable initialization
        self.currentDirectory = os.getcwd()
        self.index = []
        self.iter = []
        self.threshold = []
        self.file = 0
        self.updatedCoords = []
        self.drs = []
        cfg = auxiliaryfunctions.read_config(config)
        self.humanscorer = cfg['scorer']
        self.move2corner = cfg['move2corner']
        self.center = cfg['corner2move2']
        self.colormap = plt.get_cmap(cfg['colormap'])
        self.colormap = self.colormap.reversed()
        self.markerSize = cfg['dotsize']
        self.alpha = cfg['alphavalue']
        self.iterationindex = cfg['iteration']
        self.project_path=cfg['project_path']
        self.bodyparts = cfg['bodyparts']
        self.threshold = 0.1
        self.img_size = (10,6)# (imgW, imgH)  # width, height in inches.
        self.preview = False
# ###########################################################################
# functions for button responses
# ###########################################################################
    # BUTTONS FUNCTIONS FOR HOTKEYS
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)
        elif event.GetKeyCode() == wx.WXK_LEFT:
            self.prevImage(event=None)

    def closewindow(self, event):
        self.Destroy()

    def homeButton(self,event):
        self.toolbar.home()
        MainFrame.updateZoomPan(self)
        self.zoom.SetValue(False)
        self.pan.SetValue(False)
        self.statusbar.SetStatusText("")


    def panButton(self,event):
        if self.pan.GetValue() == True:
            self.toolbar.pan()
            self.statusbar.SetStatusText("Pan On")
            self.zoom.SetValue(False)
        else:
            self.toolbar.pan()
            self.statusbar.SetStatusText("Pan Off")


    def zoomButton(self, event):
        if self.zoom.GetValue() == True:
            self.toolbar.zoom()
            self.statusbar.SetStatusText("Zoom On")
            self.pan.SetValue(False)
        else:
            self.toolbar.zoom()
            self.statusbar.SetStatusText("Zoom Off")

    def activateSlider(self,event):
        """
        Activates the slider to increase the markersize
        """
        self.checkSlider = event.GetEventObject()
        if self.checkSlider.GetValue() == True:
            self.activate_slider = True
            self.slider.Enable(True)
            MainFrame.updateZoomPan(self)
        else:
            self.slider.Enable(False)

    def OnSliderScroll(self, event):
        """
        Adjust marker size for plotting the annotations
        """
        self.markerSize = self.slider.GetValue()
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
        self.updatedCoords = []

        img_name = Path(self.index[self.iter]).name
        self.axes.clear()
        self.figure.delaxes(self.figure.axes[1])
        self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)
        MainFrame.plot(self,self.img)

    def browseDir(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """

        fname = str('machinelabels-iter'+str(self.iterationindex)+'.h5')
        self.statusbar.SetStatusText("Looking for a folder to start refining...")
        cwd = os.path.join(os.getcwd(),'labeled-data')
#        dlg = wx.FileDialog(self, "Choose the machinelabels file for current iteration.",cwd, "",wildcard=fname,style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        print(platform.system())
        if platform.system()=='Darwin':  
            dlg = wx.FileDialog(self, "Choose the machinelabels file for current iteration.",cwd, fname ,wildcard="(*.h5)|*.h5",style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) 
        else:
            dlg = wx.FileDialog(self, "Choose the machinelabels file for current iteration.",cwd, "",wildcard=fname,style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            self.data_file = dlg.GetPath()
            self.dir = str(Path(self.data_file).parents[0])
            self.fileName = str(Path(self.data_file).stem)
            self.load.Enable(False)
            self.next.Enable(True)
            self.save.Enable(True)
            self.zoom.Enable(True)
            self.pan.Enable(True)
            self.home.Enable(True)
            self.quit.Enable(True)
        else:
            dlg.Destroy()
            self.Destroy()
        dlg.Destroy()

        try:
            self.dataname = str(self.data_file)

        except:
            print("No machinelabels file found!")
            self.Destroy()
        self.statusbar.SetStatusText('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        self.preview = True
        self.iter = 0

        if os.path.isfile(self.dataname):
            self.Dataframe = pd.read_hdf(self.dataname,'df_with_missing')
            self.Dataframe.sort_index(inplace =True)
            self.scorer = self.Dataframe.columns.get_level_values(0)[0]

            # bodyParts = self.Dataframe.columns.get_level_values(1)
            # _, idx = np.unique(bodyParts, return_index=True)
            # self.num_joints = len(self.bodyparts)
            # self.bodyparts =  bodyParts[np.sort(idx)]
            self.index = list(self.Dataframe.iloc[:,0].index)
# Reading images

            self.img = os.path.join(self.project_path,self.index[self.iter])
            img_name = Path(self.img).name
            self.norm,self.colorIndex = self.image_panel.getColorIndices(self.img,self.bodyparts)
# Adding Slider and Checkbox

            self.choiceBox,self.slider,self.checkBox = self.choice_panel.addCheckBoxSlider(self.bodyparts,self.file,self.markerSize)
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.checkBox.Bind(wx.EVT_CHECKBOX,self.activateSlider)
            self.slider.Enable(False)
# Show image
# Setting axis title:dont want to show the threshold as it is not selected yet.
            self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)

            instruction = wx.MessageBox('1. Enter the likelihood threshold. \n\n2. Each prediction will be shown with a unique color. \n All the data points above the threshold will be marked as circle filled with a unique color. All the data points below the threshold will be marked with a hollow circle. \n\n3. Enable the checkbox to adjust the marker size. \n\n4.  Hover your mouse over data points to see the labels and their likelihood. \n\n5. Left click and drag to move the data points. \n\n6. Right click on any data point to remove it. Be careful, you cannot undo this step. \n Click once on the zoom button to zoom-in the image.The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full;default view. \n\n7. When finished click \'Save\' to save all the changes. \n\n8. Click OK to continue', 'User instructions', wx.OK | wx.ICON_INFORMATION)

            if instruction == 4 :
                """
                If ok is selected then the image is updated with the thresholded value of the likelihood
                """
                textBox = wx.TextEntryDialog(self, "Select the likelihood threshold",caption = "Enter the threshold",value="0.1")
                textBox.ShowModal()
                self.threshold = float(textBox.GetValue())
                textBox.Destroy()
                self.img = os.path.join(self.project_path,self.index[self.iter])
                img_name = Path(self.img).name
                self.axes.clear()
                self.preview = False
                self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)
                MainFrame.plot(self,self.img)
                MainFrame.saveEachImage(self)
            else:
                self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)
                MainFrame.plot(self,self.img)
                MainFrame.saveEachImage(self)

        else:
            msg = wx.MessageBox('No Machinelabels file found! Want to retry?', 'Error!', wx.YES_NO | wx.ICON_WARNING)
            if msg == 2:
                self.load.Enable(True)
                self.next.Enable(False)
                self.save.Enable(False)

    def nextImage(self, event):
        """
        Reads the next image and enables the user to move the annotations
        """
#  Checks for the last image and disables the Next button
        if len(self.index) - self.iter == 1:
            self.next.Enable(False)
            return
        self.prev.Enable(True)

# Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)

        MainFrame.saveEachImage(self)
        self.statusbar.SetStatusText('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))

        self.iter = self.iter + 1

        if len(self.index) > self.iter:
            self.updatedCoords = []
            self.img = os.path.join(self.project_path,self.index[self.iter])
            img_name = Path(self.img).name

# Plotting
            self.axes.clear()
            self.figure.delaxes(self.figure.axes[1]) # Removes the axes corresponding to the colorbar
            self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)

            im = io.imread(self.img)
            if np.max(im) == 0:
                msg = wx.MessageBox('Invalid image. Click Yes to remove', 'Error!', wx.YES_NO | wx.ICON_WARNING)
                if msg == 2:
                    self.Dataframe = self.Dataframe.drop(self.index[self.iter])
                    self.index = list(self.Dataframe.iloc[:,0].index)
                self.iter = self.iter - 1

                self.img = os.path.join(self.project_path,self.index[self.iter])
                img_name = Path(self.img).name

                self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)

            MainFrame.plot(self,self.img)
        else:
            self.next.Enable(False)
        MainFrame.saveEachImage(self)

    def prevImage(self, event):
        """
        Checks the previous Image and enables user to move the annotations.
        """

        MainFrame.saveEachImage(self)

# Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)

        self.statusbar.SetStatusText('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        self.next.Enable(True)
        self.iter = self.iter - 1

        # Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.prev.Enable(False)

        if self.iter >= 0:
            self.updatedCoords = []
# Reading Image
            self.img = os.path.join(self.project_path,self.index[self.iter])
            img_name = Path(self.img).name

# Plotting
            self.axes.clear()
            self.figure.delaxes(self.figure.axes[1]) # Removes the axes corresponding to the colorbar
            self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.threshold,self.bodyparts,self.colormap,self.preview)

            MainFrame.plot(self,self.img)
        else:
            self.prev.Enable(False)
        MainFrame.saveEachImage(self)

    def quitButton(self, event):
        """
        Quits the GUI
        """
        self.statusbar.SetStatusText("")
        dlg = wx.MessageDialog(None,"Are you sure?", "Quit!",wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            print("Closing... The refined labels are stored in a subdirectory under labeled-data. Use the function 'merge_datasets' to augment the training dataset, and then re-train a network using create_training_dataset followed by train_network!")
            self.Destroy()
        else:
            self.save.Enable(True)
        
    def helpButton(self,event):
        """
        Opens Instructions
        """
        self.statusbar.SetStatusText('Help')
# Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        wx.MessageBox('1. Enter the likelihood threshold. \n\n2. All the data points above the threshold will be marked as circle filled with a unique color. All the data points below the threshold will be marked with a hollow circle. \n\n3. Enable the checkbox to adjust the marker size (you will not be able to zoom/pan/home until the next frame). \n\n4. Hover your mouse over data points to see the labels and their likelihood. \n\n5. LEFT click+drag to move the data points. \n\n6. RIGHT click on any data point to remove it. Be careful, you cannot undo this step! \n Click once on the zoom button to zoom-in the image. The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full default view. \n\n7. When finished click \'Save\' to save all the changes. \n\n8. Click OK to continue', 'User instructions', wx.OK | wx.ICON_INFORMATION)


    def onChecked(self, event):
      MainFrame.saveEachImage(self)
      self.cb = event.GetEventObject()
      if self.cb.GetValue() == True:
          self.slider.Enable(True)
      else:
          self.slider.Enable(False)

    def check_labels(self):
        print("Checking labels if they are outside the image")
        for i in self.Dataframe.index:
            image_name = os.path.join(self.project_path,i)
            im = PIL.Image.open(image_name)
            width, height = im.size
            for bpindex, bp in enumerate(self.bodyparts):
                testCondition = self.Dataframe.loc[i,(self.scorer,bp,'x')] > width or self.Dataframe.loc[i,(self.scorer,bp,'x')] < 0 or self.Dataframe.loc[i,(self.scorer,bp,'y')] > height or self.Dataframe.loc[i,(self.scorer,bp,'y')] <0
                if testCondition:
                    print("Found %s outside the image %s.Setting it to NaN" %(bp,i))
                    self.Dataframe.loc[i,(self.scorer,bp,'x')] = np.nan
                    self.Dataframe.loc[i,(self.scorer,bp,'y')] = np.nan
        return(self.Dataframe)

    def saveDataSet(self, event):

        MainFrame.saveEachImage(self)

# Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        self.statusbar.SetStatusText("File saved")

        self.Dataframe = MainFrame.check_labels(self)
        self.Dataframe.columns.set_levels([self.scorer.replace(self.scorer,self.humanscorer)],level=0,inplace=True)
        self.Dataframe = self.Dataframe.drop('likelihood',axis=1,level=2)

        if Path(self.dir,'CollectedData_'+self.humanscorer+'.h5').is_file():
            print("A training dataset file is already found for this video. The refined machine labels are merged to this data!")
            DataU1 = pd.read_hdf(os.path.join(self.dir,'CollectedData_'+self.humanscorer+'.h5'), 'df_with_missing')
# combine datasets Original Col. + corrected machinefiles:
            DataCombined = pd.concat([self.Dataframe,DataU1])
# Now drop redundant ones keeping the first one [this will make sure that the refined machine file gets preference]
            DataCombined = DataCombined[~DataCombined.index.duplicated(keep='first')]
            '''
            if len(self.droppedframes)>0: #i.e. frames were dropped/corrupt. also remove them from original file (if they exist!)
                for fn in self.droppedframes:
                    try:
                        DataCombined.drop(fn,inplace=True)
                    except KeyError:
                        pass
            '''
            DataCombined.sort_index(inplace=True)
            DataCombined.to_hdf(os.path.join(self.dir,'CollectedData_'+ self.humanscorer +'.h5'), key='df_with_missing', mode='w')
            DataCombined.to_csv(os.path.join(self.dir,'CollectedData_'+ self.humanscorer +'.csv'))
        else:
            self.Dataframe.sort_index(inplace=True)
            self.Dataframe.to_hdf(os.path.join(self.dir,'CollectedData_'+ self.humanscorer+'.h5'), key='df_with_missing', mode='w')
            self.Dataframe.to_csv(os.path.join(self.dir,'CollectedData_'+ self.humanscorer +'.csv'))
            self.next.Enable(False)
            self.prev.Enable(False)
            self.slider.Enable(False)
            self.checkBox.Enable(False)

        nextFilemsg = wx.MessageBox('File saved. Do you want to refine another file?', 'Repeat?', wx.YES_NO | wx.ICON_INFORMATION)
        if nextFilemsg == 2:
            self.file = 1
            self.axes.clear()
            self.figure.delaxes(self.figure.axes[1])
            self.choiceBox.Clear(True)
            MainFrame.updateZoomPan(self)
            self.load.Enable(True)
            # self.slider.Enable(False)
            # self.checkBox.Enable(False)
            MainFrame.browseDir(self, event)

# ###########################################################################
# Other functions
# ###########################################################################
    def saveEachImage(self):
        """
        Updates the dataframe for the current image with the new datapoints
        """
        for bpindex, bp in enumerate(self.bodyparts):
            if self.updatedCoords[bpindex]:
                self.Dataframe.loc[self.Dataframe.index[self.iter],(self.scorer,bp,'x')] = self.updatedCoords[bpindex][-1][0]
                self.Dataframe.loc[self.Dataframe.index[self.iter],(self.scorer,bp,'y')] = self.updatedCoords[bpindex][-1][1]

    def getLabels(self,img_index):
        """
        Returns a list of x and y labels of the corresponding image index
        """
        self.previous_image_points = []
        for bpindex, bp in enumerate(self.bodyparts):
            image_points = [[self.Dataframe[self.scorer][bp]['x'].values[self.iter],self.Dataframe[self.scorer][bp]['y'].values[self.iter],bp,bpindex]]
            self.previous_image_points.append(image_points)
        return(self.previous_image_points)

    def plot(self,im):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """
        #small hack in case there are any 0 intensity images!
        im = io.imread(im)
        maxIntensity = np.max(im)
        if maxIntensity == 0:
            maxIntensity = np.max(im) + 255
        self.drs= []
        for bpindex, bp in enumerate(self.bodyparts):
            color = self.colormap(self.norm(self.colorIndex[bpindex]))
            if 'CollectedData_' in self.fileName:
                self.points = [self.Dataframe[self.scorer][bp]['x'].values[self.iter],self.Dataframe[self.scorer][bp]['y'].values[self.iter],1.0]
                self.likelihood = self.points[2]
            else:
                self.points = [self.Dataframe[self.scorer][bp]['x'].values[self.iter],self.Dataframe[self.scorer][bp]['y'].values[self.iter],self.Dataframe[self.scorer][bp]['likelihood'].values[self.iter]]
                self.likelihood = self.points[2]
#                print(self.points)

            if self.move2corner==True:
                ny,nx=np.shape(im)[0],np.shape(im)[1]
                if self.points[0]>nx or self.points[0]<0:
                    self.points[0]=self.center[0]
                if self.points[1]>ny or self.points[1]<0:
                    self.points[1]= self.center[1]

            if not ('CollectedData_' in self.fileName) and self.likelihood < self.threshold:
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, facecolor = 'None', edgecolor = color)]
            else:
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = color, alpha=self.alpha)]

            self.axes.add_patch(circle[0])
            self.dr = auxfun_drag.DraggablePoint(circle[0],bp,self.likelihood)
            self.dr.connect()
            self.dr.coords = MainFrame.getLabels(self,self.iter)[bpindex]
            self.drs.append(self.dr)
            self.updatedCoords.append(self.dr.coords)
        self.figure.canvas.draw()

    def updateZoomPan(self):
            # Checks if zoom/pan button is ON
        if self.pan.GetValue() == True:
            self.toolbar.pan()
            self.pan.SetValue(False)
        if self.zoom.GetValue() == True:
            self.toolbar.zoom()
            self.zoom.SetValue(False)

def show(config):
    app = wx.App()
    frame = MainFrame(None,config).Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()