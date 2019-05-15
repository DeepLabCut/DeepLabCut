"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

import os
import os.path
import glob
import cv2
import wx
import wx.lib.scrolledpanel as SP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import argparse
from deeplabcut.generate_training_dataset import auxfun_drag_label
from deeplabcut.utils import auxiliaryfunctions
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

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
        self.orig_xlim = None
        self.orig_ylim = None
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def getfigure(self):
        return(self.figure)

    def drawplot(self,img,img_name,itr,index,bodyparts,cmap,keep_view=False):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.clear()

        # convert the image to RGB as you are showing the image with matplotlib
        im = cv2.imread(img)[...,::-1]
        ax = self.axes.imshow(im,cmap=cmap)
        self.orig_xlim = self.axes.get_xlim()
        self.orig_ylim = self.axes.get_ylim()
        divider = make_axes_locatable(self.axes)
        colorIndex = np.linspace(np.min(im),np.max(im),len(bodyparts))
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.figure.colorbar(ax, cax=cax,spacing='proportional', ticks=colorIndex)
        cbar.set_ticklabels(bodyparts[::-1])
        self.axes.set_title(str(str(itr)+"/"+str(len(index)-1) +" "+ img_name ))
        if keep_view:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        self.toolbar = NavigationToolbar(self.canvas)
        return(self.figure,self.axes,self.canvas,self.toolbar)

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)

    def getColorIndices(self,img,bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        im = cv2.imread(img)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(im))
        ticks = np.linspace(0,np.max(im),len(bodyparts))[::-1]
        return norm, ticks



class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)


class ScrollPanel(SP.ScrolledPanel):
    def __init__(self, parent):
        SP.ScrolledPanel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
        self.Layout()
    def on_focus(self,event):
        pass

    def addRadioButtons(self,bodyparts,fileIndex,markersize):
        """
        Adds radio buttons for each bodypart on the right panel
        """
        self.choiceBox = wx.BoxSizer(wx.VERTICAL)
        choices = [l for l in bodyparts]
        self.fieldradiobox = wx.RadioBox(self,label='Select a bodypart to label',
                                    style=wx.RA_SPECIFY_ROWS,choices=choices)
        self.slider = wx.Slider(self, -1, markersize, 1, markersize*3,size=(250, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        self.slider.Enable(False)
        self.checkBox = wx.CheckBox(self, id=wx.ID_ANY,label = 'Adjust marker size.')
        self.choiceBox.Add(self.slider, 0, wx.ALL, 5 )
        self.choiceBox.Add(self.checkBox, 0, wx.ALL, 5 )
        self.choiceBox.Add(self.fieldradiobox, 0, wx.EXPAND|wx.ALL, 10)
        self.SetSizerAndFit(self.choiceBox)
        self.Layout()
        return(self.choiceBox,self.fieldradiobox,self.slider,self.checkBox)

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

        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'DeepLabCut2.0 - Labeling ToolBox',
                            size = wx.Size(self.gui_size), pos = wx.DefaultPosition, style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("Looking for a folder to start labeling. Click 'Load frames' to begin.")
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

        self.SetSizeHints(wx.Size(self.gui_size)) #  This sets the minimum size of the GUI. It can scale now!
###################################################################################################################################################

# Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!

        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)

        self.image_panel = ImagePanel(vSplitter, config,self.gui_size)
        self.choice_panel = ScrollPanel(vSplitter)
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
        self.load = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Load frames")
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

        self.lock = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Lock View")
        widgetsizer.Add(self.lock, 1, wx.ALL, 15)
        self.lock.Bind(wx.EVT_CHECKBOX, self.lockChecked)
        self.widget_panel.SetSizer(widgetsizer)
        self.lock.Enable(False)

        self.save = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Save")
        widgetsizer.Add(self.save , 1, wx.ALL, 15)
        self.save.Bind(wx.EVT_BUTTON, self.saveDataSet)
        self.save.Enable(False)

        widgetsizer.AddStretchSpacer(15)
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit , 1, wx.ALL|wx.ALIGN_RIGHT, 15)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)

        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        self.widget_panel.Layout()

###############################################################################################################################
# Variables initialization

        self.currentDirectory = os.getcwd()
        self.index = []
        self.iter = []
        self.file = 0
        self.updatedCoords = []
        self.dataFrame = None
        self.config_file = config
        self.new_labels = False
        self.buttonCounter = []
        self.bodyparts2plot = []
        self.drs = []
        self.num = []
        self.view_locked=False

###############################################################################################################################
# BUTTONS FUNCTIONS FOR HOTKEYS
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)
        elif event.GetKeyCode() == wx.WXK_LEFT:
            self.prevImage(event=None)

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
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
        self.buttonCounter = []
        self.markerSize = self.slider.GetValue()
        img_name = Path(self.index[self.iter]).name
        self.figure.delaxes(self.figure.axes[1])
        self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.bodyparts,self.colormap,keep_view=True)

#        self.axes.callbacks.connect('xlim_changed', self.onZoom)
#        self.axes.callbacks.connect('ylim_changed', self.onZoom)
        self.buttonCounter = MainFrame.plot(self,self.img)

    def quitButton(self, event):
        """
        Asks user for its inputs and then quits the GUI
        """
        self.statusbar.SetStatusText("Qutting now!")

        nextFilemsg = wx.MessageBox('Do you want to label another data set?', 'Repeat?', wx.YES_NO | wx.ICON_INFORMATION)
        if nextFilemsg == 2:
            self.file = 1
            self.buttonCounter = []
            self.updatedCoords = []
            self.dataFrame = None
            self.bodyparts = []
            self.new_labels = self.new_labels
            self.axes.clear()
            self.figure.delaxes(self.figure.axes[1])
            self.choiceBox.Clear(True)
            MainFrame.updateZoomPan(self)
            MainFrame.browseDir(self, event)
            self.save.Enable(True)
        else:
            self.Destroy()
            print("You can now check the labels, using 'check_labels' before proceeding. Then, you can use the function 'create_training_dataset' to create the training dataset.")

    def helpButton(self,event):
        """
        Opens Instructions
        """
        MainFrame.updateZoomPan(self)
        wx.MessageBox('1. Select one of the body parts from the radio buttons to add a label (if necessary change config.yaml first to edit the label names). \n\n2. Right clicking on the image will add the selected label and the next available label will be selected from the radio button. \n The label will be marked as circle filled with a unique color.\n\n3. To change the marker size, mark the checkbox and move the slider. \n\n4. Hover your mouse over this newly added label to see its name. \n\n5. Use left click and drag to move the label position.  \n\n6. Once you are happy with the position, right click to add the next available label. You can always reposition the old labels, if required. You can delete a label with the middle button mouse click. \n\n7. Click Next/Previous to move to the next/previous image.\n User can also add a missing label by going to a previous/next image and using the left click to add the selected label.\n NOTE: the user cannot add a label if the label is already present. \n\n8. When finished labeling all the images, click \'Save\' to save all the labels as a .h5 file. \n\n9. Click OK to continue using the labeling GUI.', 'User instructions', wx.OK | wx.ICON_INFORMATION)
        self.statusbar.SetStatusText("Help")

    def homeButton(self,event):
        self.image_panel.resetView()
        self.figure.canvas.draw()
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

#    def onZoom(self, ax):
#        print(self.zoom.GetValue())
#        if self.zoom.GetValue():
#            self.updateZoomPan()
#            self.statusbar.SetStatusText("Zoom Off")

    def onButtonRelease(self, event):
        if self.pan.GetValue():
            self.updateZoomPan()
            self.statusbar.SetStatusText("Pan Off")

    def lockChecked(self, event):
        self.cb = event.GetEventObject()
        self.view_locked=self.cb.GetValue()

    def onClick(self,event):
        """
        This function adds labels and auto advances to the next label.
        """
        x1 = event.xdata
        y1 = event.ydata

        if event.button == 3:
            if self.rdb.GetSelection() in self.buttonCounter :
                wx.MessageBox('%s is already annotated. \n Select another body part to annotate.' % (str(self.bodyparts[self.rdb.GetSelection()])), 'Error!', wx.OK | wx.ICON_ERROR)
            else:
                color = self.colormap(self.norm(self.colorIndex[self.rdb.GetSelection()]))
                circle = [patches.Circle((x1, y1), radius = self.markerSize, fc=color, alpha=self.alpha)]
                self.num.append(circle)
                self.axes.add_patch(circle[0])
                self.dr = auxfun_drag_label.DraggablePoint(circle[0],self.bodyparts[self.rdb.GetSelection()])
                self.dr.connect()
                self.buttonCounter.append(self.rdb.GetSelection())
                self.dr.coords = [[x1,y1,self.bodyparts[self.rdb.GetSelection()],self.rdb.GetSelection()]]
                self.drs.append(self.dr)
                self.updatedCoords.append(self.dr.coords)
                if self.rdb.GetSelection() < len(self.bodyparts) -1:
                    self.rdb.SetSelection(self.rdb.GetSelection() + 1)
                self.figure.canvas.draw()

        self.canvas.mpl_disconnect(self.onClick)
        self.canvas.mpl_disconnect(self.onButtonRelease)

    def browseDir(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """
        self.statusbar.SetStatusText("Looking for a folder to start labeling...")
        cwd = os.path.join(os.getcwd(),'labeled-data')
        dlg = wx.DirDialog(self, "Choose the directory where your extracted frames are saved:",cwd, style = wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.dir = dlg.GetPath()
            self.load.Enable(False)
            self.next.Enable(True)
            self.save.Enable(True)
        else:
            dlg.Destroy()
            self.Close(True)
        dlg.Destroy()

# Enabling the zoom, pan and home buttons
        self.zoom.Enable(True)
        self.home.Enable(True)
        self.pan.Enable(True)
        self.lock.Enable(True)

# Reading config file and its variables
        self.cfg = auxiliaryfunctions.read_config(self.config_file)
        self.scorer = self.cfg['scorer']
        self.bodyparts = self.cfg['bodyparts']
        self.videos = self.cfg['video_sets'].keys()
        self.markerSize = self.cfg['dotsize']
        self.alpha = self.cfg['alphavalue']
        self.colormap = plt.get_cmap(self.cfg['colormap'])
        self.colormap = self.colormap.reversed()
        self.project_path=self.cfg['project_path']
        self.index =np.sort([fn for fn in glob.glob(os.path.join(self.dir,'*.png')) if ('labeled.png' not in fn)])
        self.statusbar.SetStatusText('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        self.relativeimagenames=['labeled'+n.split('labeled')[1] for n in self.index]#[n.split(self.project_path+'/')[1] for n in self.index]

# Reading the existing dataset,if already present
        try:
            self.dataFrame = pd.read_hdf(os.path.join(self.dir,'CollectedData_'+self.scorer+'.h5'),'df_with_missing')
            self.dataFrame.sort_index(inplace=True)
            self.prev.Enable(True)

# Finds the first empty row in the dataframe and sets the iteration to that index
            for idx,j in enumerate(self.dataFrame.index):
                values = self.dataFrame.loc[j,:].values
                if np.prod(np.isnan(values)) == 1:
                    self.iter = idx
                    break
                else:
                    self.iter = idx

        except:
            a = np.empty((len(self.index),2,))
            a[:] = np.nan
            for bodypart in self.bodyparts:
                index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(a, columns = index, index = self.relativeimagenames)
                self.dataFrame = pd.concat([self.dataFrame, frame],axis=1)
            self.iter = 0

# Reading the image name
        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name
        self.norm,self.colorIndex = self.image_panel.getColorIndices(self.img,self.bodyparts)

# Checking for new frames and adding them to the existing dataframe
        old_imgs = np.sort(list(self.dataFrame.index))
        self.newimages = list(set(self.relativeimagenames) - set(old_imgs))
        if self.newimages == []:
            pass
        else:
            print("Found new frames..")
# Create an empty dataframe with all the new images and then merge this to the existing dataframe.
            self.df=None
            a = np.empty((len(self.newimages),2,))
            a[:] = np.nan
            for bodypart in self.bodyparts:
                index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(a, columns = index, index = self.newimages)
                self.df = pd.concat([self.df, frame],axis=1)
            self.dataFrame = pd.concat([self.dataFrame, self.df],axis=0)
# Sort it by the index values
            self.dataFrame.sort_index(inplace=True)

# checks for unique bodyparts
        if len(self.bodyparts)!=len(set(self.bodyparts)):
          print("Error - bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again. Quitting for now!")
          self.Close(True)

# Extracting the list of new labels
        oldBodyParts = self.dataFrame.columns.get_level_values(1)
        _, idx = np.unique(oldBodyParts, return_index=True)
        oldbodyparts2plot =  list(oldBodyParts[np.sort(idx)])
        self.new_bodyparts =  [x for x in self.bodyparts if x not in oldbodyparts2plot ]
# Checking if user added a new label
        if self.new_bodyparts==[]: # i.e. no new label
            self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.bodyparts,self.colormap)
            #self.axes.callbacks.connect('xlim_changed', self.onZoom)
            #self.axes.callbacks.connect('ylim_changed', self.onZoom)

            self.choiceBox,self.rdb,self.slider,self.checkBox = self.choice_panel.addRadioButtons(self.bodyparts,self.file,self.markerSize)
            self.buttonCounter = MainFrame.plot(self,self.img)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        else:
            dlg = wx.MessageDialog(None,"New label found in the config file. Do you want to see all the other labels?", "New label found",wx.YES_NO | wx.ICON_WARNING)
            result = dlg.ShowModal()
            if result == wx.ID_NO:
                self.bodyparts = self.new_bodyparts
                self.norm,self.colorIndex = self.image_panel.getColorIndices(self.img,self.bodyparts)
            a = np.empty((len(self.index),2,))
            a[:] = np.nan
            for bodypart in self.new_bodyparts:
                index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(a, columns = index, index = self.relativeimagenames)
                self.dataFrame = pd.concat([self.dataFrame, frame],axis=1)

            self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,self.index,self.bodyparts,self.colormap)
            #self.axes.callbacks.connect('xlim_changed', self.onZoom)
            #self.axes.callbacks.connect('ylim_changed', self.onZoom)

            self.choiceBox,self.rdb,self.slider,self.checkBox = self.choice_panel.addRadioButtons(self.bodyparts,self.file,self.markerSize)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
            self.buttonCounter = MainFrame.plot(self,self.img)

        self.checkBox.Bind(wx.EVT_CHECKBOX,self.activateSlider)
        self.slider.Bind(wx.EVT_SLIDER,self.OnSliderScroll)

    def nextImage(self,event):
        """
        Moves to next image
        """
#  Checks for the last image and disables the Next button
        if len(self.index) - self.iter == 1:
            self.next.Enable(False)
            return
        self.prev.Enable(True)

# Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)

        self.statusbar.SetStatusText('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        self.rdb.SetSelection(0)
        self.file = 1
# Refreshing the button counter
        self.buttonCounter = []
        MainFrame.saveEachImage(self)

        self.iter = self.iter + 1

        if len(self.index) >= self.iter:
            self.updatedCoords = MainFrame.getLabels(self,self.iter)
            self.img = self.index[self.iter]
            img_name = Path(self.index[self.iter]).name
            self.figure.delaxes(self.figure.axes[1]) # Removes the axes corresponding to the colorbar
            self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,
                                                                                       self.index,self.bodyparts,
                                                                                       self.colormap,
                                                                                       keep_view=self.view_locked)
#            self.axes.callbacks.connect('xlim_changed', self.onZoom)
#            self.axes.callbacks.connect('ylim_changed', self.onZoom)

            self.buttonCounter = MainFrame.plot(self,self.img)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)

    def prevImage(self, event):
        """
        Checks the previous Image and enables user to move the annotations.
        """
# Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.prev.Enable(False)
            return
        else:
            self.next.Enable(True)
# Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        self.statusbar.SetStatusText('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        MainFrame.saveEachImage(self)

        self.buttonCounter = []
        self.iter = self.iter - 1

        self.rdb.SetSelection(0)
        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name
        self.figure.delaxes(self.figure.axes[1]) # Removes the axes corresponding to the colorbar
        self.figure,self.axes,self.canvas,self.toolbar = self.image_panel.drawplot(self.img,img_name,self.iter,
                                                                                   self.index,self.bodyparts,
                                                                                   self.colormap,
                                                                                   keep_view=self.view_locked)
#        self.axes.callbacks.connect('xlim_changed', self.onZoom)
#        self.axes.callbacks.connect('ylim_changed', self.onZoom)

        self.buttonCounter = MainFrame.plot(self,self.img)
        self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
        self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        MainFrame.saveEachImage(self)

    def getLabels(self,img_index):
        """
        Returns a list of x and y labels of the corresponding image index
        """
        self.previous_image_points = []
        for bpindex, bp in enumerate(self.bodyparts):
            image_points = [[self.dataFrame[self.scorer][bp]['x'].values[self.iter],self.dataFrame[self.scorer][bp]['y'].values[self.iter],bp,bpindex]]
            self.previous_image_points.append(image_points)
        return(self.previous_image_points)

    def plot(self,img):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """
        self.drs= []
        self.updatedCoords = []
        for bpindex, bp in enumerate(self.bodyparts):
            color = self.colormap(self.norm(self.colorIndex[bpindex]))
            self.points = [self.dataFrame[self.scorer][bp]['x'].values[self.iter],self.dataFrame[self.scorer][bp]['y'].values[self.iter]]
            circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = color, alpha=self.alpha)]
            self.axes.add_patch(circle[0])
            self.dr = auxfun_drag_label.DraggablePoint(circle[0],self.bodyparts[bpindex])
            self.dr.connect()
            self.dr.coords = MainFrame.getLabels(self,self.iter)[bpindex]
            self.drs.append(self.dr)
            self.updatedCoords.append(self.dr.coords)
            if np.isnan(self.points)[0] == False:
                self.buttonCounter.append(bpindex)
        self.figure.canvas.draw()

        return(self.buttonCounter)

    def saveEachImage(self):
        """
        Saves data for each image
        """
        for idx, bp in enumerate(self.updatedCoords):
            self.dataFrame.loc[self.relativeimagenames[self.iter]][self.scorer, bp[0][-2],'x' ] = bp[-1][0]
            self.dataFrame.loc[self.relativeimagenames[self.iter]][self.scorer, bp[0][-2],'y' ] = bp[-1][1]

    def saveDataSet(self,event):
        """
        Saves the final dataframe
        """
        self.statusbar.SetStatusText("File saved")
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
# Windows compatible
        self.dataFrame.sort_index(inplace=True)
        self.dataFrame.to_csv(os.path.join(self.dir,"CollectedData_" + self.scorer + ".csv"))
        self.dataFrame.to_hdf(os.path.join(self.dir,"CollectedData_" + self.scorer + '.h5'),'df_with_missing',format='table', mode='w')

    def onChecked(self, event):
      self.cb = event.GetEventObject()
      if self.cb.GetValue() == True:
          self.slider.Enable(True)
          self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
          self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
      else:
          self.slider.Enable(False)

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