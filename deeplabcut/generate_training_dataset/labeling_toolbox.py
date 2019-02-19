"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu
"""

import os
import glob
import wx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os.path
from pathlib import Path
import argparse
import yaml
from deeplabcut.generate_training_dataset import auxfun_drag_label
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar


# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################


# minic small screen:
# displaysize = (400, 400)

# Note, if the variable Screens = 2, it assumes two screens in landscape next to eachother! If you use a different configuration, consider changing displaysize to your known display size. see troubleshooting for more information https://github.com/AlexEMG/DeepLabCut/wiki/Troubleshooting-Tips.

# On Windows, there can be a issue with the sizing on start, so you can scale it down then resize on your screen. Namely, set winHack=.5 and this solves this issue. Thanks to Federico Claudi for troubleshooting this with us!

class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent, config, Screens, scale_w, scale_h, winHack, img_scale):

        # Get screen size
        self.size = wx.GetDisplaySize()
        w = self.size[0]
        h = self.size[1]
        self.gui_width = (w * scale_w) / Screens
        self.gui_height = (h * scale_h)

        # print("Scaled GUI width", self.gui_width, "and height", self.gui_height)
        if self.gui_width < 600 or self.gui_height < 500:
            print("Your screen width", w, "and height", h)
            print("Scaled GUI width", self.gui_width, "and height", self.gui_height)
            print("Please adjust scale_h and scale_w, or get a bigger screen!")

        # Initialize window
        wx.Frame.__init__(self, None, title="DeepLabCut2.0 - Labeling GUI",
                          size=(self.gui_width * winHack, self.gui_height * winHack), style=wx.DEFAULT_FRAME_STYLE)

        self.SetBackgroundColour("#ffffff")

        self.Bind(wx.EVT_CLOSE, self.closeWindow)

        # Create status bar
        self.StatusBar = self.CreateStatusBar()
        self.StatusBar.SetForegroundColour(wx.RED)
        self.StatusBar.SetStatusText("")

        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

        self.LoadButton = wx.Button(self, -1, "Load Frames", size=(150, 40),
                                    pos=(self.gui_width * .1, self.gui_height * .9))
        self.LoadButton.Bind(wx.EVT_BUTTON, self.load)
        self.LoadButton.Enable(True)

        self.HelpButton = wx.Button(self, -1, "Help", size=(80, 40), pos=(self.gui_width * .3, self.gui_height * .9))
        self.HelpButton.Bind(wx.EVT_BUTTON, self.help)
        self.HelpButton.Enable(True)

        self.PrevButton = wx.Button(self, -1, "Prev Frame", size=(120, 40),
                                    pos=(self.gui_width * .4, self.gui_height * .9))
        self.PrevButton.Bind(wx.EVT_BUTTON, self.prevImage)
        self.PrevButton.Enable(False)

        self.NextButton = wx.Button(self, -1, "Next Frame", size=(120, 40),
                                    pos=(self.gui_width * .5, self.gui_height * .9))
        self.NextButton.Bind(wx.EVT_BUTTON, self.nextImage)
        self.NextButton.Enable(False)

        self.SaveButton = wx.Button(self, -1, "Save", size=(80, 40), pos=(self.gui_width * .6, self.gui_height * .9))
        self.SaveButton.Bind(wx.EVT_BUTTON, self.save)
        self.SaveButton.Enable(False)

        self.CloseButton = wx.Button(self, -1, "Quit", size=(80, 40), pos=(self.gui_width * .8, self.gui_height * .9))
        self.CloseButton.Bind(wx.EVT_BUTTON, self.closeWindow)

        self.ZoomButton = wx.Button(self, -1, "Zoom", size=(60, 30), pos=(self.gui_width * .65, self.gui_height * .85))
        self.ZoomButton.Bind(wx.EVT_BUTTON, self.zoom)
        self.ZoomButton.Enable(False)

        self.PanButton = wx.Button(self, -1, "Pan", size=(60, 30), pos=(self.gui_width * .75, self.gui_height * .85))
        self.PanButton.Bind(wx.EVT_BUTTON, self.pan)
        self.PanButton.Enable(False)

        self.HomeButton = wx.Button(self, -1, "Home", size=(60, 30), pos=(self.gui_width * .85, self.gui_height * .85))
        self.HomeButton.Bind(wx.EVT_BUTTON, self.home)
        self.HomeButton.Enable(False)

        self.AddLabelsCheckBox = wx.CheckBox(self, label='Add new labels to existing dataset?',
                                             pos=(self.gui_width * .1, self.gui_height * .85))
        self.AddLabelsCheckBox.Bind(wx.EVT_CHECKBOX, self.newLabel)

        self.AdjustMarkerSizeLabel = wx.StaticText(self, label='Adjust marker size',
                                                   pos=(self.gui_width * .43, self.gui_height * .85))
        self.AdjustMarkerSizeSlider = wx.Slider(self, -1, 18, 0, 20, size=(200, -1),
                                                pos=(self.gui_width * .40, self.gui_height * .78),
                                                style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.AdjustMarkerSizeSlider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
        self.AdjustMarkerSizeSlider.Enable(False)

        self.InitNextFrameCheckBox = wx.CheckBox(self, label='Initialize labels on next frame using current positions?',
                                                 pos=(self.gui_width*.6, self.gui_height*.81))
        self.InitNextFrameCheckBox.Bind(wx.EVT_CHECKBOX, self.initNextFrame)
        self.InitNextFrameCheckBox.Enable(False)

        # Define variables
        # Name of each frame filename
        self.frame_filenames = []
        # Colormap
        self.colormap = cm.hsv  # note will be overwritten by colormap from config file during execution

        self.bodyParts=[]

        # Dictionary - keys are body parts, values are list of coordinates for each frame
        self.bodyPartCoords={}
        # Dictionary - keys are body parts, values are patch (circle) for that body part
        self.bodyPartPatches={}
        # Dictionary - keys are body parts, values are draggable for that body part
        self.bodyPartDraggablePoints={}

        # Data frame to save
        self.dataFrame = None
        # Config file
        self.config_file = config

        # Whether or not adding new labels to existing dataset
        self.adding_new_labels = False
        # Whether or not initializing next frame using locations from this frame
        self.initializing_next_frame = False
        # Whether or not zoom mode is on
        self.zoom_mode=False
        # Whether or not pan mode is on
        self.pan_mode=False

        # Whether or not changes have been made and not saved
        self.dirty=False

        # Size of image to show
        imgW = self.gui_width * img_scale  # was 12 inches (perhaps add dpi!)
        imgH = self.gui_height * img_scale  # was 7 inches
        self.img_size = (imgW, imgH)  # width, height in inches.

    """
    Add new labels to existing dataset? toggled
    """
    def newLabel(self, event):
        chk = event.GetEventObject()
        if chk.GetValue():
            self.adding_new_labels = True
            self.AddLabelsCheckBox.Enable(False)
        else:
            self.adding_new_labels = False

    """
    Initialize labels in next frame using positions from this frame? toggled
    """
    def initNextFrame(self, event):
        chk = event.GetEventObject()
        if chk.GetValue():
            self.initializing_next_frame = True
        else:
            self.initializing_next_frame = False

    """
    L/R arrow key pressed
    """
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)
        elif event.GetKeyCode() == wx.WXK_LEFT:
            self.prevImage(event=None)
        elif event.GetKeyCode() == wx.WXK_UP:
            if self.bodyPartRadioBox.GetSelection()>0:
                self.bodyPartRadioBox.Select(self.bodyPartRadioBox.GetSelection()-1)
        elif event.GetKeyCode() == wx.WXK_DOWN:
            if self.bodyPartRadioBox.GetSelection()<len(self.bodyParts)-1:
                self.bodyPartRadioBox.Select(self.bodyPartRadioBox.GetSelection()+1)

    """
    Zoom button pressed
    """
    def zoom(self, event):
        self.zoom_mode=not self.zoom_mode
        if self.zoom_mode:
            self.StatusBar.SetStatusText("Zoom")
        else:
            self.StatusBar.SetStatusText("")

        self.toolbar.zoom()
        self.Refresh(eraseBackground=True)

    """
    Home button pressed
    """
    def home(self, event):
        self.ax1f1.set_xlim([0,self.im.shape[1]])
        self.ax1f1.set_ylim([self.im.shape[0],0])
        self.Refresh(eraseBackground=True)
        self.fig1.canvas.draw()

    """
    Pan button pressed
    """
    def pan(self, event):
        self.pan_mode=not self.pan_mode
        if self.pan_mode:
            self.StatusBar.SetStatusText("Pan")
        else:
            self.StatusBar.SetStatusText("")
        for bodypart in self.bodyPartDraggablePoints:
            if self.bodyPartDraggablePoints[bodypart] is not None:
                if self.zoom_mode:
                    self.bodyPartDraggablePoints[bodypart].disconnect()
                else:
                    self.bodyPartDraggablePoints[bodypart].connect(self.onRelease)
        self.toolbar.pan()
        self.Refresh(eraseBackground=True)

    """
    Quit button pressed
    """
    def closeWindow(self, event):
        if self.dirty:
            savemsg = wx.MessageBox('Save changes before exiting?', 'Save?', wx.YES_NO | wx.ICON_INFORMATION)
            if savemsg == 2:
                self.save(event)
        print("You can now check the labels, using 'check_labels' before proceeding. Then,  you can use the function 'create_training_dataset' to create the training dataset.")
        self.Destroy()

    """
    Help button pressed
    """
    def help(self, event):
        wx.MessageBox(
            '1. Select one of the body parts from the radio buttons to add a label (if necessary change config.yaml first to edit the label names). \n\n'
            '2. RIGHT clicking on the image will add the selected label. \n '
            'The label will be marked as circle filled with a unique color. \n\n'
            '3. Hover your mouse over this newly added label to see its name. \n\n'
            '4. LEFT click and drag to move the label position. \n\n'
            '5. To change the marker size move the slider. \n\n'
            '6. Once you are happy with the position, select another body part from the radio button. \n\n'
            '7. Check \'Initialize labels on next frame using current positions?\' to carry the current labels to the next (or previous frame). \n'
            'This will only work if no labels have yet been added to that frame. \n\n'
            '8. Click Next Frame to move to the next image (Prev Frame to move back). \n\n'
            '9. When finished labeling all the images, click \'Save\' to save all the labels as a .h5 file. \n\n'
            '10. Click OK to continue using the labeling GUI.',
            'User instructions', wx.OK | wx.ICON_INFORMATION)

    """
    Mouse button released after dragging - update body part coord
    (callback from draggable point)
    """
    def onRelease(self, coord, bodypart):
        self.bodyPartCoords[bodypart][self.frameIndex] = coord
        self.dirty=True
        if self.bodyParts[self.bodyPartRadioBox.GetSelection()]==bodypart and self.bodyPartRadioBox.GetSelection() < len(self.bodyParts) - 1:
            self.bodyPartRadioBox.Select(self.bodyPartRadioBox.GetSelection() + 1)

    """
    Mouse button clicked
    """
    def onClick(self, event):
        x1 = event.xdata
        y1 = event.ydata
        normalize = mcolors.Normalize(vmin=np.min(self.colorparams), vmax=np.max(self.colorparams))
        bodypart = self.bodyParts[self.bodyPartRadioBox.GetSelection()]

        # Right-click - adding or deleting coord
        if event.button == 3 and not self.zoom_mode and not self.pan_mode:

            # Coord not already added for this body part - adding
            if self.bodyPartPatches[bodypart] is None:
                color = self.colormap(normalize(self.bodyPartRadioBox.GetSelection()))
                bodyPartPatch = patches.Circle((x1, y1), radius=self.markerSize, fc=color, alpha=0.5)
                self.ax1f1.add_patch(bodyPartPatch)
                self.bodyPartPatches[bodypart]=bodyPartPatch
                draggablePoint = auxfun_drag_label.DraggablePoint(bodyPartPatch, bodypart)
                draggablePoint.connect(self.onRelease)
                draggablePoint.coords = [[x1, y1, bodypart, self.bodyPartRadioBox.GetSelection()]]
                self.bodyPartDraggablePoints[bodypart]=draggablePoint
                self.bodyPartCoords[bodypart][self.frameIndex] = (x1, y1)
                self.dirty = True

            # Coord already added for this body part - deleting
            else:
                """
                To remove a predicted label. Internally, the coordinates of the selected predicted label is replaced with
                nan. The user needs to right click for the event. After right
                click the data point is removed from the plot.
                """
                msg = wx.MessageBox('Do you want to remove %s ? You cannot undo this step!' % bodypart, 'Remove!',
                                    wx.YES_NO | wx.ICON_WARNING)
                if msg == 2:
                    self.bodyPartPatches[bodypart].remove()
                    self.bodyPartDraggablePoints[bodypart].annot.set_visible(False)
                    self.bodyPartDraggablePoints[bodypart]=None
                    self.bodyPartPatches[bodypart]=None
                    self.bodyPartCoords[bodypart][self.frameIndex] = (np.nan, np.nan)
                    self.dirty = True

            if self.bodyPartRadioBox.GetSelection()<len(self.bodyParts)-1:
                self.bodyPartRadioBox.Select(self.bodyPartRadioBox.GetSelection()+1)

        elif event.button == 2:
            self.zoom(None)

        self.fig1.canvas.draw()
        self.canvas.mpl_disconnect(self.onClick)


    """
    Load button pressed
    """
    def load(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """
        if self.dirty:
            savemsg = wx.MessageBox('Save changes before loading new frames?', 'Save?', wx.YES_NO | wx.ICON_INFORMATION)
            if savemsg == 2:
                self.save(event)

        from skimage import io

        dlg = wx.DirDialog(self, "Choose the directory where your extracted frames are saved:",
                           os.path.join(os.getcwd(), 'labeled-data'), style=wx.DD_DEFAULT_STYLE)

        # Get selected directory
        if dlg.ShowModal() == wx.ID_OK:
            self.dir = dlg.GetPath()
            self.NextButton.Enable(True)
            self.HelpButton.Enable(True)
            self.AdjustMarkerSizeSlider.Enable(True)
            self.ZoomButton.Enable(True)
            self.PanButton.Enable(True)
            self.HomeButton.Enable(True)
            self.SaveButton.Enable(True)
            self.InitNextFrameCheckBox.Enable(True)
        else:
            dlg.Destroy()
            return

        dlg.Destroy()

        # Load info from config file
        with open(str(self.config_file), 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile)
        self.scorer = self.cfg['scorer']
        self.bodyParts = self.cfg['bodyparts']
        self.markerSize = self.cfg['dotsize']
        self.colormap = plt.get_cmap(self.cfg['colormap'])
        self.frame_filenames = sorted(glob.glob(os.path.join(self.dir, '*.png')))
        self.frame_filenames.sort()  # sort the files thx to Robert Eppley for this suggestion
        print('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        self.relative_frame_filenames = ['labeled' + n.split('labeled')[1] for n in self.frame_filenames]

        self.AdjustMarkerSizeSlider.SetValue(self.markerSize)

        # Init body part coords
        self.bodyPartCoords = {}
        self.bodyPartPatches = {}
        self.bodyPartDraggablePoints={}
        for bodypart in self.bodyParts:
            self.bodyPartCoords[bodypart]=[]
            self.bodyPartPatches[bodypart]=None
            self.bodyPartDraggablePoints[bodypart]=None
            for frame in self.frame_filenames:
                self.bodyPartCoords[bodypart].append((np.nan,np.nan))

        # Frame index
        self.frameIndex = 0

        # Create plot and show image
        self.fig1, (self.ax1f1) = plt.subplots(figsize=self.img_size, facecolor="None")
        self.im = io.imread(self.frame_filenames[self.frameIndex])
        self.im_axis = self.ax1f1.imshow(self.im, self.colormap)
        img_name = Path(self.frame_filenames[self.frameIndex]).name  # self.index[self.iter].split('/')[-1]
        self.ax1f1.set_title(str(str(self.frameIndex + 1) + "/" + str(len(self.frame_filenames)) + " " + img_name))
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig1)
        self.toolbar = NavigationToolbar(self.canvas)

        # checks for unique bodyparts
        if len(self.bodyParts) != len(set(self.bodyParts)):
            print("Error! bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again. Quiting for now!")
            self.Destroy()

        self.colorparams = list(range(0, len(self.bodyParts) + 1))
        normalize = mcolors.Normalize(vmin=np.min(self.colorparams), vmax=np.max(self.colorparams))

        # Adding new labels to existing data
        if self.adding_new_labels:
            # Read existing data
            self.oldDF = pd.read_hdf(os.path.join(self.dir, 'CollectedData_' + self.scorer + '.h5'), 'df_with_missing')
            oldBodyParts = self.oldDF.columns.get_level_values(1)
            _, idx = np.unique(oldBodyParts, return_index=True)
            oldbodyparts2plot = list(oldBodyParts[np.sort(idx)])
            self.bodyParts = list(set(self.bodyParts) - set(oldbodyparts2plot))
        # Not adding new labels
        else:
            self.AddLabelsCheckBox.Enable(False)

            df_fname=os.path.join(self.dir, 'CollectedData_' + self.scorer + '.h5')
            if os.path.exists(df_fname):
                oldDF = pd.read_hdf(df_fname, 'df_with_missing')
                for bodypart in self.bodyParts:
                    for idx in range(len(self.frame_filenames)):
                        if self.relative_frame_filenames[idx] in oldDF.index.tolist():
                            coord = (oldDF.loc[self.relative_frame_filenames[idx]][self.scorer, bodypart, 'x'],
                                     oldDF.loc[self.relative_frame_filenames[idx]][self.scorer, bodypart, 'y'])
                            self.bodyPartCoords[bodypart][idx]=coord
                    coord=self.bodyPartCoords[bodypart][self.frameIndex]
                    if not np.isnan(coord[0]):
                        color = self.colormap(normalize(self.frameIndex))
                        bodyPartPatch = patches.Circle(coord, radius=self.markerSize, fc=color, alpha=0.5)
                        self.ax1f1.add_patch(bodyPartPatch)
                        self.bodyPartPatches[bodypart] = bodyPartPatch

                        draggablePoint = auxfun_drag_label.DraggablePoint(bodyPartPatch, bodypart)
                        draggablePoint.connect(self.onRelease)
                        draggablePoint.coords = [[coord[0], coord[1], bodypart, self.frameIndex]]
                        self.bodyPartDraggablePoints[bodypart] = draggablePoint
        self.fig1.canvas.draw()

        # Create radio box with all body parts
        self.bodyPartRadioBox = wx.RadioBox(self, id=1, label="Select a body part to annotate",
                                            pos=(self.gui_width * .83, self.gui_height * .1),
                                            choices=self.bodyParts,
                                            majorDimension=1, style=wx.RA_SPECIFY_COLS,
                                            validator=wx.DefaultValidator,
                                            name=wx.RadioBoxNameStr)

        # Create colorbar
        self.cbar = self.fig1.colorbar(self.im_axis, ax=self.ax1f1)
        self.cbar.set_ticks(range(12, np.max(self.im), int(np.floor(np.max(self.im) / len(self.bodyParts)))))
        self.cbar.set_ticklabels(self.bodyParts)

        self.canvas.mpl_connect('button_press_event', self.onClick)
        self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        self.ax1f1.callbacks.connect('xlim_changed', self.onZoom)
        self.ax1f1.callbacks.connect('ylim_changed', self.onZoom)

        a = np.empty((len(self.frame_filenames), 2,))
        a[:] = np.nan
        for bodypart in self.bodyParts:
            index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],
                                               names=['scorer', 'bodyparts', 'coords'])
            # frame = pd.DataFrame(a, columns = index, index = self.index)
            frame = pd.DataFrame(a, columns=index, index=self.relative_frame_filenames)
            self.dataFrame = pd.concat([self.dataFrame, frame], axis=1)
        self.dirty=False

    """
    Refresh (after next or prev image)
    """
    def refresh(self):
        from skimage import io
        normalize = mcolors.Normalize(vmin=np.min(self.colorparams), vmax=np.max(self.colorparams))

        xlim = self.ax1f1.get_xlim()
        ylim = self.ax1f1.get_ylim()

        self.canvas.Destroy()
        plt.close(self.fig1)
        self.ax1f1.clear()
        self.bodyPartRadioBox.SetSelection(0)
        self.fig1, (self.ax1f1) = plt.subplots(figsize=self.img_size, facecolor="None")

        # read the image
        self.im = io.imread(self.frame_filenames[self.frameIndex])
        # Plotting
        self.im_axis = self.ax1f1.imshow(self.im, self.colormap)
        self.cbar = self.fig1.colorbar(self.im_axis, ax=self.ax1f1)
        self.cbar.set_ticks(range(12, np.max(self.im), int(np.floor(np.max(self.im) / len(self.bodyParts)))))
        self.cbar.set_ticklabels(self.bodyParts)
        img_name = Path(self.frame_filenames[self.frameIndex]).name  # self.index[self.iter].split('/')[-1]
        self.ax1f1.set_title(str(str(self.frameIndex + 1) + "/" + str(len(self.frame_filenames)) + " " + img_name))
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig1)
        self.canvas.mpl_connect('button_press_event', self.onClick)
        self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        self.ax1f1.callbacks.connect('xlim_changed', self.onZoom)
        self.ax1f1.callbacks.connect('ylim_changed', self.onZoom)

        # Recreate toolbar for zooming
        self.toolbar = NavigationToolbar(self.canvas)

        # Draw bodyPart coords
        for idx, bodypart in enumerate(self.bodyParts):
            coord = self.bodyPartCoords[bodypart][self.frameIndex]
            if not np.isnan(coord[0]):
                color = self.colormap(normalize(idx))
                bodyPartPatch = patches.Circle(coord, radius=self.markerSize, fc=color, alpha=0.5)
                self.ax1f1.add_patch(bodyPartPatch)
                self.bodyPartPatches[bodypart] = bodyPartPatch

                draggablePoint = auxfun_drag_label.DraggablePoint(bodyPartPatch, bodypart)
                draggablePoint.connect(self.onRelease)
                draggablePoint.coords = [[coord[0], coord[1], bodypart, idx]]
                self.bodyPartDraggablePoints[bodypart] = draggablePoint
            else:
                self.bodyPartPatches[bodypart]=None
                self.bodyPartDraggablePoints[bodypart]=None

        self.ax1f1.set_xlim(xlim)
        self.ax1f1.set_ylim(ylim)

        self.fig1.canvas.draw()


    def onZoom(self, ax):
        if self.zoom_mode:
            self.zoom_mode=False
            self.StatusBar.SetStatusText("")
            self.toolbar.ToggleTool(self.toolbar.wx_ids['Zoom'], False)
            self.toolbar.zoom(None)

    def onButtonRelease(self, event):
        if self.pan_mode:
            self.pan_mode = False
            self.StatusBar.SetStatusText("")
            self.toolbar.ToggleTool(self.toolbar.wx_ids['Pan'], False)
            self.toolbar.pan(None)

    """
    Moves to previous image
    """
    def prevImage(self, event):
        # Checks for the first image and disables the Prev button + diesbt load the next if RIGHT arrow key pressed
        if self.frameIndex == 1:
            self.PrevButton.Enable(False)

        if self.frameIndex>=1:
            self.NextButton.Enable(True)
            self.frameIndex = self.frameIndex - 1

            if self.zoom_mode:
                self.zoom(None)
            if self.pan_mode:
                self.pan(None)

            if self.initializing_next_frame:
                # Don't initialize if there are any labels in the next frame
                init_coords=True
                for bodypart in self.bodyParts:
                    # Check if previous frame has coordinate - if it does, don't change
                    if not np.isnan(self.bodyPartCoords[bodypart][self.frameIndex][0]):
                        init_coords=False
                        break
                # Init coordinates
                if init_coords:
                    for bodypart in self.bodyParts:
                        # Initialize coordinate based on last frame coordinate
                        self.bodyPartCoords[bodypart][self.frameIndex]=self.bodyPartCoords[bodypart][self.frameIndex+1]
                    self.dirty=True

            self.refresh()


    """
    Moves to next Image
    """
    def nextImage(self, event):
        # Checks for the last image and disables the Next button + diesbt load the next if RIGHT arrow key pressed
        if self.frameIndex==len(self.frame_filenames)-2:
            self.NextButton.Enable(False)

        if self.frameIndex<=len(self.frame_filenames)-2:
            self.PrevButton.Enable(True)
            self.frameIndex = self.frameIndex + 1

            if self.zoom_mode:
                self.zoom(None)
            if self.pan_mode:
                self.pan(None)

            if self.initializing_next_frame:
                # Don't initialize if there are any labels in the previous frame
                init_coords = True
                for bodypart in self.bodyParts:
                    # Check if next frame has coordinate - if it does, don't change
                    if not np.isnan(self.bodyPartCoords[bodypart][self.frameIndex][0]):
                        init_coords=False
                        break
                # Init coordinates
                if init_coords:
                    for bodypart in self.bodyParts:
                        self.bodyPartCoords[bodypart][self.frameIndex]=self.bodyPartCoords[bodypart][self.frameIndex-1]
                    self.dirty=True
            self.refresh()


    """
    Save button pressed
    """
    def save(self, event):
        """
        Saves the final dataframe
        """
        for bodypart in self.bodyParts:
            for idx in range(len(self.frame_filenames)):
                coord=self.bodyPartCoords[bodypart][idx]
                self.dataFrame.loc[self.relative_frame_filenames[idx]][self.scorer, bodypart, 'x'] = coord[0]
                self.dataFrame.loc[self.relative_frame_filenames[idx]][self.scorer, bodypart, 'y'] = coord[1]

        if self.adding_new_labels:
            self.dataFrame = pd.concat([self.oldDF, self.dataFrame], axis=1)

        # Windows compatible
        self.dataFrame.to_csv(os.path.join(self.dir, "CollectedData_" + self.scorer + ".csv"))
        self.dataFrame.to_hdf(os.path.join(self.dir, "CollectedData_" + self.scorer + '.h5'), 'df_with_missing',
                              format='table', mode='w')

        self.dirty=False

    """
    Adjust marker size for plotting labels
    """
    def OnSliderScroll(self, event):
        self.markerSize = self.AdjustMarkerSizeSlider.GetValue()

        for bodypart in self.bodyParts:
            bodypartPatch=self.bodyPartPatches[bodypart]
            if bodypartPatch is not None:
                bodypartPatch.set_radius(self.markerSize)
        self.fig1.canvas.draw()


from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, Toolbar


class MatplotPanel(wx.Panel):
    def __init__(self, parent, config):
        panel1 = wx.Panel.__init__(self, parent)
        self.figure = Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.toolbar = Toolbar(self.canvas)  # matplotlib toolbar
        self.toolbar.EnableTool(1, False)
        self.toolbar.Realize()
        # self.toolbar.set_active([0,1])
        self.toolbar.update()
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        # sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        self.Refresh(eraseBackground=True)


def GetToolBar(self):  # https://matplotlib.org/examples/user_interfaces/embedding_in_wx3.html
    # You will need to override GetToolBar if you are using an
    # unmanaged toolbar in your frame
    return self.toolbar


def show(config, Screens=1, scale_w=.8, scale_h=.9, winHack=1, img_scale=0.0075):
    app = wx.App()
    frame = MainFrame(None, config, Screens, scale_w, scale_h, winHack, img_scale).Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()

