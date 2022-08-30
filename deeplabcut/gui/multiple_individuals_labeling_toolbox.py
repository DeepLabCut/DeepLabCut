"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import glob
import os
import os.path
from pathlib import Path

import re
import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wx
import wx.lib.scrolledpanel as SP
from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from deeplabcut.gui import auxfun_drag
from deeplabcut.gui.widgets import BasePanel, WidgetPanel, BaseFrame
from deeplabcut.utils import (
    auxiliaryfunctions,
    auxfun_multianimal,
    auxiliaryfunctions_3d,
    conversioncode,
)

# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(BasePanel):
    def __init__(self, parent, config, config3d, sourceCam, gui_size, **kwargs):
        super(ImagePanel, self).__init__(parent, config, gui_size, **kwargs)
        self.config = config
        self.cfg = auxiliaryfunctions.read_config(self.config)
        self.config3d = config3d
        self.sourceCam = sourceCam
        self.toolbar = None

    def retrieveData_and_computeEpLines(self, img, imNum):

        # load labeledPoints and fundamental Matrix

        if self.config3d is not None:
            cfg_3d = auxiliaryfunctions.read_config(self.config3d)
            cams = cfg_3d["camera_names"]
            path_camera_matrix = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)[2]
            path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
            stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)

            for cam in cams:
                if cam in img:
                    labelCam = cam
                    if self.sourceCam is None:
                        sourceCam = [
                            otherCam for otherCam in cams if cam not in otherCam
                        ][0]
                    else:
                        sourceCam = self.sourceCam

            sourceCamIdx = np.where(np.array(cams) == sourceCam)[0][0]
            labelCamIdx = np.where(np.array(cams) == labelCam)[0][0]
            if sourceCamIdx < labelCamIdx:
                camera_pair = cams[sourceCamIdx] + "-" + cams[labelCamIdx]
                sourceCam_numInPair = 1
            else:
                camera_pair = cams[labelCamIdx] + "-" + cams[sourceCamIdx]
                sourceCam_numInPair = 2

            fundMat = stereo_file[camera_pair]["F"]
            sourceCam_path = os.path.split(img.replace(labelCam, sourceCam))[0]

            cfg = auxiliaryfunctions.read_config(self.config)
            scorer = cfg["scorer"]

            try:
                dataFrame = pd.read_hdf(
                    os.path.join(sourceCam_path, "CollectedData_" + scorer + ".h5")
                )
                dataFrame.sort_index(inplace=True)
            except IOError:
                print(
                    "source camera images have not yet been labeled, or you have opened this folder in the wrong mode!"
                )
                return None, None, None

            # Find offset terms for drawing epipolar Lines
            # Get crop params for camera being labeled
            foundEvent = 0
            eventSearch = re.compile(os.path.split(os.path.split(img)[0])[1])
            cropPattern = re.compile("[0-9]{1,4}")
            with open(self.config, "rt") as config:
                for line in config:
                    if foundEvent == 1:
                        crop_labelCam = np.int32(re.findall(cropPattern, line))
                        break
                    if eventSearch.search(line) is not None:
                        foundEvent = 1
            # Get crop params for other camera
            foundEvent = 0
            eventSearch = re.compile(os.path.split(sourceCam_path)[1])
            cropPattern = re.compile("[0-9]{1,4}")
            with open(self.config, "rt") as config:
                for line in config:
                    if foundEvent == 1:
                        crop_sourceCam = np.int32(re.findall(cropPattern, line))
                        break
                    if eventSearch.search(line) is not None:
                        foundEvent = 1

            labelCam_offsets = [crop_labelCam[0], crop_labelCam[2]]
            sourceCam_offsets = [crop_sourceCam[0], crop_sourceCam[2]]

            sourceCam_pts = np.asarray(dataFrame, dtype=np.int32)
            sourceCam_pts = sourceCam_pts.reshape(
                (sourceCam_pts.shape[0], int(sourceCam_pts.shape[1] / 2), 2)
            )
            sourceCam_pts = np.moveaxis(sourceCam_pts, [0, 1, 2], [1, 0, 2])
            sourceCam_pts[..., 0] = sourceCam_pts[..., 0] + sourceCam_offsets[0]
            sourceCam_pts[..., 1] = sourceCam_pts[..., 1] + sourceCam_offsets[1]

            sourcePts = sourceCam_pts[:, imNum, :]

            epLines_source2label = cv2.computeCorrespondEpilines(
                sourcePts, int(sourceCam_numInPair), fundMat
            )
            epLines_source2label.reshape(-1, 3)

            return epLines_source2label, sourcePts, labelCam_offsets

        else:
            return None, None, None

    def drawEpLines(self, drawImage, lines, sourcePts, offsets, colorIndex, cmap, norm):
        height, width, depth = drawImage.shape
        for line, pt, cIdx in zip(lines, sourcePts, colorIndex):
            if pt[0] > -1000:
                coeffs = line[0]
                x0, y0 = map(int, [0 - offsets[0], -coeffs[2] / coeffs[1] - offsets[1]])
                x1, y1 = map(
                    int,
                    [
                        width,
                        -(coeffs[2] + coeffs[0] * (width + offsets[0])) / coeffs[1]
                        - offsets[1],
                    ],
                )
                color = cmap(norm(cIdx), bytes=True)[:-1]
                color = tuple([int(x) for x in color])
                drawImage = cv2.line(drawImage, (x0, y0), (x1, y1), color, 1)
        return drawImage

    def drawplot(self, img, img_name, itr, index, bodyparts, cmap, keep_view=False):
        individuals = self.cfg["individuals"]
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.clear()
        # convert the image to RGB as you are showing the image with matplotlib
        im = cv2.imread(img)[..., ::-1]
        colorIndex = []
        for indiv in range(len(individuals)):
            colorIndex.extend(np.linspace(np.max(im), 0, len(bodyparts)))
        colorIndex = np.array(colorIndex)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(im))
        # draw epipolar lines
        epLines, sourcePts, offsets = self.retrieveData_and_computeEpLines(img, itr)
        if epLines is not None:
            im = self.drawEpLines(
                im.copy(), epLines, sourcePts, offsets, colorIndex, cmap, norm
            )
        ax = self.axes.imshow(im, cmap=cmap)
        self.orig_xlim = self.axes.get_xlim()
        self.orig_ylim = self.axes.get_ylim()
        #        divider = make_axes_locatable(self.axes)
        #        colorIndex = np.linspace(np.min(im),np.max(im),len(bodyparts))
        #        cax = divider.append_axes("right", size="5%", pad=0.05)
        #        cbar = self.figure.colorbar(ax, cax=cax,spacing='proportional', ticks=colorIndex)
        #        cbar.set_ticklabels(bodyparts[::-1])
        self.axes.set_title(str(str(itr) + "/" + str(len(index) - 1) + " " + img_name))
        # self.figure.canvas.draw()
        if keep_view:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        if self.toolbar is None:
            self.toolbar = NavigationToolbar(self.canvas)
        return (self.figure, self.axes, self.canvas, self.toolbar, ax)

    def addcolorbar(self, img, ax, itr, bodyparts, cmap):
        im = cv2.imread(img)
        divider = make_axes_locatable(self.axes)
        colorIndex = np.linspace(np.min(im), np.max(im), len(bodyparts))
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.figure.colorbar(
            ax, cax=cax, spacing="proportional", ticks=colorIndex
        )
        cbar.set_ticklabels(bodyparts[::-1])
        self.figure.canvas.draw()
        if self.toolbar is None:
            self.toolbar = NavigationToolbar(self.canvas)
        return (self.figure, self.axes, self.canvas, self.toolbar)

    def getColorIndices(self, img, bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        im = cv2.imread(img)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(im))
        ticks = np.linspace(0, np.max(im), len(bodyparts))[::-1]
        return norm, ticks


class ScrollPanel(SP.ScrolledPanel):
    def __init__(self, parent):
        SP.ScrolledPanel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
        self.Layout()

    def on_focus(self, event):
        pass

    def addRadioButtons(self, bodyparts, individual_names, fileIndex, markersize):
        """
        Adds radio buttons for each bodypart on the right panel
        """
        self.choiceBox = wx.BoxSizer(wx.VERTICAL)
        choices = [l for l in bodyparts]
        self.fieldradiobox = wx.RadioBox(
            self,
            label="Select a bodypart to label",
            majorDimension=3,
            style=wx.RA_SPECIFY_COLS,
            choices=choices,
        )
        self.change_marker = wx.Slider(
            self,
            -1,
            markersize,
            1,
            markersize * 3,
            size=(250, -1),
            style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.change_marker.Enable(False)
        names = [k for k in individual_names]
        self.individualradiobox = wx.RadioBox(
            self,
            label="Select an individual",
            majorDimension=3,
            style=wx.RA_SPECIFY_COLS,
            choices=names,
        )

        self.checkBox = wx.CheckBox(self, id=wx.ID_ANY, label="Adjust marker size")
        self.choiceBox.Add(self.change_marker, 0, wx.ALL, 5)
        self.choiceBox.Add(self.checkBox, 0, wx.ALL, 5)
        self.choiceBox.Add(self.individualradiobox, 0, wx.EXPAND | wx.ALL, 10)

        self.choiceBox.Add(self.fieldradiobox, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(self.choiceBox)
        self.Layout()
        return (
            self.choiceBox,
            self.individualradiobox,
            self.fieldradiobox,
            self.change_marker,
            self.checkBox,
        )

    def clearBoxer(self):
        self.choiceBox.Clear(True)


class MainFrame(BaseFrame):
    def __init__(self, parent, config, config3d, sourceCam, jump_unlabeled):
        super(MainFrame, self).__init__(
            "DeepLabCut 2.2 - Multiple Individuals Labeling", parent
        )
        self.jump_unlabeled = jump_unlabeled
        self.statusbar.SetStatusText(
            "Looking for a folder to start labeling. Click 'Load frames' to begin."
        )
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

        ###################################################################################################################################################

        # Splitting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!

        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)

        self.image_panel = ImagePanel(
            vSplitter, config, config3d, sourceCam, self.gui_size
        )
        self.choice_panel = ScrollPanel(vSplitter)

        vSplitter.SplitVertically(
            self.image_panel, self.choice_panel, sashPosition=self.gui_size[0] * 0.8
        )
        vSplitter.SetSashGravity(1)
        self.widget_panel = WidgetPanel(topSplitter)
        topSplitter.SplitHorizontally(
            vSplitter, self.widget_panel, sashPosition=self.gui_size[1] * 0.83
        )  # 0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

        ###################################################################################################################################################
        # Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        self.load = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Load frames")
        widgetsizer.Add(self.load, 1, wx.ALL, 15)
        self.load.Bind(wx.EVT_BUTTON, self.browseDir)

        self.prev = wx.Button(self.widget_panel, id=wx.ID_ANY, label="<<Previous")
        widgetsizer.Add(self.prev, 1, wx.ALL, 15)
        self.prev.Bind(wx.EVT_BUTTON, self.prevImage)
        self.prev.Enable(False)

        self.next = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Next>>")
        widgetsizer.Add(self.next, 1, wx.ALL, 15)
        self.next.Bind(wx.EVT_BUTTON, self.nextImage)
        self.next.Enable(False)

        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        widgetsizer.Add(self.help, 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)
        self.help.Enable(True)
        self.zoom = wx.ToggleButton(self.widget_panel, label="Zoom")
        widgetsizer.Add(self.zoom, 1, wx.ALL, 15)
        self.zoom.Bind(wx.EVT_TOGGLEBUTTON, self.zoomButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.zoom.Enable(False)

        self.home = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Home")
        widgetsizer.Add(self.home, 1, wx.ALL, 15)
        self.home.Bind(wx.EVT_BUTTON, self.homeButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.home.Enable(False)

        self.pan = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Pan")
        widgetsizer.Add(self.pan, 1, wx.ALL, 15)
        self.pan.Bind(wx.EVT_TOGGLEBUTTON, self.panButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.pan.Enable(False)

        self.lock = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Lock View")
        widgetsizer.Add(self.lock, 1, wx.ALL, 15)
        self.lock.Bind(wx.EVT_CHECKBOX, self.lockChecked)
        self.widget_panel.SetSizer(widgetsizer)
        self.lock.Enable(False)

        self.save = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Save")
        widgetsizer.Add(self.save, 1, wx.ALL, 15)
        self.save.Bind(wx.EVT_BUTTON, self.saveDataSet)
        self.save.Enable(False)

        widgetsizer.AddStretchSpacer(15)
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit, 1, wx.ALL, 15)
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
        self.flag = True
        self.config_file = config
        self.new_labels = False
        self.buttonCounter = []
        self.bodyparts2plot = []
        self.drs = []
        self.num = []
        self.are_unique_bodyparts_present = True
        self.view_locked = False
        # Workaround for MAC - xlim and ylim changed events seem to be triggered too often so need to make sure that the
        # xlim and ylim have actually changed before turning zoom off
        self.prezoom_xlim = []
        self.prezoom_ylim = []

    ###############################################################################################################################
    # BUTTONS FUNCTIONS FOR HOTKEYS
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)
        elif event.GetKeyCode() == wx.WXK_LEFT:
            self.prevImage(event=None)
        elif event.GetKeyCode() == wx.WXK_DOWN:
            self.nextLabel(event=None)
        elif event.GetKeyCode() == wx.WXK_UP:
            self.previousLabel(event=None)
        elif event.GetKeyCode() == wx.WXK_BACK:
            pos_abs = event.GetPosition()
            inv = self.axes.transData.inverted()
            pos_rel = list(inv.transform(pos_abs))
            y1, y2 = self.axes.get_ylim()
            pos_rel[1] = y1 - pos_rel[1] + y2  # Recall y-axis is inverted
            i = np.nanargmin(
                [self.calc_distance(*dp.point.center, *pos_rel) for dp in self.drs]
            )
            closest_dp = self.drs[i]
            msg = wx.MessageBox(
                f"Do you want to remove the label {closest_dp.individual_names}:{closest_dp.bodyParts}?",
                "Remove!",
                wx.YES_NO | wx.ICON_WARNING,
            )
            if msg == 2:
                closest_dp.delete_data()
                self.buttonCounter[closest_dp.individual_names].remove(
                    closest_dp.bodyParts
                )
        elif event.ControlDown() and event.GetKeyCode() == 67:
            self.duplicate_labels()

    def duplicate_labels(self):
        if self.iter >= 1:
            curr_individual = self.individualrdb.GetStringSelection()
            curr_image = self.relativeimagenames[self.iter]
            prev_image = self.relativeimagenames[self.iter - 1]
            idx = pd.IndexSlice
            self.dataFrame.loc[
                curr_image, idx[:, curr_individual]
            ] = self.dataFrame.loc[prev_image, idx[:, curr_individual]].values
            img_name = Path(self.index[self.iter]).name
            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
                self.image_axis,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.multibodyparts,
                self.colormap,
                keep_view=self.view_locked,
            )
            if curr_individual == "single":
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.uniquebodyparts
                )
                self.buttonCounter = MainFrame.plot(self, self.img)
            else:
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.multibodyparts
                )
                self.buttonCounter = MainFrame.plot(self, self.img)

    def activateSlider(self, event):
        """
        Activates the slider to increase the markersize
        """
        self.checkSlider = event.GetEventObject()
        if self.checkSlider.GetValue():
            self.activate_slider = True
            self.change_marker_size.Enable(True)
            MainFrame.updateZoomPan(self)
        else:
            self.change_marker_size.Enable(False)

    def OnSliderScroll(self, event):
        """
        Adjust marker size for plotting the annotations
        """
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
        self.updatedCoords = []
        self.markerSize = self.change_marker_size.GetValue()
        self.edgewidth = self.markerSize // 3
        img_name = Path(self.index[self.iter]).name
        (
            self.figure,
            self.axes,
            self.canvas,
            self.toolbar,
            self.image_axis,
        ) = self.image_panel.drawplot(
            self.img,
            img_name,
            self.iter,
            self.index,
            self.multibodyparts,
            self.colormap,
            keep_view=True,
        )
        self.axes.callbacks.connect("xlim_changed", self.onZoom)
        self.axes.callbacks.connect("ylim_changed", self.onZoom)
        self.buttonCounter = MainFrame.plot(self, self.img)

    def quitButton(self, event):
        """
        Asks user for its inputs and then quits the GUI
        """
        #        MainFrame.saveDataSet(self, event)
        self.statusbar.SetStatusText("Quitting now!")

        nextFilemsg = wx.MessageBox(
            "Do you want to label another data set?",
            "Repeat?",
            wx.YES_NO | wx.ICON_INFORMATION,
        )
        if nextFilemsg == 2:
            self.file = 1
            self.buttonCounter = {i: [] for i in self.individual_names}
            self.updatedCoords = []
            self.dataFrame = None
            self.multibodyparts = []
            self.new_labels = self.new_labels
            self.axes.clear()
            self.figure.delaxes(self.figure.axes[1])
            self.choiceBox.Clear(True)
            MainFrame.updateZoomPan(self)
            MainFrame.browseDir(self, event)
        else:
            self.Destroy()
            print(
                "You can now check the labels, using 'check_labels' before proceeding. Then,  you can use the function 'create_training_dataset' to create the training dataset."
            )

    def helpButton(self, event):
        """
        Opens Instructions
        """
        MainFrame.updateZoomPan(self)
        wx.MessageBox(
            "1. Select an individual and one of the body parts from the radio buttons to add a label (if necessary change config.yaml first to edit the label names). \n\n2. Right clicking on the image will add the selected label and the next available label will be selected from the radio button. \n The label will be marked as circle filled with a unique color (and individual ID a unique color on the rim).\n\n3. To change the marker size, mark the checkbox and move the slider, then uncheck the box. \n\n4. Hover your mouse over this newly added label to see its name. \n\n5. Use left click and drag to move the label position.  \n\n6. Once you are happy with the position, right click to add the next available label. You can always reposition the old labels, if required. You can delete a label with the middle button mouse click (or click 'delete' key). \n\n7. Click Next/Previous to move to the next/previous image (or hot-key arrows left and right).\n User can also re-label a deleted point by going to a previous/next image then returning to the current image. \n NOTE: the user cannot add a label if the label is already present. \n \n8. You can click Cntrl+C to copy+paste labels from a previous image into the current image. For maDLC, you do this for each individual. \n\n9. When finished labeling all the images, click 'Save' to save all the labels as a .h5 file. \n\n10. Click OK to continue using the labeling GUI. For more tips and hotkeys: see docs!!",
            "User instructions",
            wx.OK | wx.ICON_INFORMATION,
        )
        self.statusbar.SetStatusText("Help")

    def onButtonRelease(self, event):
        if self.pan.GetValue():
            self.updateZoomPan()
            self.statusbar.SetStatusText("Pan Off")

    def onClick(self, event):
        """
        This function adds labels and auto advances to the next label.
        """
        x1 = event.xdata
        y1 = event.ydata
        if event.button == 3:
            num_indiv = self.individualrdb.GetSelection()
            indiv = self.individual_names[num_indiv]
            idcolor = self.idmap(num_indiv)
            if self.individualrdb.GetStringSelection() == "single":
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.uniquebodyparts
                )
                if (
                    self.uniquebodyparts[self.rdb.GetSelection()]
                    in self.buttonCounter[indiv]
                ):
                    wx.MessageBox(
                        "%s is already annotated for %s. \n Select another body part to annotate."
                        % (
                            str(self.uniquebodyparts[self.rdb.GetSelection()]),
                            str(
                                self.individual_names[self.individualrdb.GetSelection()]
                            ),
                        ),
                        "Error!",
                        wx.OK | wx.ICON_ERROR,
                    )
                else:
                    color = self.colormap(
                        self.norm(self.colorIndex[self.rdb.GetSelection()])
                    )
                    circle = [
                        patches.Circle(
                            (x1, y1),
                            radius=self.markerSize,
                            fc=color,
                            ec=idcolor,
                            lw=self.edgewidth,
                            alpha=self.alpha,
                        )
                    ]
                    self.num.append(circle)
                    self.axes.add_patch(circle[0])
                    self.dr = auxfun_drag.DraggablePoint(
                        circle[0],
                        self.uniquebodyparts[self.rdb.GetSelection()],
                        individual_names=indiv,
                    )
                    self.dr.connect()
                    self.buttonCounter[indiv].append(
                        self.uniquebodyparts[self.rdb.GetSelection()]
                    )
                    self.dr.coords = [
                        [x1, y1, indiv, self.uniquebodyparts[self.rdb.GetSelection()]]
                    ]
                    self.drs.append(self.dr)
                    self.updatedCoords.append(self.dr.coords)

                    if self.rdb.GetSelection() < len(self.uniquebodyparts) - 1:
                        self.rdb.SetSelection(self.rdb.GetSelection() + 1)
                    else:
                        self.rdb.SetSelection(0)
                        if (
                            self.individualrdb.GetSelection()
                            < len(self.individual_names) - 1
                        ):
                            self.individualrdb.SetSelection(
                                self.individualrdb.GetSelection() + 1
                            )
                        MainFrame.select_individual(self, event)

            else:
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.multibodyparts
                )
                if (
                    self.multibodyparts[self.rdb.GetSelection()]
                    in self.buttonCounter[indiv]
                ):
                    wx.MessageBox(
                        "%s is already annotated for %s. \n Select another body part to annotate."
                        % (
                            str(self.multibodyparts[self.rdb.GetSelection()]),
                            str(
                                self.individual_names[self.individualrdb.GetSelection()]
                            ),
                        ),
                        "Error!",
                        wx.OK | wx.ICON_ERROR,
                    )
                else:
                    color = self.colormap(
                        self.norm(self.colorIndex[self.rdb.GetSelection()])
                    )
                    circle = [
                        patches.Circle(
                            (x1, y1),
                            radius=self.markerSize,
                            fc=color,
                            ec=idcolor,
                            lw=self.edgewidth,
                            alpha=self.alpha,
                        )
                    ]
                    self.num.append(circle)
                    self.axes.add_patch(circle[0])
                    self.dr = auxfun_drag.DraggablePoint(
                        circle[0],
                        self.multibodyparts[self.rdb.GetSelection()],
                        individual_names=indiv,
                    )
                    self.dr.connect()
                    self.buttonCounter[indiv].append(
                        self.multibodyparts[self.rdb.GetSelection()]
                    )
                    self.dr.coords = [
                        [x1, y1, indiv, self.multibodyparts[self.rdb.GetSelection()]]
                    ]
                    self.drs.append(self.dr)
                    self.updatedCoords.append(self.dr.coords)

                    if self.rdb.GetSelection() < len(self.multibodyparts) - 1:
                        self.rdb.SetSelection(self.rdb.GetSelection() + 1)
                    else:
                        self.rdb.SetSelection(0)
                        if (
                            self.individualrdb.GetSelection()
                            < len(self.individual_names) - 1
                        ):
                            self.individualrdb.SetSelection(
                                self.individualrdb.GetSelection() + 1
                            )
                        MainFrame.select_individual(self, event)
        self.canvas.mpl_disconnect(self.onClick)

    def nextLabel(self, event):
        """
        This function is to create a hotkey to skip down on the radio button panel.
        """
        if self.rdb.GetSelection() < len(self.multibodyparts) - 1:
            self.rdb.SetSelection(self.rdb.GetSelection() + 1)

    def previousLabel(self, event):
        """
        This function is to create a hotkey to skip up on the radio button panel.
        """
        if self.rdb.GetSelection() > 0:
            self.rdb.SetSelection(self.rdb.GetSelection() - 1)

    def browseDir(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """
        if self.jump_unlabeled:
            self.dir = str(auxiliaryfunctions.find_next_unlabeled_folder(
                self.config_file
            ))
        else:
            self.statusbar.SetStatusText("Looking for a folder to start labeling...")
            cwd = os.path.join(os.getcwd(), "labeled-data")
            dlg = wx.DirDialog(
                self,
                "Choose the directory where your extracted frames are saved:",
                cwd,
                style=wx.DD_DEFAULT_STYLE,
            )
            if dlg.ShowModal() != wx.ID_OK:
                dlg.Destroy()
                self.Close(True)
                return
            self.dir = dlg.GetPath()
            dlg.Destroy()

        self.load.Enable(False)
        self.next.Enable(True)
        self.save.Enable(True)

        # Enabling the zoom, pan and home buttons
        self.zoom.Enable(True)
        self.home.Enable(True)
        self.pan.Enable(True)
        self.lock.Enable(True)

        # Reading config file and its variables
        self.cfg = auxiliaryfunctions.read_config(self.config_file)
        self.scorer = self.cfg["scorer"]
        (
            individuals,
            uniquebodyparts,
            multianimalbodyparts,
        ) = auxfun_multianimal.extractindividualsandbodyparts(self.cfg)

        self.multibodyparts = multianimalbodyparts
        # checks for unique bodyparts
        if len(self.multibodyparts) != len(set(self.multibodyparts)):
            print(
                "Error - bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again. Quitting for now!"
            )
            self.Close(True)

        self.uniquebodyparts = uniquebodyparts
        self.individual_names_ = individuals
        self.individual_names = self.individual_names_.copy()

        self.videos = self.cfg["video_sets"].keys()
        self.markerSize = self.cfg["dotsize"]
        self.edgewidth = self.markerSize // 3
        self.alpha = self.cfg["alphavalue"]
        self.colormap = plt.get_cmap(self.cfg["colormap"])
        self.colormap = self.colormap.reversed()
        self.idmap = plt.cm.get_cmap("Set1", len(individuals))
        self.project_path = self.cfg["project_path"]

        if not self.uniquebodyparts:
            self.are_unique_bodyparts_present = False

        self.buttonCounter = {i: [] for i in self.individual_names}
        self.index = np.sort(
            [
                fn
                for fn in glob.glob(os.path.join(self.dir, "*.png"))
                if ("labeled.png" not in fn)
            ]
        )
        self.statusbar.SetStatusText(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        relativeimagenames = [
            "labeled" + n.split("labeled")[1] for n in self.index
        ]  # [n.split(self.project_path+'/')[1] for n in self.index]
        self.relativeimagenames = [tuple(name.split(os.path.sep))
                                   for name in relativeimagenames]
        # Reading the existing dataset,if already present
        try:
            self.dataFrame = pd.read_hdf(
                os.path.join(self.dir, "CollectedData_" + self.scorer + ".h5")
            )
            conversioncode.guarantee_multiindex_rows(self.dataFrame)
            self.dataFrame.sort_index(inplace=True)
            self.prev.Enable(True)
            # Finds the first empty row in the dataframe and sets the iteration to that index
            self.iter = np.argmax(np.isnan(self.dataFrame.values).all(axis=1))
        except FileNotFoundError:
            # Create an empty data frame
            self.dataFrame = MainFrame.create_dataframe(
                self,
                self.dataFrame,
                self.relativeimagenames,
                self.individual_names,
                self.uniquebodyparts,
                self.multibodyparts,
            )
            self.iter = 0

        # Cache original bodyparts
        self._old_multi = (
            self.dataFrame.xs(self.individual_names[0], axis=1, level="individuals")
            .columns.get_level_values("bodyparts")
            .unique()
            .to_list()
        )
        self._old_unique = (
            self.dataFrame.loc[
                :, self.dataFrame.columns.get_level_values("individuals") == "single"
            ]
            .columns.get_level_values("bodyparts")
            .unique()
            .to_list()
        )

        # Reading the image name
        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name

        # Checking for new frames and adding them to the existing dataframe
        old_imgs = sorted(self.dataFrame.index)
        self.newimages = list(set(self.relativeimagenames) - set(old_imgs))
        if self.newimages:
            print("Found new frames..")
            # Create an empty dataframe with all the new images and then merge this to the existing dataframe.
            self.df = MainFrame.create_dataframe(
                self,
                None,
                self.newimages,
                self.individual_names,
                self.uniquebodyparts,
                self.multibodyparts,
            )
            self.dataFrame = pd.concat([self.dataFrame, self.df], axis=0)
            self.dataFrame.sort_index(inplace=True)
            # Rearrange bodypart columns in config order
            bodyparts = self.multibodyparts + self.uniquebodyparts
            self.dataFrame.reindex(
                bodyparts, axis=1, level=self.dataFrame.columns.names.index("bodyparts")
            )
        # Test whether there are missing frames and superfluous data
        if len(old_imgs) > len(self.relativeimagenames):
            missing_frames = set(old_imgs).difference(self.relativeimagenames)
            self.dataFrame.drop(missing_frames, inplace=True)

        # Check whether new individuals or body parts were added
        old_animals = self.dataFrame.columns.get_level_values("individuals").unique()
        self.new_animals = [x for x in self.individual_names if x not in old_animals]
        self.new_multi = [x for x in self.multibodyparts if x not in self._old_multi]
        self.new_unique = [x for x in self.uniquebodyparts if x not in self._old_unique]

        if self.new_animals:
            dlg = wx.MessageDialog(
                None,
                "New individual found in the config file. Do you want to see all the other ones?",
                "New individual found",
                wx.YES_NO | wx.ICON_WARNING,
            )
            result = dlg.ShowModal()
            if result == wx.ID_NO:
                self.individual_names = self.new_animals
            self.dataFrame = MainFrame.create_dataframe(
                self,
                self.dataFrame,
                self.relativeimagenames,
                self.new_animals,
                self._old_unique,
                self._old_multi,
            )

        # Checking if user added a new label
        if not any([self.new_multi, self.new_unique]):  # i.e. no new labels
            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
                self.image_axis,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.multibodyparts,
                self.colormap,
                keep_view=self.view_locked,
            )
        else:
            # Found new labels in either multiple bodyparts or unique bodyparts
            dlg = wx.MessageDialog(
                None,
                "New label found in the config file. Do you want to see all the other labels?",
                "New label found",
                wx.YES_NO | wx.ICON_WARNING,
            )
            result = dlg.ShowModal()
            if result == wx.ID_NO:
                if self.new_multi:
                    self.multibodyparts = self.new_multi
                if self.new_unique:
                    self.uniquebodyparts = self.new_unique

            self.dataFrame = MainFrame.create_dataframe(
                self,
                self.dataFrame,
                self.relativeimagenames,
                self.dataFrame.columns.get_level_values("individuals").unique(),
                self.new_unique,
                self.new_multi,
            )
            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
                self.image_axis,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.multibodyparts,
                self.colormap,
                keep_view=self.view_locked,
            )

        self.axes.callbacks.connect("xlim_changed", self.onZoom)
        self.axes.callbacks.connect("ylim_changed", self.onZoom)

        if self.individual_names[0] == "single":
            (
                self.choiceBox,
                self.individualrdb,
                self.rdb,
                self.change_marker_size,
                self.checkBox,
            ) = self.choice_panel.addRadioButtons(
                self.uniquebodyparts, self.individual_names, self.file, self.markerSize
            )
            self.image_panel.addcolorbar(
                self.img,
                self.image_axis,
                self.iter,
                self.uniquebodyparts,
                self.colormap,
            )
        else:
            (
                self.choiceBox,
                self.individualrdb,
                self.rdb,
                self.change_marker_size,
                self.checkBox,
            ) = self.choice_panel.addRadioButtons(
                self.multibodyparts, self.individual_names, self.file, self.markerSize
            )
            self.image_panel.addcolorbar(
                self.img, self.image_axis, self.iter, self.multibodyparts, self.colormap
            )
        self.individualrdb.Bind(wx.EVT_RADIOBOX, self.select_individual)
        # check if single is selected when radio buttons are changed
        if self.individualrdb.GetStringSelection() == "single":
            self.norm, self.colorIndex = self.image_panel.getColorIndices(
                self.img, self.uniquebodyparts
            )
        else:
            self.norm, self.colorIndex = self.image_panel.getColorIndices(
                self.img, self.multibodyparts
            )
        self.buttonCounter = MainFrame.plot(self, self.img)
        self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)

        self.checkBox.Bind(wx.EVT_CHECKBOX, self.activateSlider)
        self.change_marker_size.Bind(wx.EVT_SLIDER, self.OnSliderScroll)

    def create_dataframe(
        self,
        dataFrame,
        relativeimagenames,
        individual_names,
        uniquebodyparts,
        multibodyparts,
    ):
        a = np.empty((len(relativeimagenames), 2))
        a[:] = np.nan
        for prfxindex, prefix in enumerate(individual_names):
            if uniquebodyparts is not None:
                if prefix == "single":
                    for c, bp in enumerate(uniquebodyparts):
                        cols = pd.MultiIndex.from_product(
                            [[self.scorer], [prefix], [bp], ["x", "y"]],
                            names=["scorer", "individuals", "bodyparts", "coords"],
                        )
                        index = pd.MultiIndex.from_tuples(relativeimagenames)
                        frame = pd.DataFrame(a, columns=cols, index=index)
                        dataFrame = pd.concat([dataFrame, frame], axis=1)
                else:
                    for c, bp in enumerate(multibodyparts):
                        cols = pd.MultiIndex.from_product(
                            [[self.scorer], [prefix], [bp], ["x", "y"]],
                            names=["scorer", "individuals", "bodyparts", "coords"],
                        )
                        index = pd.MultiIndex.from_tuples(relativeimagenames)
                        frame = pd.DataFrame(a, columns=cols, index=index)
                        dataFrame = pd.concat([dataFrame, frame], axis=1)
            else:
                for c, bp in enumerate(multibodyparts):
                    cols = pd.MultiIndex.from_product(
                        [[self.scorer], [prefix], [bp], ["x", "y"]],
                        names=["scorer", "individuals", "bodyparts", "coords"],
                    )
                    index = pd.MultiIndex.from_tuples(relativeimagenames)
                    frame = pd.DataFrame(a, columns=cols, index=index)
                    dataFrame = pd.concat([dataFrame, frame], axis=1)
        dataFrame.sort_index(inplace=True)
        return dataFrame

    def select_individual(self, event):
        individualName = self.individualrdb.GetStringSelection()
        self.change_marker_size.Hide()
        self.change_marker_size.Destroy()
        if individualName == "single":
            self.checkBox.Hide()
            self.individualrdb.Hide()
            self.rdb.Hide()
            (
                self.choiceBox,
                self.individualrdb,
                self.rdb,
                self.change_marker_size,
                self.checkBox,
            ) = self.choice_panel.addRadioButtons(
                self.uniquebodyparts, self.individual_names, self.file, self.markerSize
            )
            self.individualrdb.SetStringSelection(individualName)
            self.individualrdb.Bind(wx.EVT_RADIOBOX, self.select_individual)
            self.figure.delaxes(self.figure.axes[1])
            self.image_panel.addcolorbar(
                self.img,
                self.image_axis,
                self.iter,
                self.uniquebodyparts,
                self.colormap,
            )
            self.checkBox.Bind(wx.EVT_CHECKBOX, self.activateSlider)
            self.change_marker_size.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
        else:
            self.checkBox.Hide()
            self.individualrdb.Hide()
            self.rdb.Hide()
            (
                self.choiceBox,
                self.individualrdb,
                self.rdb,
                self.change_marker_size,
                self.checkBox,
            ) = self.choice_panel.addRadioButtons(
                self.multibodyparts, self.individual_names, self.file, self.markerSize
            )
            self.individualrdb.SetStringSelection(individualName)
            self.change_marker_size.Show()
            self.checkBox.Show()
            self.individualrdb.Show()
            self.rdb.Show()
            self.individualrdb.Bind(wx.EVT_RADIOBOX, self.select_individual)
            self.figure.delaxes(self.figure.axes[1])
            self.image_panel.addcolorbar(
                self.img, self.image_axis, self.iter, self.multibodyparts, self.colormap
            )
            self.checkBox.Bind(wx.EVT_CHECKBOX, self.activateSlider)
            self.change_marker_size.Bind(wx.EVT_SLIDER, self.OnSliderScroll)

    def nextImage(self, event):
        """
        Moves to next image
        """
        self.individualrdb.SetSelection(0)
        MainFrame.select_individual(self, event)
        #  Checks for the last image and disables the Next button
        if len(self.index) - self.iter == 1:
            self.next.Enable(False)
            return
        self.prev.Enable(True)

        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)

        self.statusbar.SetStatusText(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        self.rdb.SetSelection(0)
        self.individualrdb.SetSelection(0)
        self.file = 1
        # Refreshing the button counters
        self.buttonCounter = {i: [] for i in self.individual_names}
        MainFrame.saveEachImage(self)
        self.iter = self.iter + 1

        if len(self.index) >= self.iter:
            self.updatedCoords = []
            self.img = self.index[self.iter]
            img_name = Path(self.index[self.iter]).name
            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
                self.image_axis,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.multibodyparts,
                self.colormap,
                keep_view=self.view_locked,
            )
            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)
            if self.individualrdb.GetStringSelection() == "single":
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.uniquebodyparts
                )
                self.buttonCounter = MainFrame.plot(self, self.img)
            else:
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.multibodyparts
                )
                self.buttonCounter = MainFrame.plot(self, self.img)

            self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)

    def prevImage(self, event):
        """
        Checks the previous Image and enables user to move the annotations.
        """
        self.individualrdb.SetSelection(0)
        MainFrame.select_individual(self, event)
        MainFrame.saveEachImage(self)
        # Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.prev.Enable(False)
            return
        else:
            self.next.Enable(True)
        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        self.statusbar.SetStatusText(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        self.buttonCounter = {i: [] for i in self.individual_names}
        self.iter = self.iter - 1

        self.rdb.SetSelection(0)
        self.individualrdb.SetSelection(0)
        self.updatedCoords = []
        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name
        (
            self.figure,
            self.axes,
            self.canvas,
            self.toolbar,
            self.image_axis,
        ) = self.image_panel.drawplot(
            self.img,
            img_name,
            self.iter,
            self.index,
            self.multibodyparts,
            self.colormap,
            keep_view=self.view_locked,
        )
        self.axes.callbacks.connect("xlim_changed", self.onZoom)
        self.axes.callbacks.connect("ylim_changed", self.onZoom)
        self.buttonCounter = MainFrame.plot(self, self.img)

        self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)

    def plot(self, img):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """
        self.drs = []
        self.updatedCoords = []
        if len(self.individual_names) > 1:
            self.norm, self.colorIndex = self.image_panel.getColorIndices(
                self.img, self.multibodyparts
            )
        for j, ind in enumerate(self.individual_names):
            idcolor = self.idmap(j)
            if ind not in self.dataFrame.columns.get_level_values(1):
                continue
            if ind == "single":
                for c, bp in enumerate(self.uniquebodyparts):
                    image_points = [
                        [
                            self.dataFrame[self.scorer][ind][bp]["x"].values[self.iter],
                            self.dataFrame[self.scorer][ind][bp]["y"].values[self.iter],
                            ind,
                            bp,
                        ]
                    ]
                    self.points = [
                        self.dataFrame[self.scorer][ind][bp]["x"].values[self.iter],
                        self.dataFrame[self.scorer][ind][bp]["y"].values[self.iter],
                    ]
                    self.norm, self.colorIndex = self.image_panel.getColorIndices(
                        self.img, self.uniquebodyparts
                    )
                    color = self.colormap(self.norm(self.colorIndex[c]))
                    circle = patches.Circle(
                        (self.points[0], self.points[1]),
                        radius=self.markerSize,
                        fc=color,
                        ec=idcolor,
                        lw=self.edgewidth,
                        alpha=self.alpha,
                    )
                    self.axes.add_patch(circle)
                    self.dr = auxfun_drag.DraggablePoint(
                        circle, self.uniquebodyparts[c], individual_names=ind
                    )
                    self.dr.connect()
                    self.dr.coords = image_points
                    self.drs.append(self.dr)
                    self.updatedCoords.append(self.dr.coords)
                    if not np.isnan(self.points)[0]:
                        self.buttonCounter[ind].append(self.uniquebodyparts[c])
            else:
                for c, bp in enumerate(self.multibodyparts):
                    image_points = [
                        [
                            self.dataFrame[self.scorer][ind][bp]["x"].values[self.iter],
                            self.dataFrame[self.scorer][ind][bp]["y"].values[self.iter],
                            ind,
                            bp,
                        ]
                    ]
                    self.points = [
                        self.dataFrame[self.scorer][ind][bp]["x"].values[self.iter],
                        self.dataFrame[self.scorer][ind][bp]["y"].values[self.iter],
                    ]
                    color = self.colormap(self.norm(self.colorIndex[c]))
                    circle = patches.Circle(
                        (self.points[0], self.points[1]),
                        radius=self.markerSize,
                        fc=color,
                        ec=idcolor,
                        lw=self.edgewidth,
                        alpha=self.alpha,
                    )
                    self.axes.add_patch(circle)
                    self.dr = auxfun_drag.DraggablePoint(
                        circle, self.multibodyparts[c], individual_names=ind
                    )
                    self.dr.connect()
                    self.dr.coords = image_points
                    self.drs.append(self.dr)
                    self.updatedCoords.append(self.dr.coords)
                    if not np.isnan(self.points)[0]:
                        self.buttonCounter[ind].append(self.multibodyparts[c])
        MainFrame.saveEachImage(self)
        self.figure.canvas.draw()
        return self.buttonCounter

    def saveEachImage(self):
        """
        Saves data for each image
        """

        for idx, bp in enumerate(self.updatedCoords):

            x, y, ind = bp[-1][:3]
            bpt = bp[0][-1]
            self.dataFrame.loc[
                self.relativeimagenames[self.iter], (self.scorer, ind, bpt, "x")
            ] = x
            self.dataFrame.loc[
                self.relativeimagenames[self.iter], (self.scorer, ind, bpt, "y")
            ] = y

    def saveDataSet(self, event):
        """
        Saves the final dataframe
        """
        self.statusbar.SetStatusText("File saved")
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)

        # Windows compatible
        self.dataFrame.sort_index(inplace=True)
        # Discard data associated with bodyparts that are no longer in the config
        config_bpts = self.cfg["multianimalbodyparts"] + self.cfg["uniquebodyparts"]
        valid = [
            bp in config_bpts
            for bp in self.dataFrame.columns.get_level_values("bodyparts")
        ]
        self.dataFrame = self.dataFrame.loc[:, valid]
        # Re-organize the dataframe so the CSV looks consistent with the config
        self.dataFrame = self.dataFrame.reindex(
            columns=self.individual_names_, level="individuals"
        ).reindex(columns=config_bpts, level="bodyparts")
        self.dataFrame.to_csv(
            os.path.join(self.dir, "CollectedData_" + self.scorer + ".csv")
        )
        self.dataFrame.to_hdf(
            os.path.join(self.dir, "CollectedData_" + self.scorer + ".h5"),
            "df_with_missing",
        )

    def onChecked(self, event):
        self.cb = event.GetEventObject()
        if self.cb.GetValue():
            self.change_marker_size.Enable(True)
            self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)
        else:
            self.change_marker_size.Enable(False)


def show(config, config3d, sourceCam, jump_unlabeled=False):
    app = wx.App()
    frame = MainFrame(None, config, config3d, sourceCam, jump_unlabeled).Show()
    app.MainLoop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("config3d")
    parser.add_argument("sourceCam")
    cli_args = parser.parse_args()
