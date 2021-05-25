"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCutDeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import os
import os.path
import platform
from pathlib import Path

# from skimage import io
import PIL
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
from skimage import io

from deeplabcut.gui import auxfun_drag
from deeplabcut.gui.widgets import BasePanel, WidgetPanel, BaseFrame
from deeplabcut.utils import auxiliaryfunctions, visualization


# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(BasePanel):
    def drawplot(
        self, img, img_name, itr, index, threshold, cmap, preview, keep_view=False
    ):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.clear()
        im = io.imread(img)
        self.ax = self.axes.imshow(im, cmap=cmap)
        self.orig_xlim = self.axes.get_xlim()
        self.orig_ylim = self.axes.get_ylim()
        if not preview:
            self.axes.set_title(
                str(
                    str(itr)
                    + "/"
                    + str(len(index) - 1)
                    + " "
                    + str(Path(index[itr]).stem)
                    + " "
                    + " Threshold chosen is: "
                    + str("{0:.2f}".format(threshold))
                )
            )
        else:
            self.axes.set_title(
                str(
                    str(itr)
                    + "/"
                    + str(len(index) - 1)
                    + " "
                    + str(Path(index[itr]).stem)
                )
            )
        if keep_view:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        self.figure.canvas.draw()
        if not hasattr(self, "toolbar"):
            self.toolbar = NavigationToolbar(self.canvas)
        return (self.figure, self.axes, self.canvas, self.toolbar, self.ax)

    def getColorIndices(self, img, bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        im = io.imread(img)
        norm = mcolors.Normalize(vmin=np.min(im), vmax=np.max(im))
        ticks = np.linspace(np.min(im), np.max(im), len(bodyparts))[::-1]
        return norm, ticks


class ScrollPanel(SP.ScrolledPanel):
    def __init__(self, parent):
        SP.ScrolledPanel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
        self.Layout()

    def on_focus(self, event):
        pass

    def addCheckBoxSlider(self, bodyparts, fileIndex, markersize):
        """
        Adds checkbox and a slider
        """
        self.choiceBox = wx.BoxSizer(wx.VERTICAL)

        self.slider = wx.Slider(
            self,
            -1,
            markersize,
            1,
            markersize * 3,
            size=(250, -1),
            style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.slider.Enable(False)
        self.checkBox = wx.CheckBox(self, id=wx.ID_ANY, label="Adjust marker size.")
        self.choiceBox.Add(self.slider, 0, wx.ALL, 5)
        self.choiceBox.Add(self.checkBox, 0, wx.ALL, 5)
        names = ["Color individuals", "Color bodyparts"]
        self.visualization_radiobox = wx.RadioBox(
            self,
            label="Select the visualization scheme",
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
            choices=names,
        )
        self.choiceBox.Add(self.visualization_radiobox, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizerAndFit(self.choiceBox)
        self.Layout()
        return (self.choiceBox, self.slider, self.checkBox, self.visualization_radiobox)

    def clearBoxer(self):
        self.choiceBox.Clear(True)


class MainFrame(BaseFrame):
    def __init__(self, parent, config):
        super(MainFrame, self).__init__("DeepLabCut - Refinement ToolBox", parent)
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

        ###################################################################################################################################################

        # Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!

        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)

        self.image_panel = ImagePanel(vSplitter, config, self.gui_size)
        self.choice_panel = ScrollPanel(vSplitter)
        # self.choice_panel.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
        # self.choice_panel.SetupScrolling(scroll_x=True, scrollToTop=False)
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
        self.load = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Load labels")
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
        # Variable initialization
        self.currentDirectory = os.getcwd()
        self.index = []
        self.iter = []
        self.threshold = []
        self.file = 0
        self.updatedCoords = []
        self.drs = []
        self.cfg = auxiliaryfunctions.read_config(config)
        self.humanscorer = self.cfg["scorer"]
        self.move2corner = self.cfg["move2corner"]
        self.center = self.cfg["corner2move2"]
        self.colormap = plt.get_cmap(self.cfg["colormap"])
        self.colormap = self.colormap.reversed()
        self.markerSize = self.cfg["dotsize"]
        self.alpha = self.cfg["alphavalue"]
        self.iterationindex = self.cfg["iteration"]
        self.project_path = self.cfg["project_path"]
        self.bodyparts = self.cfg["bodyparts"]
        self.threshold = 0.1
        self.img_size = (10, 6)  # (imgW, imgH)  # width, height in inches.
        self.preview = False
        self.view_locked = False
        # Workaround for MAC - xlim and ylim changed events seem to be triggered too often so need to make sure that the
        # xlim and ylim have actually changed before turning zoom off
        self.prezoom_xlim = []
        self.prezoom_ylim = []
        from deeplabcut.utils import auxfun_multianimal

        (
            self.individual_names,
            self.uniquebodyparts,
            self.multianimalbodyparts,
        ) = auxfun_multianimal.extractindividualsandbodyparts(self.cfg)
        # self.choiceBox,self.visualization_rdb = self.choice_panel.addRadioButtons()
        self.Colorscheme = visualization.get_cmap(
            len(self.individual_names), self.cfg["colormap"]
        )

    # ###########################################################################
    # functions for button responses
    # ###########################################################################
    # BUTTONS FUNCTIONS FOR HOTKEYS
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)
        elif event.GetKeyCode() == wx.WXK_LEFT:
            self.prevImage(event=None)
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
                f"Do you want to remove the label {closest_dp.individual_name}:{closest_dp.bodyParts}?",
                "Remove!",
                wx.YES_NO | wx.ICON_WARNING,
            )
            if msg == 2:
                closest_dp.delete_data()

    def activateSlider(self, event):
        """
        Activates the slider to increase the markersize
        """
        self.checkSlider = event.GetEventObject()
        if self.checkSlider.GetValue():
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
        #        self.axes.clear()
        self.figure.delaxes(self.figure.axes[1])
        (
            self.figure,
            self.axes,
            self.canvas,
            self.toolbar,
            self.ax,
        ) = self.image_panel.drawplot(
            self.img,
            img_name,
            self.iter,
            self.index,
            self.threshold,
            self.colormap,
            self.preview,
            keep_view=True,
        )
        self.axes.callbacks.connect("xlim_changed", self.onZoom)
        self.axes.callbacks.connect("ylim_changed", self.onZoom)

        MainFrame.plot(self, self.img)

    def browseDir(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """

        fname = str("machinelabels-iter" + str(self.iterationindex) + ".h5")
        self.statusbar.SetStatusText("Looking for a folder to start refining...")
        cwd = os.path.join(os.getcwd(), "labeled-data")
        #        dlg = wx.FileDialog(self, "Choose the machinelabels file for current iteration.",cwd, "",wildcard=fname,style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if platform.system() == "Darwin":
            dlg = wx.FileDialog(
                self,
                "Choose the machinelabels file for current iteration.",
                cwd,
                fname,
                wildcard="(*.h5)|*.h5",
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            )
        else:
            dlg = wx.FileDialog(
                self,
                "Choose the machinelabels file for current iteration.",
                cwd,
                "",
                wildcard=fname,
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            )

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
            self.lock.Enable(True)
        else:
            dlg.Destroy()
            self.Destroy()
            return
        dlg.Destroy()

        try:
            self.dataname = str(self.data_file)

        except:
            print("No machinelabels file found!")
            self.Destroy()
        self.statusbar.SetStatusText(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        self.preview = True
        self.iter = 0

        if os.path.isfile(self.dataname):
            self.Dataframe = pd.read_hdf(self.dataname)
            # Overwrite the config-defined individual names
            # with those actually present in the annotated data
            self.individual_names = (
                self.Dataframe
                    .columns
                    .get_level_values("individuals")
                    .unique()
                    .to_list()
            )
            self.Dataframe.sort_index(inplace=True)
            self.scorer = self.Dataframe.columns.get_level_values(0)[0]

            # bodyParts = self.Dataframe.columns.get_level_values(1)
            # _, idx = np.unique(bodyParts, return_index=True)
            # self.num_joints = len(self.bodyparts)
            # self.bodyparts =  bodyParts[np.sort(idx)]
            self.index = list(self.Dataframe.iloc[:, 0].index)
            # Reading images

            self.img = os.path.join(self.project_path, self.index[self.iter])
            img_name = Path(self.img).name
            self.norm, self.colorIndex = self.image_panel.getColorIndices(
                self.img, self.bodyparts
            )
            # Adding Slider and Checkbox

            (
                self.choiceBox,
                self.slider,
                self.checkBox,
                self.visualization_rdb,
            ) = self.choice_panel.addCheckBoxSlider(
                self.bodyparts, self.file, self.markerSize
            )
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.checkBox.Bind(wx.EVT_CHECKBOX, self.activateSlider)
            self.visualization_rdb.Bind(wx.EVT_RADIOBOX, self.clear_plot)
            self.slider.Enable(False)

            # take into account of all the bodyparts for the colorscheme. Sort the bodyparts to have same order as in the config file
            self.all_bodyparts = np.array(
                self.multianimalbodyparts + self.uniquebodyparts
            )
            _, return_idx = np.unique(self.all_bodyparts, return_index=True)
            self.all_bodyparts = list(self.all_bodyparts[np.sort(return_idx)])

            # Show image
            # Setting axis title:dont want to show the threshold as it is not selected yet.

            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
                self.ax,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.threshold,
                self.colormap,
                self.preview,
            )
            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)

            instruction = wx.MessageBox(
                "1. Enter the likelihood threshold. \n\n2. Each prediction will be shown with a unique color. \n All the data points above the threshold will be marked as circle filled with a unique color. All the data points below the threshold will be marked with a hollow circle. \n\n3. Enable the checkbox to adjust the marker size. \n\n4.  Hover your mouse over data points to see the labels and their likelihood. \n\n5. Left click and drag to move the data points. \n\n6. Right click on any data point to remove it. Be careful, you cannot undo this step. \n Click once on the zoom button to zoom-in the image.The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full;default view. \n\n7. When finished click 'Save' to save all the changes. \n\n8. Click OK to continue",
                "User instructions",
                wx.OK | wx.ICON_INFORMATION,
            )

            if instruction == 4:
                """
                If ok is selected then the image is updated with the thresholded value of the likelihood
                """
                textBox = wx.TextEntryDialog(
                    self,
                    "Select the likelihood threshold",
                    caption="Enter the threshold",
                    value="0.1",
                )
                textBox.ShowModal()
                self.threshold = float(textBox.GetValue())
                textBox.Destroy()
                self.img = os.path.join(self.project_path, self.index[self.iter])
                img_name = Path(self.img).name
                self.axes.clear()
                self.preview = False
                # self.figure.delaxes(self.figure.axes[1]) # Removes the axes corresponding to the colorbar
                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                )
                self.axes.callbacks.connect("xlim_changed", self.onZoom)
                self.axes.callbacks.connect("ylim_changed", self.onZoom)
                MainFrame.plot(self, self.img)
                MainFrame.saveEachImage(self)
            else:
                # self.figure.delaxes(self.figure.axes[1]) # Removes the axes corresponding to the colorbar
                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                )
                self.axes.callbacks.connect("xlim_changed", self.onZoom)
                self.axes.callbacks.connect("ylim_changed", self.onZoom)
                MainFrame.plot(self, self.img)
                MainFrame.saveEachImage(self)

        else:
            msg = wx.MessageBox(
                "No Machinelabels file found! Want to retry?",
                "Error!",
                wx.YES_NO | wx.ICON_WARNING,
            )
            if msg == 2:
                self.load.Enable(True)
                self.next.Enable(False)
                self.save.Enable(False)

    def clear_plot(self, event):
        MainFrame.saveEachImage(self)
        self.figure.delaxes(self.figure.axes[1])
        [p.remove() for p in reversed(self.axes.patches)]
        self.plot(self.img)

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
        # print(self.Dataframe.head())
        self.statusbar.SetStatusText(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )

        self.iter = self.iter + 1

        if len(self.index) > self.iter:
            self.updatedCoords = []
            self.img = os.path.join(self.project_path, self.index[self.iter])
            img_name = Path(self.img).name

            # Plotting
            self.figure.delaxes(
                self.figure.axes[1]
            )  # Removes the axes corresponding to the colorbar
            if self.visualization_rdb.GetSelection() == 0:
                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                    keep_view=self.view_locked,
                )
            else:
                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                    keep_view=self.view_locked,
                )
            im = io.imread(self.img)
            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)
            if np.max(im) == 0:
                msg = wx.MessageBox(
                    "Invalid image. Click Yes to remove",
                    "Error!",
                    wx.YES_NO | wx.ICON_WARNING,
                )
                if msg == 2:
                    self.Dataframe = self.Dataframe.drop(self.index[self.iter])
                    self.index = list(self.Dataframe.iloc[:, 0].index)
                self.iter = self.iter - 1

                self.img = os.path.join(self.project_path, self.index[self.iter])
                img_name = Path(self.img).name

                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                    keep_view=self.view_locked,
                )
                self.axes.callbacks.connect("xlim_changed", self.onZoom)
                self.axes.callbacks.connect("ylim_changed", self.onZoom)
            MainFrame.plot(self, self.img)
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

        self.statusbar.SetStatusText(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        self.next.Enable(True)
        self.iter = self.iter - 1

        # Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.prev.Enable(False)

        if self.iter >= 0:
            self.updatedCoords = []
            # Reading Image
            self.img = os.path.join(self.project_path, self.index[self.iter])
            img_name = Path(self.img).name

            # Plotting
            self.figure.delaxes(
                self.figure.axes[1]
            )  # Removes the axes corresponding to the colorbar
            if self.visualization_rdb.GetSelection() == 0:
                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                    keep_view=self.view_locked,
                )
            else:
                (
                    self.figure,
                    self.axes,
                    self.canvas,
                    self.toolbar,
                    self.ax,
                ) = self.image_panel.drawplot(
                    self.img,
                    img_name,
                    self.iter,
                    self.index,
                    self.threshold,
                    self.colormap,
                    self.preview,
                    keep_view=self.view_locked,
                )
            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)
            MainFrame.plot(self, self.img)
        else:
            self.prev.Enable(False)
        MainFrame.saveEachImage(self)

    def quitButton(self, event):
        """
        Quits the GUI
        """
        self.statusbar.SetStatusText("")
        dlg = wx.MessageDialog(
            None, "Are you sure?", "Quit!", wx.YES_NO | wx.ICON_WARNING
        )
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            print(
                "Closing... The refined labels are stored in a subdirectory under labeled-data. Use the function 'merge_datasets' to augment the training dataset, and then re-train a network using create_training_dataset followed by train_network!"
            )
            self.Destroy()
        else:
            self.save.Enable(True)

    def helpButton(self, event):
        """
        Opens Instructions
        """
        self.statusbar.SetStatusText("Help")
        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        wx.MessageBox(
            "1. Enter the likelihood threshold. \n\n2. All the data points above the threshold will be marked as circle filled with a unique color. All the data points below the threshold will be marked with a hollow circle. \n\n3. Enable the checkbox to adjust the marker size (you will not be able to zoom/pan/home until the next frame). \n\n4. Hover your mouse over data points to see the labels and their likelihood. \n\n5. LEFT click+drag to move the data points. \n\n6. RIGHT click on any data point to remove it. Be careful, you cannot undo this step! \n Click once on the zoom button to zoom-in the image. The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full default view. \n\n7. When finished click 'Save' to save all the changes. \n\n8. Click OK to continue",
            "User instructions",
            wx.OK | wx.ICON_INFORMATION,
        )

    def onChecked(self, event):
        MainFrame.saveEachImage(self)
        self.cb = event.GetEventObject()
        if self.cb.GetValue():
            self.slider.Enable(True)
        else:
            self.slider.Enable(False)

    def force_outside_labels_Nans(self, index, ind, bodyparts):
        """
        Checks the dataframe for any labels outside the image and forces them Nans.
        """
        for bpindex, bp in enumerate(bodyparts):
            testCondition = (
                self.Dataframe.loc[index, (self.scorer, ind, bp, "x")] > self.width
                or self.Dataframe.loc[index, (self.scorer, ind, bp, "x")] < 0
                or self.Dataframe.loc[index, (self.scorer, ind, bp, "y")] > self.height
                or self.Dataframe.loc[index, (self.scorer, ind, bp, "y")] < 0
            )
            if testCondition:
                print("Found %s outside the image %s.Setting it to NaN" % (bp, index))
                self.Dataframe.loc[index, (self.scorer, ind, bp, "x")] = np.nan
                self.Dataframe.loc[index, (self.scorer, ind, bp, "y")] = np.nan
        return self.Dataframe

    def check_labels(self):
        print("Checking labels if they are outside the image")
        for i in self.Dataframe.index:
            image_name = os.path.join(self.project_path, i)
            im = PIL.Image.open(image_name)
            self.width, self.height = im.size
            for ind in self.individual_names:
                if ind == "single":
                    self.Dataframe = MainFrame.force_outside_labels_Nans(
                        self, i, ind, self.uniquebodyparts
                    )
                else:
                    self.Dataframe = MainFrame.force_outside_labels_Nans(
                        self, i, ind, self.multianimalbodyparts
                    )
        return self.Dataframe

    def saveDataSet(self, event):

        MainFrame.saveEachImage(self)

        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        self.statusbar.SetStatusText("File saved")

        self.Dataframe = MainFrame.check_labels(self)
        # Overwrite machine label file
        self.Dataframe.to_hdf(self.dataname, key="df_with_missing", mode="w")

        self.Dataframe.columns.set_levels(
            [self.scorer.replace(self.scorer, self.humanscorer)], level=0, inplace=True
        )
        self.Dataframe = self.Dataframe.drop("likelihood", axis=1, level=3)

        if Path(self.dir, "CollectedData_" + self.humanscorer + ".h5").is_file():
            print(
                "A training dataset file is already found for this video. The refined machine labels are merged to this data!"
            )
            DataU1 = pd.read_hdf(
                os.path.join(self.dir, "CollectedData_" + self.humanscorer + ".h5")
            )
            # combine datasets Original Col. + corrected machinefiles:
            DataCombined = pd.concat([self.Dataframe, DataU1])
            # Now drop redundant ones keeping the first one [this will make sure that the refined machine file gets preference]
            DataCombined = DataCombined[~DataCombined.index.duplicated(keep="first")]
            """
            if len(self.droppedframes)>0: #i.e. frames were dropped/corrupt. also remove them from original file (if they exist!)
                for fn in self.droppedframes:
                    try:
                        DataCombined.drop(fn,inplace=True)
                    except KeyError:
                        pass
            """
            DataCombined.sort_index(inplace=True)
            DataCombined.to_hdf(
                os.path.join(self.dir, "CollectedData_" + self.humanscorer + ".h5"),
                key="df_with_missing",
                mode="w",
            )
            DataCombined.to_csv(
                os.path.join(self.dir, "CollectedData_" + self.humanscorer + ".csv")
            )
        else:
            self.Dataframe.sort_index(inplace=True)
            self.Dataframe.to_hdf(
                os.path.join(self.dir, "CollectedData_" + self.humanscorer + ".h5"),
                key="df_with_missing",
                mode="w",
            )
            self.Dataframe.to_csv(
                os.path.join(self.dir, "CollectedData_" + self.humanscorer + ".csv")
            )
            self.next.Enable(False)
            self.prev.Enable(False)
            self.slider.Enable(False)
            self.checkBox.Enable(False)

        nextFilemsg = wx.MessageBox(
            "File saved. Do you want to refine another file?",
            "Repeat?",
            wx.YES_NO | wx.ICON_INFORMATION,
        )
        if nextFilemsg == 2:
            self.file = 1
            self.axes.clear()
            self.figure.delaxes(self.figure.axes[1])
            self.choiceBox.Clear(True)
            MainFrame.updateZoomPan(self)
            self.load.Enable(True)
            MainFrame.browseDir(self, event)

    # ###########################################################################
    # Other functions
    # ###########################################################################
    def saveEachImage(self):
        """
        Updates the dataframe for the current image with the new datapoints
        """

        for bpindex, bp in enumerate(self.updatedCoords):
            self.Dataframe.loc[self.Dataframe.index[self.iter]][
                self.scorer, bp[0][-1], bp[0][-3], "x"
            ] = self.updatedCoords[bpindex][-1][0]
            self.Dataframe.loc[self.Dataframe.index[self.iter]][
                self.scorer, bp[0][-1], bp[0][-3], "y"
            ] = self.updatedCoords[bpindex][-1][1]

    def getLabels(self, img_index, ind, bodyparts):
        """
        Returns a list of x and y labels of the corresponding image index
        """
        self.previous_image_points = []
        for bpindex, bp in enumerate(bodyparts):
            image_points = [
                [
                    self.Dataframe[self.scorer][ind][bp]["x"].values[self.iter],
                    self.Dataframe[self.scorer][ind][bp]["y"].values[self.iter],
                    bp,
                    bpindex,
                    ind,
                ]
            ]
            self.previous_image_points.append(image_points)
        return self.previous_image_points

    def plot(self, im):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """
        # small hack in case there are any 0 intensity images!
        img = io.imread(im)
        maxIntensity = np.max(img)
        if maxIntensity == 0:
            maxIntensity = np.max(img) + 255

        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.drs = []

        if (
            self.visualization_rdb.GetSelection() == 0
        ):  # i.e. for color scheme for individuals
            self.Colorscheme = visualization.get_cmap(
                len(self.individual_names), self.cfg["colormap"]
            )
            self.norm, self.colorIndex = self.image_panel.getColorIndices(
                im, self.individual_names
            )
            cbar = self.figure.colorbar(
                self.ax, cax=cax, spacing="proportional", ticks=self.colorIndex
            )
            cbar.set_ticklabels(self.individual_names)
        else:  # i.e. for color scheme for all bodyparts
            self.Colorscheme = visualization.get_cmap(
                len(self.all_bodyparts), self.cfg["colormap"]
            )
            self.norm, self.colorIndex = self.image_panel.getColorIndices(
                im, self.all_bodyparts
            )
            cbar = self.figure.colorbar(
                self.ax, cax=cax, spacing="proportional", ticks=self.colorIndex
            )
            cbar.set_ticklabels(self.all_bodyparts)

        for ci, ind in enumerate(self.individual_names):
            col_idx = (
                0  # variable for iterating through the colorscheme for all bodyparts
            )
            image_points = []
            if ind == "single":
                if self.visualization_rdb.GetSelection() == 0:
                    for c, bp in enumerate(self.uniquebodyparts):
                        self.points = [
                            self.Dataframe[self.scorer][ind][bp]["x"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["y"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["likelihood"].values[
                                self.iter
                            ],
                        ]
                        self.likelihood = self.points[2]

                        # fix move to corner
                        if self.move2corner:
                            ny, nx = np.shape(img)[0], np.shape(img)[1]
                            if self.points[0] > nx or self.points[0] < 0:
                                print("fixing x for ", bp)
                                self.points[0] = self.center[0]
                            if self.points[1] > ny or self.points[1] < 0:
                                print("fixing y for ", bp)
                                self.points[1] = self.center[1]

                        if self.likelihood < self.threshold:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    facecolor="None",
                                    edgecolor=self.Colorscheme(ci),
                                    alpha=self.alpha,
                                )
                            ]
                        else:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(ci),
                                    alpha=self.alpha,
                                )
                            ]
                        self.axes.add_patch(self.circle[0])
                        self.dr = auxfun_drag.DraggablePoint(
                            self.circle[0],
                            bp,
                            individual_names=ind,
                            likelihood=self.likelihood,
                        )
                        self.dr.connect()
                        self.dr.coords = MainFrame.getLabels(
                            self, self.iter, ind, self.uniquebodyparts
                        )[c]
                        self.drs.append(self.dr)
                        self.updatedCoords.append(self.dr.coords)
                else:
                    for c, bp in enumerate(self.uniquebodyparts):
                        self.points = [
                            self.Dataframe[self.scorer][ind][bp]["x"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["y"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["likelihood"].values[
                                self.iter
                            ],
                        ]
                        self.likelihood = self.points[2]

                        # fix move to corner
                        if self.move2corner:
                            ny, nx = np.shape(img)[0], np.shape(img)[1]
                            if self.points[0] > nx or self.points[0] < 0:
                                print("fixing x for ", bp)
                                self.points[0] = self.center[0]
                            if self.points[1] > ny or self.points[1] < 0:
                                print("fixing y for ", bp)
                                self.points[1] = self.center[1]

                        if self.likelihood < self.threshold:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc="None",
                                    edgecolor=self.Colorscheme(col_idx),
                                    alpha=self.alpha,
                                )
                            ]
                        else:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(col_idx),
                                    alpha=self.alpha,
                                )
                            ]
                        self.axes.add_patch(self.circle[0])
                        col_idx = col_idx + 1
                        self.dr = auxfun_drag.DraggablePoint(
                            self.circle[0],
                            bp,
                            individual_names=ind,
                            likelihood=self.likelihood,
                        )
                        self.dr.connect()
                        self.dr.coords = MainFrame.getLabels(
                            self, self.iter, ind, self.uniquebodyparts
                        )[c]
                        self.drs.append(self.dr)
                        self.updatedCoords.append(self.dr.coords)
            else:
                if self.visualization_rdb.GetSelection() == 0:
                    for c, bp in enumerate(self.multianimalbodyparts):
                        self.points = [
                            self.Dataframe[self.scorer][ind][bp]["x"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["y"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["likelihood"].values[
                                self.iter
                            ],
                        ]
                        self.likelihood = self.points[2]

                        # fix move to corner
                        if self.move2corner:
                            ny, nx = np.shape(img)[0], np.shape(img)[1]
                            if self.points[0] > nx or self.points[0] < 0:
                                print("fixing x for ", bp)
                                self.points[0] = self.center[0]
                            if self.points[1] > ny or self.points[1] < 0:
                                print("fixing y for ", bp)
                                self.points[1] = self.center[1]

                        if self.likelihood < self.threshold:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc="None",
                                    edgecolor=self.Colorscheme(ci),
                                    alpha=self.alpha,
                                )
                            ]
                        else:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(ci),
                                    alpha=self.alpha,
                                )
                            ]
                        self.axes.add_patch(self.circle[0])
                        self.dr = auxfun_drag.DraggablePoint(
                            self.circle[0],
                            bp,
                            individual_names=ind,
                            likelihood=self.likelihood,
                        )
                        self.dr.connect()
                        self.dr.coords = MainFrame.getLabels(
                            self, self.iter, ind, self.multianimalbodyparts
                        )[c]
                        self.drs.append(self.dr)
                        self.updatedCoords.append(self.dr.coords)
                else:
                    for c, bp in enumerate(self.multianimalbodyparts):
                        self.points = [
                            self.Dataframe[self.scorer][ind][bp]["x"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["y"].values[self.iter],
                            self.Dataframe[self.scorer][ind][bp]["likelihood"].values[
                                self.iter
                            ],
                        ]
                        self.likelihood = self.points[2]

                        # fix move to corner
                        if self.move2corner:
                            ny, nx = np.shape(img)[0], np.shape(img)[1]
                            if self.points[0] > nx or self.points[0] < 0:
                                print("fixing x for ", bp)
                                self.points[0] = self.center[0]
                            if self.points[1] > ny or self.points[1] < 0:
                                print("fixing y for ", bp)
                                self.points[1] = self.center[1]

                        if self.likelihood < self.threshold:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc="None",
                                    edgecolor=self.Colorscheme(col_idx),
                                    alpha=self.alpha,
                                )
                            ]
                        else:
                            self.circle = [
                                patches.Circle(
                                    (self.points[0], self.points[1]),
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(col_idx),
                                    alpha=self.alpha,
                                )
                            ]
                        self.axes.add_patch(self.circle[0])
                        col_idx = col_idx + 1
                        self.dr = auxfun_drag.DraggablePoint(
                            self.circle[0],
                            bp,
                            individual_names=ind,
                            likelihood=self.likelihood,
                        )
                        self.dr.connect()
                        self.dr.coords = MainFrame.getLabels(
                            self, self.iter, ind, self.multianimalbodyparts
                        )[c]
                        self.drs.append(self.dr)
                        self.updatedCoords.append(self.dr.coords)
        self.figure.canvas.draw()


def show(config):
    app = wx.App()
    frame = MainFrame(None, config).Show()
    app.MainLoop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    cli_args = parser.parse_args()
