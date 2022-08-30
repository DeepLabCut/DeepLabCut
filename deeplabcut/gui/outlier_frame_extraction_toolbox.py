"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import wx
import wx.lib.scrolledpanel as SP
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import io
from skimage.util import img_as_ubyte

from deeplabcut.create_project import add
from deeplabcut.gui.widgets import BasePanel, WidgetPanel, BaseFrame
from deeplabcut.utils import (
    auxiliaryfunctions,
    visualization,
    conversioncode,
)
from deeplabcut.utils.auxfun_videos import VideoWriter


# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(BasePanel):
    def getColorIndices(self, img, bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        norm = mcolors.Normalize(vmin=np.min(img), vmax=np.max(img))
        ticks = np.linspace(np.min(img), np.max(img), len(bodyparts))[::-1]
        return norm, ticks


class ScrollPanel(SP.ScrolledPanel):
    def __init__(self, parent):
        SP.ScrolledPanel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
        self.Layout()

    def on_focus(self, event):
        pass

    def addRadioButtons(self):
        """
        Adds radio buttons for each bodypart on the right panel
        """
        self.choiceBox = wx.BoxSizer(wx.VERTICAL)
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
        return (self.choiceBox, self.visualization_radiobox)


class MainFrame(BaseFrame):
    """Contains the main GUI and button boxes"""

    def __init__(
        self, parent, config, video, shuffle, Dataframe, savelabeled, multianimal
    ):
        super(MainFrame, self).__init__(
            "DeepLabCut2.0 - Manual Outlier Frame Extraction", parent
        )

        ###################################################################################################################################################
        # Splitting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!
        # topSplitter = wx.SplitterWindow(self)
        #
        # self.image_panel = ImagePanel(topSplitter, config,video,shuffle,Dataframe,self.gui_size)
        # self.widget_panel = WidgetPanel(topSplitter)
        #
        # topSplitter.SplitHorizontally(self.image_panel, self.widget_panel,sashPosition=self.gui_size[1]*0.83)#0.9
        # topSplitter.SetSashGravity(1)
        # sizer = wx.BoxSizer(wx.VERTICAL)
        # sizer.Add(topSplitter, 1, wx.EXPAND)
        # self.SetSizer(sizer)

        # Splitting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!

        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)

        self.image_panel = ImagePanel(vSplitter, config, self.gui_size)
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

        self.load_button_sizer = wx.BoxSizer(wx.VERTICAL)
        self.help_button_sizer = wx.BoxSizer(wx.VERTICAL)

        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        self.help_button_sizer.Add(self.help, 1, wx.ALL, 15)
        #        widgetsizer.Add(self.help , 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)

        widgetsizer.Add(self.help_button_sizer, 1, wx.ALL, 0)

        self.grab = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Grab Frames")
        widgetsizer.Add(self.grab, 1, wx.ALL, 15)
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.grab.Enable(True)

        widgetsizer.AddStretchSpacer(5)
        self.slider = wx.Slider(
            self.widget_panel,
            id=wx.ID_ANY,
            value=0,
            minValue=0,
            maxValue=1,
            size=(200, -1),
            style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        widgetsizer.Add(self.slider, 1, wx.ALL, 5)
        self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)

        widgetsizer.AddStretchSpacer(5)
        self.start_frames_sizer = wx.BoxSizer(wx.VERTICAL)
        self.end_frames_sizer = wx.BoxSizer(wx.VERTICAL)

        self.start_frames_sizer.AddSpacer(15)
        #        self.startFrame = wx.SpinCtrl(self.widget_panel, value='0', size=(100, -1), min=0, max=120)
        self.startFrame = wx.SpinCtrl(
            self.widget_panel, value="0", size=(100, -1)
        )  # ,style=wx.SP_VERTICAL)
        self.startFrame.Enable(False)
        self.start_frames_sizer.Add(self.startFrame, 1, wx.EXPAND | wx.ALIGN_LEFT, 15)
        start_text = wx.StaticText(self.widget_panel, label="Start Frame Index")
        self.start_frames_sizer.Add(start_text, 1, wx.EXPAND | wx.ALIGN_LEFT, 15)
        self.checkBox = wx.CheckBox(
            self.widget_panel, id=wx.ID_ANY, label="Range of frames"
        )
        self.checkBox.Bind(wx.EVT_CHECKBOX, self.activate_frame_range)
        self.start_frames_sizer.Add(self.checkBox, 1, wx.EXPAND | wx.ALIGN_LEFT, 15)
        #
        self.end_frames_sizer.AddSpacer(15)
        self.endFrame = wx.SpinCtrl(
            self.widget_panel, value="1", size=(160, -1)
        )  # , min=1, max=120)
        self.endFrame.Enable(False)
        self.end_frames_sizer.Add(self.endFrame, 1, wx.EXPAND | wx.ALIGN_LEFT, 15)
        end_text = wx.StaticText(self.widget_panel, label="Number of Frames")
        self.end_frames_sizer.Add(end_text, 1, wx.EXPAND | wx.ALIGN_LEFT, 15)
        self.updateFrame = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Update")
        self.end_frames_sizer.Add(self.updateFrame, 1, wx.EXPAND | wx.ALIGN_LEFT, 15)
        self.updateFrame.Bind(wx.EVT_BUTTON, self.updateSlider)
        self.updateFrame.Enable(False)

        widgetsizer.Add(self.start_frames_sizer, 1, wx.ALL, 0)
        widgetsizer.AddStretchSpacer(5)
        widgetsizer.Add(self.end_frames_sizer, 1, wx.ALL, 0)
        widgetsizer.AddStretchSpacer(15)

        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit, 1, wx.ALL, 15)
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
        self.firstFrame = 0
        self.Colorscheme = []

        # Read config file
        self.cfg = auxiliaryfunctions.read_config(config)
        self.Task = self.cfg["Task"]
        self.start = self.cfg["start"]
        self.stop = self.cfg["stop"]
        self.date = self.cfg["date"]
        self.trainFraction = self.cfg["TrainingFraction"]
        self.trainFraction = self.trainFraction[0]
        self.videos = self.cfg["video_sets"].keys()
        self.bodyparts = self.cfg["bodyparts"]
        self.colormap = plt.get_cmap(self.cfg["colormap"])
        self.colormap = self.colormap.reversed()
        self.markerSize = self.cfg["dotsize"]
        self.alpha = self.cfg["alphavalue"]
        self.iterationindex = self.cfg["iteration"]
        self.cropping = self.cfg["cropping"]
        self.video_names = [Path(i).stem for i in self.videos]
        self.config_path = Path(config)
        self.video_source = Path(video).resolve()
        self.shuffle = shuffle
        self.Dataframe = Dataframe
        conversioncode.guarantee_multiindex_rows(self.Dataframe)
        self.savelabeled = savelabeled
        self.multianimal = multianimal
        if self.multianimal:
            from deeplabcut.utils import auxfun_multianimal

            (
                self.individual_names,
                self.uniquebodyparts,
                self.multianimalbodyparts,
            ) = auxfun_multianimal.extractindividualsandbodyparts(self.cfg)
            self.choiceBox, self.visualization_rdb = self.choice_panel.addRadioButtons()
            self.Colorscheme = visualization.get_cmap(
                len(self.individual_names), self.cfg["colormap"]
            )
            self.visualization_rdb.Bind(wx.EVT_RADIOBOX, self.clear_plot)
        # Read the video file
        self.vid = VideoWriter(str(self.video_source))
        if self.cropping:
            self.vid.set_bbox(
                self.cfg["x1"], self.cfg["x2"], self.cfg["y1"], self.cfg["y2"]
            )
        self.filename = Path(self.video_source).name
        self.numberFrames = len(self.vid)
        self.strwidth = int(np.ceil(np.log10(self.numberFrames)))
        # Set the values of slider and range of frames
        self.startFrame.SetMax(self.numberFrames - 1)
        self.slider.SetMax(self.numberFrames - 1)
        self.endFrame.SetMax(self.numberFrames - 1)
        self.startFrame.Bind(wx.EVT_SPINCTRL, self.updateSlider)  # wx.EVT_SPIN
        # Set the status bar
        self.statusbar.SetStatusText("Working on video: {}".format(self.filename))
        # Adding the video file to the config file.
        if self.vid.name not in self.video_names:
            add.add_new_videos(self.config_path, [self.video_source])

        self.update()
        self.plot_labels()
        self.widget_panel.Layout()

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
            print("Quitting for now!")
            self.Destroy()

    def updateSlider(self, event):
        self.slider.SetValue(self.startFrame.GetValue())
        self.startFrame.SetValue(self.slider.GetValue())
        self.axes.clear()
        self.figure.delaxes(self.figure.axes[1])
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.currFrame = self.slider.GetValue()
        self.update()
        self.plot_labels()

    def activate_frame_range(self, event):
        """
        Activates the frame range boxes
        """
        self.checkSlider = event.GetEventObject()
        if self.checkSlider.GetValue():
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

    def line_select_callback(self, eclick, erelease):
        "eclick and erelease are the press and release events"
        self.new_x1, self.new_y1 = eclick.xdata, eclick.ydata
        self.new_x2, self.new_y2 = erelease.xdata, erelease.ydata

    def OnSliderScroll(self, event):
        """
        Slider to scroll through the video
        """
        self.axes.clear()
        self.figure.delaxes(self.figure.axes[1])
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.currFrame = self.slider.GetValue()
        self.startFrame.SetValue(self.currFrame)
        self.update()
        self.plot_labels()

    def update(self):
        """
        Updates the image with the current slider index
        """
        self.grab.Enable(True)
        self.grab.Bind(wx.EVT_BUTTON, self.grabFrame)
        self.figure, self.axes, self.canvas = self.image_panel.getfigure()
        self.vid.set_to_frame(self.currFrame)
        frame = self.vid.read_frame(crop=self.cropping)
        if frame is not None:
            frame = img_as_ubyte(frame)
            self.ax = self.axes.imshow(frame, cmap=self.colormap)
            self.axes.set_title(
                str(
                    str(self.currFrame)
                    + "/"
                    + str(self.numberFrames - 1)
                    + " "
                    + self.filename
                )
            )
            self.figure.canvas.draw()
        else:
            print("Invalid frame")

    def chooseFrame(self):
        frame = img_as_ubyte(self.vid.read_frame(crop=self.cropping))
        fname = Path(self.filename)
        output_path = self.config_path.parents[0] / "labeled-data" / fname.stem
        img_name = (
            str(output_path)
            + "/img"
            + str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames))))
            + ".png"
        )
        if os.path.exists(img_name):
            warnings.warn("The selected frame has already been extracted; please select another one.")
            return

        self.machinefile = os.path.join(
            str(output_path), "machinelabels-iter" + str(self.iterationindex) + ".h5"
        )
        name = str(fname.stem)
        DF = self.Dataframe.iloc[[self.currFrame]]
        DF.index = pd.MultiIndex.from_tuples([
            ("labeled-data", name, "img" + str(index).zfill(self.strwidth) + ".png")
            for index in DF.index
        ])
        labeled_img_name = (
            str(output_path)
            + "/img"
            + str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames))))
            + "labeled.png"
        )

        # Check for it output path and a machine label file exist
        if output_path.exists() and Path(self.machinefile).is_file():
            io.imsave(img_name, frame)
            if self.savelabeled:
                self.figure.savefig(labeled_img_name, bbox_inches="tight")
            Data = pd.read_hdf(self.machinefile)
            conversioncode.guarantee_multiindex_rows(Data)
            DataCombined = pd.concat([Data, DF])
            DataCombined = DataCombined[~DataCombined.index.duplicated(keep="first")]
            DataCombined.to_hdf(self.machinefile, key="df_with_missing", mode="w")
            DataCombined.to_csv(os.path.join(str(output_path), "machinelabels.csv"))
        # If machine label file does not exist then create one
        elif output_path.exists() and not (Path(self.machinefile).is_file()):
            if self.savelabeled:
                self.figure.savefig(labeled_img_name, bbox_inches="tight")
            io.imsave(img_name, frame)
            DF.to_hdf(self.machinefile, key="df_with_missing", mode="w")
            DF.to_csv(os.path.join(str(output_path), "machinelabels.csv"))
        else:
            print(
                "%s path not found. Please make sure that the video was added to the config file using the function 'deeplabcut.add_new_videos'.Quitting for now!"
                % output_path
            )
            self.Destroy()

    def grabFrame(self, event):
        """
        Extracts the frame and saves in the current directory
        """

        if self.extract_range_frame:
            num_frames_extract = self.endFrame.GetValue()
            for i in range(self.currFrame, self.currFrame + num_frames_extract):
                self.currFrame = i
                self.vid.set_to_frame(self.currFrame)
                self.chooseFrame()
        else:
            self.vid.set_to_frame(self.currFrame)
            self.chooseFrame()

    def clear_plot(self, event):
        self.figure.delaxes(self.figure.axes[1])
        [p.remove() for p in reversed(self.axes.patches)]
        self.plot_labels()

    def plot_labels(self):
        """
        Plots the labels of the analyzed video
        """
        self.vid.set_to_frame(self.currFrame)
        frame = self.vid.read_frame()
        if frame is not None:
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if self.multianimal:
                # take into account of all the bodyparts for the colorscheme. Sort the bodyparts to have same order as in the config file
                self.all_bodyparts = np.array(
                    self.multianimalbodyparts + self.uniquebodyparts
                )
                _, return_idx = np.unique(self.all_bodyparts, return_index=True)
                self.all_bodyparts = list(self.all_bodyparts[np.sort(return_idx)])

                if (
                    self.visualization_rdb.GetSelection() == 0
                ):  # i.e. for color scheme for individuals
                    self.Colorscheme = visualization.get_cmap(
                        len(self.individual_names), self.cfg["colormap"]
                    )
                    self.norm, self.colorIndex = self.image_panel.getColorIndices(
                        frame, self.individual_names
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
                        frame, self.all_bodyparts
                    )
                    cbar = self.figure.colorbar(
                        self.ax, cax=cax, spacing="proportional", ticks=self.colorIndex
                    )
                    cbar.set_ticklabels(self.all_bodyparts)

                for ci, ind in enumerate(self.individual_names):
                    col_idx = 0  # variable for iterating through the colorscheme for all bodyparts
                    image_points = []
                    if ind == "single":
                        if self.visualization_rdb.GetSelection() == 0:
                            for c, bp in enumerate(self.uniquebodyparts):
                                pts = self.Dataframe.xs(
                                    (ind, bp),
                                    level=("individuals", "bodyparts"),
                                    axis=1,
                                ).values
                                self.circle = patches.Circle(
                                    pts[self.currFrame, :2],
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(ci),
                                    alpha=self.alpha,
                                )
                                self.axes.add_patch(self.circle)
                        else:
                            for c, bp in enumerate(self.uniquebodyparts):
                                pts = self.Dataframe.xs(
                                    (ind, bp),
                                    level=("individuals", "bodyparts"),
                                    axis=1,
                                ).values
                                self.circle = patches.Circle(
                                    pts[self.currFrame, :2],
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(col_idx),
                                    alpha=self.alpha,
                                )
                                self.axes.add_patch(self.circle)
                                col_idx = col_idx + 1
                    else:
                        if self.visualization_rdb.GetSelection() == 0:
                            for c, bp in enumerate(self.multianimalbodyparts):
                                pts = self.Dataframe.xs(
                                    (ind, bp),
                                    level=("individuals", "bodyparts"),
                                    axis=1,
                                ).values
                                self.circle = patches.Circle(
                                    pts[self.currFrame, :2],
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(ci),
                                    alpha=self.alpha,
                                )
                                self.axes.add_patch(self.circle)
                        else:
                            for c, bp in enumerate(self.multianimalbodyparts):
                                pts = self.Dataframe.xs(
                                    (ind, bp),
                                    level=("individuals", "bodyparts"),
                                    axis=1,
                                ).values
                                self.circle = patches.Circle(
                                    pts[self.currFrame, :2],
                                    radius=self.markerSize,
                                    fc=self.Colorscheme(col_idx),
                                    alpha=self.alpha,
                                )
                                self.axes.add_patch(self.circle)
                                col_idx = col_idx + 1
                self.figure.canvas.draw()
            else:
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    frame, self.bodyparts
                )
                cbar = self.figure.colorbar(
                    self.ax, cax=cax, spacing="proportional", ticks=self.colorIndex
                )
                cbar.set_ticklabels(self.bodyparts)
                for bpindex, bp in enumerate(self.bodyparts):
                    color = self.colormap(self.norm(self.colorIndex[bpindex]))
                    self.points = [
                        self.Dataframe.xs((bp, "x"), level=(-2, -1), axis=1).values[
                            self.currFrame
                        ],
                        self.Dataframe.xs((bp, "y"), level=(-2, -1), axis=1).values[
                            self.currFrame
                        ],
                        1.0,
                    ]
                    circle = [
                        patches.Circle(
                            (self.points[0], self.points[1]),
                            radius=self.markerSize,
                            fc=color,
                            alpha=self.alpha,
                        )
                    ]
                    self.axes.add_patch(circle[0])
                self.figure.canvas.draw()
        else:
            print("Invalid frame")

    def helpButton(self, event):
        """
        Opens Instructions
        """
        wx.MessageBox(
            "1. Use the slider to select a frame in the entire video. \n\n2. Click Grab Frames button to save the specific frame.\
        \n\n3. In the events where you need to extract a range of frames, then use the checkbox 'Range of frames' to select the starting frame index and the number of frames to extract.\
        \n Click the update button to see the frame. Click Grab Frames to select the range of frames. \n\n Click OK to continue",
            "Instructions to use!",
            wx.OK | wx.ICON_INFORMATION,
        )


def show(config, video, shuffle, Dataframe, savelabeled, multianimal):
    app = wx.App()
    frame = MainFrame(
        None, config, video, shuffle, Dataframe, savelabeled, multianimal
    ).Show()
    app.MainLoop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", "video", "shuffle", "Dataframe", "savelabeled", "multianimal"
    )
    cli_args = parser.parse_args()
