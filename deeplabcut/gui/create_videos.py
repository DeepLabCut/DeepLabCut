"""
DeepLabCut2.2+ Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import os
import pydoc
import sys

import wx

import deeplabcut

from deeplabcut.gui import LOGO_PATH
from deeplabcut.utils import auxiliaryfunctions, skeleton, auxfun_multianimal
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS


class Create_Labeled_Videos(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        # variable initialization
        self.filelist = []
        self.config = cfg
        self.bodyparts = []
        self.draw = False
        self.slow = False
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Create Labeled Videos")
        self.sizer.Add(text, pos=(0, 0), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(LOGO_PATH))
        self.sizer.Add(
            icon, pos=(0, 4), flag=wx.TOP | wx.RIGHT | wx.ALIGN_RIGHT, border=5
        )

        line1 = wx.StaticLine(self)
        self.sizer.Add(
            line1, pos=(1, 0), span=(1, 5), flag=wx.EXPAND | wx.BOTTOM, border=10
        )

        self.cfg_text = wx.StaticText(self, label="Select the config file")
        self.sizer.Add(self.cfg_text, pos=(2, 0), flag=wx.TOP | wx.LEFT, border=5)

        if sys.platform == "darwin":
            self.sel_config = wx.FilePickerCtrl(
                self,
                path="",
                style=wx.FLP_USE_TEXTCTRL,
                message="Choose the config.yaml file",
                wildcard="*.yaml",
            )
        else:
            self.sel_config = wx.FilePickerCtrl(
                self,
                path="",
                style=wx.FLP_USE_TEXTCTRL,
                message="Choose the config.yaml file",
                wildcard="config.yaml",
            )

        self.sizer.Add(
            self.sel_config, pos=(2, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5
        )
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        self.vids = wx.StaticText(self, label="Choose the videos")
        self.sizer.Add(self.vids, pos=(3, 0), flag=wx.TOP | wx.LEFT, border=10)

        self.sel_vids = wx.Button(self, label="Select videos")
        self.sizer.Add(self.sel_vids, pos=(3, 1), flag=wx.TOP | wx.EXPAND, border=10)
        self.sel_vids.Bind(wx.EVT_BUTTON, self.select_videos)

        sb = wx.StaticBox(self, label="Additional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)

        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)

        self.videotype = wx.ComboBox(self, choices=("",) + SUPPORTED_VIDEOS, style=wx.CB_READONLY)
        self.videotype.SetValue("")
        videotype_text_boxsizer.Add(
            self.videotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1
        )

        shuffle_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffle_boxsizer = wx.StaticBoxSizer(shuffle_text, wx.VERTICAL)
        self.shuffle = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffle_boxsizer.Add(self.shuffle, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)


        hbox1.Add(videotype_text_boxsizer, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox1.Add(shuffle_boxsizer, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        boxsizer.Add(hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.sizer.Add(
            boxsizer,
            pos=(4, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=5,
        )

        self.cfg = auxiliaryfunctions.read_config(self.config)
        if self.cfg.get("multianimalproject", False):
            self.plot_idv = wx.RadioBox(
                self,
                label="Create video with animal ID colored?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.plot_idv.SetSelection(1)
            hbox3.Add(self.plot_idv, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)

        self.draw_skeleton = wx.RadioBox(
            self,
            label="Include the skeleton in the video?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.draw_skeleton.Bind(wx.EVT_RADIOBOX, self.choose_draw_skeleton_options)
        self.draw_skeleton.SetSelection(1)

        self.filter = wx.RadioBox(
            self,
            label="Use filtered predictions?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.filter.SetSelection(1)

        self.video_slow = wx.RadioBox(
            self,
            label="Create a higher quality video? (slow)",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.video_slow.Bind(wx.EVT_RADIOBOX, self.choose_video_slow_options)
        self.video_slow.SetSelection(1)

        self.trail_points_text = wx.StaticBox(
            self, label="Specify the number of trail points"
        )
        trail_pointsboxsizer = wx.StaticBoxSizer(self.trail_points_text, wx.VERTICAL)
        self.trail_points = wx.SpinCtrl(self, value="0")
        trail_pointsboxsizer.Add(
            self.trail_points, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3
        )

        self.bodypart_choice = wx.RadioBox(
            self,
            label="Plot all bodyparts?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.bodypart_choice.Bind(wx.EVT_RADIOBOX, self.chooseOption)

        config_file = auxiliaryfunctions.read_config(self.config)
        if config_file.get("multianimalproject", False):
            bodyparts = config_file["multianimalbodyparts"]
        else:
            bodyparts = config_file["bodyparts"]
        self.bodyparts_to_compare = wx.CheckListBox(
            self, choices=bodyparts, style=0, name="Select the bodyparts"
        )
        self.bodyparts_to_compare.Bind(wx.EVT_CHECKLISTBOX, self.getbp)
        self.bodyparts_to_compare.Hide()

        hbox2.Add(self.draw_skeleton, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)
        hbox2.Add(trail_pointsboxsizer, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)
        hbox2.Add(self.video_slow, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)
        boxsizer.Add(hbox2, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        hbox3.Add(self.filter, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)
        hbox4.Add(self.bodypart_choice, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)
        hbox4.Add(self.bodyparts_to_compare, 3, wx.EXPAND | wx.TOP | wx.BOTTOM, 3)

        if self.cfg.get("multianimalproject", False):
            self.trajectory = wx.RadioBox(
                self,
                label="Want to plot the trajectories?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )

            self.trackerType = wx.RadioBox(
                self,
                label="Which tracker to use?",
                choices=["Default", "Transformer"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )

            hbox4.Add(self.trajectory, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
            hbox4.Add(self.trackerType, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        boxsizer.Add(hbox3, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(hbox4, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.build = wx.Button(self, label="Build skeleton")
        self.sizer.Add(self.build, pos=(5, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.build.Bind(wx.EVT_BUTTON, self.build_skeleton)
        self.build.Enable(True)

        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(5, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="RUN")
        self.sizer.Add(self.ok, pos=(5, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.create_videos)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(
            self.reset, pos=(5, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.reset.Bind(wx.EVT_BUTTON, self.reset_create_videos)

        self.sizer.AddGrowableCol(3)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def build_skeleton(self, event):
        skeleton.SkeletonBuilder(self.config)

    def select_config(self, event):
        """
        """
        self.config = self.sel_config.GetPath()

    def select_videos(self, event):
        """
        Selects the videos from the directory
        """
        cwd = os.getcwd()
        dlg = wx.FileDialog(self, "Select videos", cwd, "", "*.*", wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.vids = dlg.GetPaths()
            self.filelist = self.filelist + self.vids
            self.sel_vids.SetLabel("Total %s Videos selected" % len(self.filelist))

    def choose_draw_skeleton_options(self, event):
        if self.draw_skeleton.GetStringSelection() == "Yes":
            self.draw = True
        else:
            self.draw = False

    def plot_idv_options(self, event):
        if self.plot_idv.GetStringSelection() == "Yes":
            self.plot_idv = "individual"
        else:
            self.plot_idv = "bodypart"

    def choose_video_slow_options(self, event):
        if self.video_slow.GetStringSelection() == "Yes":
            self.slow = True
        else:
            self.slow = False

    def chooseOption(self, event):
        if self.bodypart_choice.GetStringSelection() == "No":
            self.bodyparts_to_compare.Show()
            self.getbp(event)
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)  # this sets location.
        if self.bodypart_choice.GetStringSelection() == "Yes":
            self.bodyparts_to_compare.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
            self.bodyparts = "all"

    def getbp(self, event):
        self.bodyparts = list(self.bodyparts_to_compare.GetCheckedStrings())

    def create_videos(self, event):

        shuffle = self.shuffle.GetValue()
        # self.filelist = self.filelist + self.vids

        if self.filter.GetStringSelection() == "No":
            filtered = False
        else:
            filtered = True

        if self.video_slow.GetStringSelection() == "Yes":
            self.slow = True
        else:
            self.slow = False

        if len(self.bodyparts) == 0:
            self.bodyparts = "all"

        config_file = auxiliaryfunctions.read_config(self.config)
        if config_file.get("multianimalproject", False):
            print(
                "Creating a video with the "
                + self.trackerType.GetStringSelection()
                + " tracker method!"
            )
            if self.trackerType.GetStringSelection() == "Default":
                track_method = auxfun_multianimal.get_track_method(config_file, track_method="")
            else:
                track_method = "transformer"

            if self.plot_idv.GetStringSelection() == "Yes":
                color_by = "individual"
            else:
                color_by = "bodypart"

            deeplabcut.create_labeled_video(
                self.config,
                self.filelist,
                self.videotype.GetValue(),
                shuffle=shuffle,
                save_frames=self.slow,
                draw_skeleton=self.draw,
                displayedbodyparts=self.bodyparts,
                trailpoints=self.trail_points.GetValue(),
                filtered=filtered,
                color_by=color_by,
                track_method=track_method,
            )

            if self.trajectory.GetStringSelection() == "Yes":
                deeplabcut.plot_trajectories(
                    self.config,
                    self.filelist,
                    displayedbodyparts=self.bodyparts,
                    videotype=self.videotype.GetValue(),
                    shuffle=shuffle,
                    filtered=filtered,
                    showfigures=False,
                )
        else:
            deeplabcut.create_labeled_video(
                self.config,
                self.filelist,
                self.videotype.GetValue(),
                shuffle=shuffle,
                save_frames=self.slow,
                draw_skeleton=self.draw,
                displayedbodyparts=self.bodyparts,
                trailpoints=self.trail_points.GetValue(),
                filtered=filtered,
            )

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.create_labeled_video"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        os.remove("help.txt")

    def reset_create_videos(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.videotype.SetStringSelection("")
        self.sel_vids.SetLabel("Select videos")
        self.filelist = []
        self.shuffle.SetValue(1)
        if self.draw_skeleton.IsShown():
            self.draw_skeleton.SetSelection(1)
            # self.SetSizer(self.sizer)
            # self.sizer.Fit(self)
            self.bodyparts_to_compare.Hide()
