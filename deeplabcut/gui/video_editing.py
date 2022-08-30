"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import datetime
import os
import pydoc
import sys

import wx
import wx.lib.agw.floatspin as FS

import deeplabcut

from deeplabcut.gui import LOGO_PATH


# DownSampleVideo(vname,width=-1,height=200,outsuffix='downsampled',outpath=None,rotatecw=False):


class Video_Editing(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        # variable initialization
        self.filelist = []
        self.config = cfg

        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Optional Video Editor")
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

        self.vids = wx.StaticText(self, label="Choose the video")
        self.sizer.Add(self.vids, pos=(3, 0), flag=wx.TOP | wx.LEFT, border=10)

        self.sel_vids = wx.Button(self, label="Select video")
        self.sizer.Add(self.sel_vids, pos=(3, 1), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_vids.Bind(wx.EVT_BUTTON, self.select_videos)

        sb = wx.StaticBox(self, label="Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        video_height = wx.StaticBox(
            self, label="Downsample - specify the video height (aspect ratio fixed)"
        )
        vheight_boxsizer = wx.StaticBoxSizer(video_height, wx.VERTICAL)
        self.height = wx.SpinCtrl(self, value="256", min=0, max=1000)
        vheight_boxsizer.Add(self.height, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.rotate = wx.RadioBox(
            self,
            label="Downsample: rotate video?",
            choices=["Yes", "No", "Arbitrary"],
            # majorDimension=0,
            style=wx.RA_SPECIFY_COLS,
        )
        self.rotate.SetSelection(1)

        hbox1.Add(vheight_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox1.Add(self.rotate, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        boxsizer.Add(hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        self.sizer.Add(
            boxsizer,
            pos=(4, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )
        angle = wx.StaticBox(self, label="Angle for arbitrary rotation (deg)")
        vangle_boxsizer = wx.StaticBoxSizer(angle, wx.VERTICAL)
        self.vangle = FS.FloatSpin(
            self, value="0.0", min_val=-360.0, max_val=360.0, digits=2
        )
        vangle_boxsizer.Add(self.vangle, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        video_start = wx.StaticBox(self, label="Shorten: start time (sec)")
        vstart_boxsizer = wx.StaticBoxSizer(video_start, wx.VERTICAL)
        self.vstart = wx.SpinCtrl(self, value="1", min=0, max=3600)
        vstart_boxsizer.Add(self.vstart, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        video_stop = wx.StaticBox(self, label="Shorten: stop time (sec)")
        vstop_boxsizer = wx.StaticBoxSizer(video_stop, wx.VERTICAL)
        self.vstop = wx.SpinCtrl(self, value="30", min=1, max=3600)
        vstop_boxsizer.Add(self.vstop, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        hbox2.Add(vstart_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox2.Add(vstop_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox2.Add(vangle_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(hbox2, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(5, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="DOWNSAMPLE")
        self.sizer.Add(self.ok, pos=(5, 2), flag=wx.LEFT, border=10)
        self.ok.Bind(wx.EVT_BUTTON, self.downsample_video)

        self.ok = wx.Button(self, label="SHORTEN")
        self.sizer.Add(self.ok, pos=(5, 3), flag=wx.LEFT, border=10)
        self.ok.Bind(wx.EVT_BUTTON, self.shorten_video)

        self.ok = wx.Button(self, label="CROP")
        self.sizer.Add(self.ok, pos=(5, 4), flag=wx.LEFT, border=10)
        self.ok.Bind(wx.EVT_BUTTON, self.crop_video)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(self.reset, pos=(6, 0), flag=wx.LEFT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_edit_videos)

        self.sizer.AddGrowableCol(3)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def select_config(self, event):
        """
        """
        self.config = self.sel_config.GetPath()

    def select_videos(self, event):
        """
        Selects the videos from the directory
        """
        cwd = os.getcwd()
        dlg = wx.FileDialog(
            self, "Select video to modify", cwd, "", "*.*", wx.FD_MULTIPLE
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.vids = dlg.GetPaths()
            self.filelist = self.filelist + self.vids  # [0]
            self.sel_vids.SetLabel("%s Video selected" % len(self.filelist))

    def downsample_video(self, event):
        if self.rotate.GetStringSelection() == "Yes":
            self.rotate_val = "Yes"

        elif self.rotate.GetStringSelection() == "Arbitrary":
            self.rotate_val = "Arbitrary"

        else:
            self.rotate_val = "No"

        Videos = self.filelist
        if len(Videos) > 0:
            for video in Videos:
                deeplabcut.DownSampleVideo(
                    video,
                    width=-1,
                    height=self.height.GetValue(),
                    rotatecw=self.rotate_val,
                    angle=self.vangle.GetValue(),
                )
        else:
            print("Please select a video first!")

    def shorten_video(self, event):
        def sweet_time_format(val):
            return str(datetime.timedelta(seconds=val))

        Videos = self.filelist
        if len(Videos) > 0:
            for video in Videos:
                deeplabcut.ShortenVideo(
                    video,
                    start=sweet_time_format(self.vstart.GetValue()),
                    stop=sweet_time_format(self.vstop.GetValue()),
                )

        else:
            print("Please select a video first!")

    def crop_video(self, event):
        Videos = self.filelist
        if len(Videos) > 0:
            for video in Videos:
                deeplabcut.CropVideo(video, useGUI=True)
        else:
            print("Please select a video first!")

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.DownSampleVideo"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        os.remove("help.txt")

    def reset_edit_videos(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.sel_vids.SetLabel("Select videos")
        self.filelist = []
        self.rotate.SetSelection(1)
