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
from deeplabcut import utils

from deeplabcut.gui import LOGO_PATH
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS


class TransformerID(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initialization
        self.config = cfg
        self.cfg = utils.read_config(cfg)
        self.filelist = []
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - OPTIONAL: Unsupervised ID Tracking with Transformer")
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
        # self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        self.sizer.Add(
            self.sel_config, pos=(2, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5
        )
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        self.vids = wx.StaticText(self, label="Choose the videos")
        self.sizer.Add(self.vids, pos=(3, 0), flag=wx.TOP | wx.LEFT, border=10)

        self.sel_vids = wx.Button(self, label="Select videos to analyze")
        self.sizer.Add(self.sel_vids, pos=(3, 1), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_vids.Bind(wx.EVT_BUTTON, self.select_videos)

        sb = wx.StaticBox(self, label="Optional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)
        self.videotype = wx.ComboBox(self, choices=("",) + SUPPORTED_VIDEOS, style=wx.CB_READONLY)
        self.videotype.SetValue("")
        videotype_text_boxsizer.Add(self.videotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        shuffles_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffles_text_boxsizer = wx.StaticBoxSizer(shuffles_text, wx.VERTICAL)
        self.shuffles = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffles_text_boxsizer.Add(self.shuffles, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        ntracks_text = wx.StaticBox(self, label="Specify the no. of animals")
        ntracks_text_boxsizer = wx.StaticBoxSizer(ntracks_text, wx.VERTICAL)
        self.n_tracks = wx.SpinCtrl(self, value="2", min=0, max=100)
        ntracks_text_boxsizer.Add(self.n_tracks, 2, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        ntriplets_text = wx.StaticBox(self, label="Specify the no. triplets")
        ntriplets_text_boxsizer = wx.StaticBoxSizer(ntriplets_text, wx.VERTICAL)
        self.ntriplets = wx.SpinCtrl(self, value="1000", min=100, max=5000)
        ntriplets_text_boxsizer.Add(self.ntriplets, 2, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        trackertype_text = wx.StaticBox(self, label="Specify the tracking type (ellipse recommended)")
        trackertype_text_boxsizer = wx.StaticBoxSizer(trackertype_text, wx.VERTICAL)
        trackertype = ["ellipse", "box"]
        self.trackertype = wx.ComboBox(self, choices=trackertype, style=wx.CB_READONLY)
        self.trackertype.SetValue("ellipse")
        trackertype_text_boxsizer.Add(self.trackertype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)



        hbox1.Add(videotype_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox1.Add(shuffles_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(ntracks_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(ntriplets_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(trackertype_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        boxsizer.Add(hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(hbox2, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.sizer.Add(
            boxsizer,
            pos=(4, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

### train and go ###


        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(6, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Run Transformer")
        self.sizer.Add(self.ok, pos=(6, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.run_transformer_reID)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(
            self.reset, pos=(6, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.reset.Bind(wx.EVT_BUTTON, self.reset_transformer_reID)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.transformer_reID"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")

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
            self, "Select videos to analyze", cwd, "", "*.*", wx.FD_MULTIPLE
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.vids = dlg.GetPaths()
            self.filelist = self.filelist + self.vids
            self.sel_vids.SetLabel("Total %s Videos selected" % len(self.filelist))

    def run_transformer_reID(self, event):
        deeplabcut.transformer_reID(
            config=self.config,
            videos=self.filelist,
            videotype=self.videotype.GetValue(),
            n_tracks=self.n_tracks.GetValue(),
            shuffle=self.shuffles.GetValue(),
            track_method=self.trackertype.GetValue(),
        )

    def reset_transformer_reID(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.videotype.SetValue("")
        self.sel_vids.SetLabel("Select videos to analyze")
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
