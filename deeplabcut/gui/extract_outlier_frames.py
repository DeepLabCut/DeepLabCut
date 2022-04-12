"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
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
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS

from deeplabcut.gui import LOGO_PATH


class Extract_outlier_frames(wx.Panel):
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

        text = wx.StaticText(self, label="DeepLabCut - OPTIONAL: Extract and Refine labels on Outlier Frames")
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

        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)
        self.videotype = wx.ComboBox(self, choices=("",) + SUPPORTED_VIDEOS, style=wx.CB_READONLY)
        self.videotype.SetValue("")
        videotype_text_boxsizer.Add(
            self.videotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1
        )

        shuffles_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffles_text_boxsizer = wx.StaticBoxSizer(shuffles_text, wx.VERTICAL)
        self.shuffles = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffles_text_boxsizer.Add(self.shuffles, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        outlier_algo_text = wx.StaticBox(self, label="Specify the algorithm")
        outlier_algo_text_boxsizer = wx.StaticBoxSizer(outlier_algo_text, wx.VERTICAL)
        algotypes = ["jump", "fitting", "uncertain", "manual"]
        self.algotype = wx.ComboBox(self, choices=algotypes, style=wx.CB_READONLY)
        self.algotype.SetValue("jump")
        outlier_algo_text_boxsizer.Add(self.algotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        hbox1.Add(videotype_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox1.Add(shuffles_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox1.Add(outlier_algo_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        boxsizer.Add(hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.sizer.Add(
            boxsizer,
            pos=(4, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

### LABELING ###



        self.ok = wx.Button(self, label="LAUNCH GUI")
        self.sizer.Add(self.ok, pos=(7, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.refine_labels)

        self.merge = wx.Button(self, label="Merge dataset")
        self.sizer.Add(self.merge, pos=(7, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.merge.Bind(wx.EVT_BUTTON, self.merge_dataset)
        self.merge.Enable(False)

        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(6, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="EXTRACT FRAMES")
        self.sizer.Add(self.ok, pos=(6, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.extract_outlier_frames)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(
            self.reset, pos=(6, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.reset.Bind(wx.EVT_BUTTON, self.reset_extract_outlier_frames)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.extract_outlier_frames"
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

    def extract_outlier_frames(self, event):
        deeplabcut.extract_outlier_frames(
            config=self.config,
            videos=self.filelist,
            videotype=self.videotype.GetValue(),
            shuffle=self.shuffles.GetValue(),
            outlieralgorithm=self.algotype.GetValue(),
        )

    def reset_extract_outlier_frames(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.videotype.SetValue("")
        self.algotype.SetValue("jump")
        self.sel_vids.SetLabel("Select videos to analyze")
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def refine_labels(self, event):
        self.merge.Enable(True)
        deeplabcut.refine_labels(self.config)

    def merge_dataset(self, event):
        dlg = wx.MessageDialog(
            None,
            "1. Make sure that you have refined all the labels before merging the dataset.\n\n2. If you merge the dataset, you need to re-create the training dataset before you start the training.\n\n3. Are you ready to merge the dataset?",
            "Warning",
            wx.YES_NO | wx.ICON_WARNING,
        )
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            notebook = self.GetParent()
            notebook.SetSelection(4)
            deeplabcut.merge_datasets(self.config, forceiterate=None)

    def reset_refine_labels(self, event):
        """
        Reset to default
        """
        self.config = []
        #self.sel_config.SetPath("")
        self.merge.Enable(False)
