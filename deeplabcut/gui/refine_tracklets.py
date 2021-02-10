"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import os
import pydoc
import sys

import wx

import deeplabcut

media_path = os.path.join(deeplabcut.__path__[0], "gui", "media")
logo = os.path.join(media_path, "logo.png")


class Refine_tracklets(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.config = cfg
        self.datafile = ""
        self.video = ""
        self.manager = None
        self.viz = None
        # design the panel
        sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Tracklets: Extract and Refine")
        sizer.Add(text, pos=(0, 0), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo))
        sizer.Add(icon, pos=(0, 4), flag=wx.TOP | wx.RIGHT | wx.ALIGN_RIGHT, border=5)

        line1 = wx.StaticLine(self)
        sizer.Add(line1, pos=(1, 0), span=(1, 5), flag=wx.EXPAND | wx.BOTTOM, border=10)

        self.cfg_text = wx.StaticText(self, label="Select the config file")
        sizer.Add(self.cfg_text, pos=(2, 0), flag=wx.TOP | wx.LEFT, border=5)

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
        sizer.Add(
            self.sel_config, pos=(2, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5
        )
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        self.video_text = wx.StaticText(self, label="Select the video")
        sizer.Add(self.video_text, pos=(3, 0), flag=wx.TOP | wx.LEFT, border=5)
        self.sel_video = wx.FilePickerCtrl(
            self, path="", style=wx.FLP_USE_TEXTCTRL, message="Open video"
        )
        sizer.Add(
            self.sel_video, pos=(3, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5
        )
        self.sel_video.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_video)

        self.data_text = wx.StaticText(self, label="Select the tracklet data")
        sizer.Add(self.data_text, pos=(4, 0), flag=wx.TOP | wx.LEFT, border=5)
        self.sel_datafile = wx.FilePickerCtrl(
            self, path="", style=wx.FLP_USE_TEXTCTRL, message="Open tracklet data"
        )  # wildcard="Pickle files (*.pickle)|*.pickle")
        sizer.Add(
            self.sel_datafile,
            pos=(4, 1),
            span=(1, 3),
            flag=wx.TOP | wx.EXPAND,
            border=5,
        )
        self.sel_datafile.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_datafile)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        slider_swap_text = wx.StaticBox(self, label="Specify the min swap fraction")
        slider_swap_sizer = wx.StaticBoxSizer(slider_swap_text, wx.VERTICAL)
        self.slider_swap = wx.SpinCtrl(self, value="2")
        slider_swap_sizer.Add(self.slider_swap, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox.Add(slider_swap_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        slider_track_text = wx.StaticBox(
            self, label="Specify the min relative tracklet length"
        )
        slider_track_sizer = wx.StaticBoxSizer(slider_track_text, wx.VERTICAL)
        self.slider_track = wx.SpinCtrl(self, value="2")
        slider_track_sizer.Add(
            self.slider_track, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )
        hbox.Add(slider_track_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        sizer.Add(
            hbox, pos=(5, 0), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10
        )

        hbox_ = wx.BoxSizer(wx.HORIZONTAL)
        slider_gap_text = wx.StaticBox(
            self,
            label="Specify the max gap size to fill, in frames (initial pickle file only!)",
        )
        slider_gap_sizer = wx.StaticBoxSizer(slider_gap_text, wx.VERTICAL)
        self.slider_gap = wx.SpinCtrl(self, value="5")
        slider_gap_sizer.Add(self.slider_gap, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox_.Add(slider_gap_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        traillength_text = wx.StaticBox(self, label="Trail Length (visualization)")
        traillength_sizer = wx.StaticBoxSizer(traillength_text, wx.VERTICAL)
        self.length_track = wx.SpinCtrl(self, value="25")
        traillength_sizer.Add(self.length_track, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox_.Add(traillength_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        sizer.Add(
            hbox_, pos=(6, 0), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10
        )

        # NEW ROW:
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)
        videotypes = [".avi", ".mp4", ".mov"]
        self.videotype = wx.ComboBox(self, choices=videotypes, style=wx.CB_READONLY)
        self.videotype.SetValue(".avi")
        videotype_text_boxsizer.Add(
            self.videotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )

        shuffle_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffle_boxsizer = wx.StaticBoxSizer(shuffle_text, wx.VERTICAL)
        self.shuffle = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffle_boxsizer.Add(self.shuffle, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        trainingset = wx.StaticBox(self, label="Specify the trainingset index")
        trainingset_boxsizer = wx.StaticBoxSizer(trainingset, wx.VERTICAL)
        self.trainingset = wx.SpinCtrl(self, value="0", min=0, max=100)
        trainingset_boxsizer.Add(
            self.trainingset, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )

        filter_text = wx.StaticBox(self, label="filter type")
        filter_sizer = wx.StaticBoxSizer(filter_text, wx.VERTICAL)
        filtertypes = ["median"]
        self.filter_track = wx.ComboBox(self, choices=filtertypes, style=wx.CB_READONLY)
        self.filter_track.SetValue("median")
        filter_sizer.Add(self.filter_track, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        filterlength_text = wx.StaticBox(self, label="filter: window length")
        filterlength_sizer = wx.StaticBoxSizer(filterlength_text, wx.VERTICAL)
        self.filterlength_track = wx.SpinCtrl(self, value="5")
        filterlength_sizer.Add(
            self.filterlength_track, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )

        hbox2.Add(videotype_text_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(shuffle_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(trainingset_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(filter_sizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(filterlength_sizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        sizer.Add(
            hbox2, pos=(7, 0), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10
        )

        self.ok = wx.Button(self, label="Step1: Launch GUI")
        sizer.Add(self.ok, pos=(6, 3))
        self.ok.Bind(wx.EVT_BUTTON, self.refine_tracklets)

        self.help_button = wx.Button(self, label="Help")
        sizer.Add(self.help_button, pos=(8, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.reset = wx.Button(self, label="Reset")
        sizer.Add(self.reset, pos=(8, 1), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_refine_tracklets)

        self.filter = wx.Button(self, label=" Step2: Filter Tracks")
        sizer.Add(self.filter, pos=(8, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.filter.Bind(wx.EVT_BUTTON, self.filter_after_refinement)

        self.export = wx.Button(self, label="Optional: Merge refined data")
        sizer.Add(self.export, pos=(10, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.export.Bind(wx.EVT_BUTTON, self.export_data)
        self.export.Disable()

        sizer.AddGrowableCol(2)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def filter_after_refinement(self, event):  # why is video type needed?
        shuffle = self.shuffle.GetValue()
        trainingsetindex = self.trainingset.GetValue()
        tracker = (
            "skeleton" if os.path.splitext(self.datafile)[0].endswith("sk") else "box"
        )
        window_length = self.filterlength_track.GetValue()
        if window_length % 2 != 1:
            raise ValueError("Window length should be odd.")

        deeplabcut.filterpredictions(
            self.config,
            [self.video],
            videotype=self.videotype.GetValue(),
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            filtertype=self.filter_track.GetValue(),
            track_method=tracker,
            windowlength=self.filterlength_track.GetValue(),
            save_as_csv=True,
        )

    def export_data(self, event):
        self.viz.export_to_training_data()

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.refine_tracklets"
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

    def select_datafile(self, event):
        self.datafile = self.sel_datafile.GetPath()
        self.sel_datafile.SetPath(os.path.basename(self.datafile))

    def select_video(self, event):
        self.video = self.sel_video.GetPath()
        self.sel_video.SetPath(os.path.basename(self.video))

    def refine_tracklets(self, event):
        self.manager, self.viz = deeplabcut.refine_tracklets(
            self.config,
            self.datafile,
            self.video,
            min_swap_len=self.slider_swap.GetValue(),
            min_tracklet_len=self.slider_track.GetValue(),
            max_gap=self.slider_gap.GetValue(),
            trail_len=self.length_track.GetValue(),
        )
        self.export.Enable()

    def reset_refine_tracklets(self, event):
        """
        Reset to default
        """
        self.config = ""
        self.datafile = ""
        self.video = ""
        self.sel_config.SetPath("")
        self.sel_datafile.SetPath("")
        self.sel_video.SetPath("")
        self.slider_swap.SetValue(2)
        self.slider_track.SetValue(2)
        self.slider_gap.SetValue(5)
        self.length_track.SetValue(25)
        # self.save.Enable(False)
