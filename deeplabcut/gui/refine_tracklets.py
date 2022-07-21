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
import subprocess

import wx

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS

from deeplabcut.gui import LOGO_PATH
from pathlib import Path


class Refine_tracklets(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.config = cfg
        self.cfg = auxiliaryfunctions.read_config(self.config)
        self.datafile = ""
        self.video = ""
        self.manager = None
        self.viz = None
        # design the panel
        sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - OPTIONAL Refine Tracklets")
        sizer.Add(text, pos=(0, 0), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(LOGO_PATH))
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

        self.ntracks_text = wx.StaticText(self, label="Number of animals")
        sizer.Add(self.ntracks_text, pos=(3, 0), flag=wx.TOP | wx.LEFT, border=5)
        self.ntracks = wx.SpinCtrl(
            self, value=str(len(self.cfg["individuals"])), min=1, max=1000
        )
        sizer.Add(
            self.ntracks, pos=(3, 1), span=(1, 3), flag=wx.EXPAND | wx.TOP, border=5
        )

        self.video_text = wx.StaticText(self, label="Select the video")
        sizer.Add(self.video_text, pos=(4, 0), flag=wx.TOP | wx.LEFT, border=5)
        self.sel_video = wx.FilePickerCtrl(
            self, path="", style=wx.FLP_USE_TEXTCTRL, message="Open video"
        )
        sizer.Add(
            self.sel_video, pos=(4, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5
        )
        self.sel_video.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_video)

        vbox_ = wx.BoxSizer(wx.VERTICAL)
        hbox_ = wx.BoxSizer(wx.HORIZONTAL)
        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)
        self.videotype = wx.ComboBox(self, choices=("",) + SUPPORTED_VIDEOS, style=wx.CB_READONLY)
        self.videotype.SetValue("")
        videotype_text_boxsizer.Add(
            self.videotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )

        shuffle_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffle_boxsizer = wx.StaticBoxSizer(shuffle_text, wx.VERTICAL)
        self.shuffle = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffle_boxsizer.Add(self.shuffle, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        #trainingset = wx.StaticBox(self, label="Specify the trainingset index")
        #trainingset_boxsizer = wx.StaticBoxSizer(trainingset, wx.VERTICAL)
        #self.trainingset = wx.SpinCtrl(self, value="0", min=0, max=100)
        #trainingset_boxsizer.Add(
        #    self.trainingset, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        #)

        hbox_.Add(videotype_text_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox_.Add(shuffle_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        #hbox_.Add(trainingset_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        vbox_.Add(hbox_)
        sizer.Add(vbox_, pos=(5, 0))

        self.create_tracks_btn = wx.Button(self, label="Run(or re-run) tracking")
        sizer.Add(self.create_tracks_btn, pos=(6, 1))
        self.create_tracks_btn.Bind(wx.EVT_BUTTON, self.create_tracks)

        line2 = wx.StaticLine(self)
        sizer.Add(line2, pos=(7, 0), span=(1, 5), flag=wx.EXPAND | wx.BOTTOM, border=10)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        slider_swap_text = wx.StaticBox(
            self, label="Specify the min swap length to highlight"
        )
        slider_swap_sizer = wx.StaticBoxSizer(slider_swap_text, wx.VERTICAL)
        self.slider_swap = wx.SpinCtrl(self, value="2")
        slider_swap_sizer.Add(self.slider_swap, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox.Add(slider_swap_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        sizer.Add(
            hbox, pos=(8, 0), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10
        )

        hbox_ = wx.BoxSizer(wx.HORIZONTAL)

        slider_gap_text = wx.StaticBox(
            self, label="Specify the max gap size of missing data to fill"
        )
        slider_gap_sizer = wx.StaticBoxSizer(slider_gap_text, wx.VERTICAL)
        self.slider_gap = wx.SpinCtrl(self, value="5")
        slider_gap_sizer.Add(self.slider_gap, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox_.Add(slider_gap_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        traillength_text = wx.StaticBox(self, label="Trail Length (for visualization)")
        traillength_sizer = wx.StaticBoxSizer(traillength_text, wx.VERTICAL)
        self.length_track = wx.SpinCtrl(self, value="25")
        traillength_sizer.Add(self.length_track, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox_.Add(traillength_sizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        sizer.Add(
            hbox_, pos=(9, 0), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10
        )

        line3 = wx.StaticLine(self)
        sizer.Add(
            line3, pos=(10, 0), span=(1, 5), flag=wx.EXPAND | wx.BOTTOM, border=10
        )

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

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

        hbox2.Add(filter_sizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(filterlength_sizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        sizer.Add(
            hbox2, pos=(11, 0), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10
        )

        self.inf_cfg_text = wx.Button(self, label="Edit inference_config.yaml")
        sizer.Add(self.inf_cfg_text, pos=(12, 2), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.inf_cfg_text.Bind(wx.EVT_BUTTON, self.edit_inf_config)

        self.ok = wx.Button(self, label="Refine tracks GUI Launch")
        sizer.Add(self.ok, pos=(9, 1))
        self.ok.Bind(wx.EVT_BUTTON, self.refine_tracklets)

        self.help_button = wx.Button(self, label="Help")
        sizer.Add(self.help_button, pos=(12, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.reset = wx.Button(self, label="Reset")
        sizer.Add(self.reset, pos=(12, 1), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_refine_tracklets)

        self.filter = wx.Button(
            self, label=" Optional: Filter Tracks (then you also get a CSV file!)"
        )
        sizer.Add(self.filter, pos=(11, 1), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.filter.Bind(wx.EVT_BUTTON, self.filter_after_refinement)

        self.export = wx.Button(self, label="Optional: Merge refined data into training set")
        sizer.Add(self.export, pos=(13, 1), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.export.Bind(wx.EVT_BUTTON, self.export_data)
        self.export.Disable()

        sizer.AddGrowableCol(2)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def edit_inf_config(self, event):
        # Read the infer config file
        #trainingsetindex = self.trainingset.GetValue()
        trainFraction = self.cfg["TrainingFraction"]
        self.inf_cfg_path = os.path.join(
            self.cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                trainFraction[-1], self.shuffle.GetValue(), self.cfg
            ),
            "test",
            "inference_cfg.yaml",
        )
        # let the user open the file with default text editor. Also make it mac compatible
        if sys.platform == "darwin":
            self.file_open_bool = subprocess.call(["open", self.inf_cfg_path])
            self.file_open_bool = True
        else:
            import webbrowser

            self.file_open_bool = webbrowser.open(self.inf_cfg_path)
        if self.file_open_bool:
            self.inf_cfg = auxiliaryfunctions.read_config(self.inf_cfg_path)
        else:
            raise FileNotFoundError("File not found!")

    def filter_after_refinement(self, event):  # why is video type needed?
        shuffle = self.shuffle.GetValue()
        #trainingsetindex = self.trainingset.GetValue()
        window_length = self.filterlength_track.GetValue()
        if window_length % 2 != 1:
            raise ValueError("Window length should be odd.")

        deeplabcut.filterpredictions(
            self.config,
            [self.video],
            videotype=self.videotype.GetValue(),
            shuffle=shuffle,
            #trainingsetindex=trainingsetindex,
            filtertype=self.filter_track.GetValue(),
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

    def select_video(self, event):
        self.video = self.sel_video.GetPath()
        self.sel_video.SetPath(os.path.basename(self.video))

    def create_tracks(self, event):
        deeplabcut.stitch_tracklets(
            self.config,
            [self.video],
            videotype=self.videotype.GetValue(),
            shuffle=self.shuffle.GetValue(),
            #trainingsetindex=self.trainingset.GetValue(),
            n_tracks=self.ntracks.GetValue(),
        )

    def refine_tracklets(self, event):
        DLCscorer, _ = auxiliaryfunctions.get_scorer_name(
            self.cfg,
            self.shuffle.GetValue(),
            self.cfg["TrainingFraction"][-1],
        )
        track_method = self.cfg.get("default_track_method", "ellipse")
        if track_method == "ellipse":
            method = "el"
        elif track_method == "box":
            method = "bx"
        else:
            method = "sk"
        dest = str(Path(self.video).parents[0])
        vname = Path(self.video).stem
        datafile = os.path.join(dest, vname + DLCscorer + f"_{method}.h5")
        self.manager, self.viz = deeplabcut.refine_tracklets(
            self.config,
            datafile,
            self.video,
            min_swap_len=self.slider_swap.GetValue(),
            trail_len=self.length_track.GetValue(),
            max_gap=self.slider_gap.GetValue(),
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
        self.length_track.SetValue(25)
        self.slider_gap.SetValue(5)
        # self.save.Enable(False)
