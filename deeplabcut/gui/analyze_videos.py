"""
DeepLabCut2.2+ Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import os
import platform
import pydoc
import subprocess
import sys
import webbrowser

import deeplabcut
import wx
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS
from deeplabcut.gui import LOGO_PATH


class Analyze_videos(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        # variable initialization
        self.filelist = []
        self.picklelist = []
        self.bodyparts = []
        self.config = cfg
        self.cfg = auxiliaryfunctions.read_config(self.config)
        self.draw = False
        # design the panel
        self.sizer = wx.GridBagSizer(5, 10)


        text = wx.StaticText(self, label="DeepLabCut - Step 7. Analyze Videos")

        self.sizer.Add(text, pos=(0, 0), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=10)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(LOGO_PATH))
        self.sizer.Add(
            icon, pos=(0, 10), flag=wx.TOP | wx.RIGHT | wx.ALIGN_RIGHT, border=5
        )

        line1 = wx.StaticLine(self)
        self.sizer.Add(
            line1, pos=(1, 0), span=(1, 8), flag=wx.EXPAND | wx.BOTTOM, border=10
        )

        self.cfg_text = wx.StaticText(self, label="Select the config file")
        self.sizer.Add(self.cfg_text, pos=(2, 0), flag=wx.TOP | wx.LEFT, border=10)

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

        self.sel_vids = wx.Button(self, label="Select videos to analyze")
        self.sizer.Add(self.sel_vids, pos=(3, 1), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_vids.Bind(wx.EVT_BUTTON, self.select_videos)

        sb = wx.StaticBox(self, label="Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox5 = wx.BoxSizer(wx.HORIZONTAL)

#### BOX 1 ####

        shuffle_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffle_boxsizer = wx.StaticBoxSizer(shuffle_text, wx.VERTICAL)
        self.shuffle = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffle_boxsizer.Add(self.shuffle, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)

        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)

        self.videotype = wx.ComboBox(self, choices=("",) + SUPPORTED_VIDEOS, style=wx.CB_READONLY)
        self.videotype.SetValue("")
        videotype_text_boxsizer.Add(
            self.videotype, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1
        )


        boxsizer.Add(self.hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
        self.hbox1.Add(shuffle_boxsizer, 0, 0)
        self.hbox1.Add(videotype_text_boxsizer, 0, 0)

        if self.cfg.get("multianimalproject", False):
            ntracks_text = wx.StaticBox(self, label="Number of animals")
            ntracks_boxsizer = wx.StaticBoxSizer(ntracks_text, wx.VERTICAL)
            self.ntracks = wx.SpinCtrl(self, value=str(len(self.cfg["individuals"])), min=1, max=1000)
            ntracks_boxsizer.Add(self.ntracks, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)
            self.hbox1.Add(ntracks_boxsizer, 1, 1)



#### BOX 2 ####

        if self.cfg.get("multianimalproject", False):

            self.robust = wx.RadioBox(
                self,
                label="Use ffprobe to read video metadata (slow but robust)",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.robust.SetSelection(1)


            self.create_video_with_all_detections = wx.RadioBox(
                self,
                label="Create video for checking detections",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,)
            self.create_video_with_all_detections.SetSelection(0)


            boxsizer.Add(self.hbox2, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox2.Add(self.robust, 0,  wx.EXPAND | wx.TOP | wx.BOTTOM,0)
            self.hbox2.Add(self.create_video_with_all_detections,0, wx.EXPAND | wx.TOP | wx.BOTTOM,0)

            self.calibrate = wx.RadioBox(
                self,
                label="Calibrate animal assembly?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,)
            self.calibrate.SetSelection(1)


            self.identity_toggle = wx.RadioBox(
                self,
                label="Assemble using animal identity?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,)
            self.identity_toggle.SetSelection(1)

            self.hbox2.Add(self.calibrate, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox2.Add(self.identity_toggle, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)


#### BOX 3 ####

            self.csv = wx.RadioBox(
                self,
                label="Want to save result(s) as csv?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.csv.SetSelection(1)

            self.nwb = wx.RadioBox(
                self,
                label="Want to save result(s) as nwb?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.nwb.SetSelection(1)

            self.filter = wx.RadioBox(
                self,
                label="Want to filter the predictions? (+ csv file)",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.filter.SetSelection(1)

            self.trajectory = wx.RadioBox(
                self,
                label="Want to plot the trajectories?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )

            self.showfigs = wx.RadioBox(
                self,
                label="Want plots to pop up?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.trajectory.Bind(wx.EVT_RADIOBOX, self.chooseOption)
            self.trajectory.SetSelection(1)

            self.hbox3.Add(self.csv, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox3.Add(self.nwb, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox3.Add(self.filter, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox3.Add(self.showfigs, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox3.Add(self.trajectory, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)

        else:
            self.dynamic = wx.RadioBox(
                self,
                label="Want to dynamically crop bodyparts?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.dynamic.SetSelection(1)
            self.hbox3.Add(self.dynamic, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)

            self.csv = wx.RadioBox(
                self,
                label="Want to save result(s) as csv?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.csv.SetSelection(1)

            self.nwb = wx.RadioBox(
                self,
                label="Want to save result(s) as nwb?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.nwb.SetSelection(1)

            self.filter = wx.RadioBox(
                self,
                label="Want to filter the predictions?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.filter.SetSelection(1)

            self.trajectory = wx.RadioBox(
                self,
                label="Want to plot the trajectories?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )

            self.showfigs = wx.RadioBox(
                self,
                label="Want plots to pop up?",
                choices=["Yes", "No"],
                majorDimension=1,
                style=wx.RA_SPECIFY_COLS,
            )
            self.trajectory.Bind(wx.EVT_RADIOBOX, self.chooseOption)
            self.trajectory.SetSelection(0)
            # Trigger chooseOption as if the radio button had been clicked
            wx.PostEvent(
                self.trajectory,
                wx.CommandEvent(wx.wxEVT_COMMAND_RADIOBOX_SELECTED),
            )

            self.hbox1.Add(self.csv, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox1.Add(self.nwb, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox1.Add(self.filter, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox3.Add(self.showfigs, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)
            self.hbox3.Add(self.trajectory, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)






        boxsizer.Add(self.hbox3, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 0)




        config_file = auxiliaryfunctions.read_config(self.config)
        if config_file.get("multianimalproject", False):
            bodyparts = config_file["multianimalbodyparts"]
        else:
            bodyparts = config_file["bodyparts"]
        self.trajectory_to_plot = wx.CheckListBox(
            self, choices=bodyparts, style=0, name="Select the bodyparts"
        )
        self.trajectory_to_plot.Bind(wx.EVT_CHECKLISTBOX, self.getbp)
        self.trajectory_to_plot.SetCheckedItems(range(len(bodyparts)))
        self.trajectory_to_plot.Hide()

        self.draw_skeleton = wx.RadioBox(
            self,
            label="Include the skeleton in the video?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.draw_skeleton.Bind(wx.EVT_RADIOBOX, self.choose_draw_skeleton_options)
        self.draw_skeleton.SetSelection(1)
        self.draw_skeleton.Hide()

        self.trail_points_text = wx.StaticBox(
            self, label="Specify the number of trail points"
        )
        trail_pointsboxsizer = wx.StaticBoxSizer(self.trail_points_text, wx.VERTICAL)
        self.trail_points = wx.SpinCtrl(self, value="1")
        trail_pointsboxsizer.Add(
            self.trail_points, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )
        self.trail_points_text.Hide()
        self.trail_points.Hide()

        self.hbox3.Add(self.trajectory_to_plot, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 1)
        boxsizer.Add(self.hbox3, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.hbox4.Add(self.draw_skeleton, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.hbox4.Add(trail_pointsboxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        boxsizer.Add(self.hbox4, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        self.sizer.Add(
            boxsizer,
            pos=(5, 0),
            span=(1, 10),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(7, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Analyze Videos")
        self.sizer.Add(self.ok, pos=(7, 4), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.ok.Bind(wx.EVT_BUTTON, self.analyze_videos)


        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(
            self.reset, pos=(7, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.reset.Bind(wx.EVT_BUTTON, self.reset_analyze_videos)

        self.edit_config_file = wx.Button(self, label="Edit config.yaml")
        self.sizer.Add(self.edit_config_file, pos=(8, 4))
        self.edit_config_file.Bind(wx.EVT_BUTTON, self.edit_config)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def edit_config(self, event):
        """
        """
        if platform.system() == "Darwin":
            self.file_open_bool = subprocess.call(["open", self.config])
            self.file_open_bool = True
        else:
            self.file_open_bool = webbrowser.open(self.config)
        if self.file_open_bool:
            self.pose_cfg = auxiliaryfunctions.read_config(self.config)
        else:
            raise FileNotFoundError("File not found!")

    def edit_inf_config(self, event):
        # Read the infer config file
        cfg = auxiliaryfunctions.read_config(self.config)
        #trainFraction = cfg["TrainingFraction"][trainingsetindex]
        self.inf_cfg_path = os.path.join(
            cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                trainFraction, self.shuffle.GetValue(), cfg
            ),
            "test",
            "inference_cfg.yaml",
        )
        # let the user open the file with default text editor. Also make it mac compatible
        if sys.platform == "darwin":
            self.file_open_bool = subprocess.call(["open", self.inf_cfg_path])
            self.file_open_bool = True
        else:
            self.file_open_bool = webbrowser.open(self.inf_cfg_path)
        if self.file_open_bool:
            self.inf_cfg = auxiliaryfunctions.read_config(self.inf_cfg_path)
        else:
            raise FileNotFoundError("File not found!")

    def activate_change_wd(self, event):
        """
        Activates the option to change the working directory
        """
        self.change_wd = event.GetEventObject()
        if self.change_wd.GetValue():
            self.sel_wd.Enable(True)
        else:
            self.sel_wd.Enable(False)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.analyze_videos"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
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

    def choose_draw_skeleton_options(self, event):
        if self.draw_skeleton.GetStringSelection() == "Yes":
            self.draw = True
        else:
            self.draw = False

    def analyze_videos(self, event):

        shuffle = self.shuffle.GetValue()
        videotype = self.videotype.GetValue(),

        if self.cfg.get("multianimalproject", False):
            print("DLC network loading and video analysis starting ... ")
            auto_track=True
        else:
            if self.csv.GetStringSelection() == "Yes":
                save_as_csv = True
            else:
                save_as_csv = False
            if self.dynamic.GetStringSelection() == "No":
                dynamic = (False, 0.5, 10)
            else:
                dynamic = (True, 0.5, 10)
            if self.filter.GetStringSelection() == "No":
                _filter = False
            else:
                _filter = True

        if self.cfg["cropping"] == "True":
            crop = self.cfg["x1"], self.cfg["x2"], self.cfg["y1"], self.cfg["y2"]
        else:
            crop = None

        if self.cfg.get("multianimalproject", False):
            if self.robust.GetStringSelection() == "No":
                robust = False
            else:
                robust = True
            scorername = deeplabcut.analyze_videos(
                self.config,
                self.filelist,
                self.videotype.GetValue(),
                shuffle=shuffle,
                gputouse=None,
                cropping=crop,
                robust_nframes=robust,
                auto_track=True,
                n_tracks=self.ntracks.GetValue(),
                calibrate=self.calibrate.GetStringSelection() == "Yes",
                identity_only=self.identity_toggle.GetStringSelection() == "Yes",
            )

            if self.create_video_with_all_detections.GetStringSelection() == "Yes":
                deeplabcut.create_video_with_all_detections(
                    self.config,
                    self.filelist,
                    self.videotype.GetValue(),
                    shuffle=shuffle,
                )
            if self.filter.GetStringSelection() == "Yes":
                deeplabcut.filterpredictions(
                    self.config,
                    self.filelist,
                    self.videotype.GetValue(),
                    )

            if self.csv.GetStringSelection() == "Yes":
                deeplabcut.analyze_videos_converth5_to_csv(self.filelist,listofvideos=True)

            if self.nwb.GetStringSelection() == "Yes":
                deeplabcut.analyze_videos_converth5_to_nwb(
                    self.config,
                    self.filelist,
                    listofvideos=True,
                )

            if self.trajectory.GetStringSelection() == "Yes":
                if self.showfigs.GetStringSelection() == "No":
                    showfig = False
                else:
                    showfig = True
                deeplabcut.plot_trajectories(self.config,self.filelist,self.videotype.GetValue(),showfigures=showfig)
        else:
            scorername = deeplabcut.analyze_videos(
                self.config,
                self.filelist,
                self.videotype.GetValue(),
                shuffle=shuffle,
                gputouse=None,
                save_as_csv=save_as_csv,
                cropping=crop,
                dynamic=dynamic,
            )
            if _filter:
                deeplabcut.filterpredictions(
                    self.config,
                    self.filelist,
                    self.videotype.GetValue(),
                    shuffle=shuffle,
                    filtertype="median",
                    windowlength=5,
                    save_as_csv=save_as_csv,
                )

            if self.trajectory.GetStringSelection() == "Yes":
                if self.showfigs.GetStringSelection() == "No":
                    showfig = False
                else:
                    showfig = True
                deeplabcut.plot_trajectories(
                    self.config,
                    self.filelist,
                    self.videotype.GetValue(),
                    displayedbodyparts=self.bodyparts,
                    shuffle=shuffle,
                    filtered=_filter,
                    showfigures=showfig,
                )

    def reset_analyze_videos(self, event):
        """
        Reset to default
        """
        if self.cfg.get("multianimalproject", False):
            self.create_video_with_all_detections.SetSelection(1)
        else:
            self.csv.SetSelection(1)
            self.nwb.SetSelection(1)
            self.filter.SetSelection(1)
            self.trajectory.SetSelection(0)
            self.dynamic.SetSelection(1)
            # self.select_destfolder.SetPath("None")
        #self.config = []
        #self.sel_config.SetPath("")
        self.sel_vids.SetLabel("Select videos to analyze")
        self.filelist = []
        self.shuffle.SetValue(1)
        if self.draw_skeleton.IsShown():
            self.draw_skeleton.SetSelection(1)
            self.draw_skeleton.Hide()
            self.trail_points_text.Hide()
            self.trail_points.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)

    def chooseOption(self, event):
        if self.trajectory.GetStringSelection() == "Yes":
            self.trajectory_to_plot.Show()
            self.getbp(event)
        if self.trajectory.GetStringSelection() == "No":
            self.trajectory_to_plot.Hide()
            self.bodyparts = []
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def getbp(self, event):
        self.bodyparts = list(self.trajectory_to_plot.GetCheckedStrings())
