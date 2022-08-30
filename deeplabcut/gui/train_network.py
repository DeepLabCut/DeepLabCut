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
import subprocess
import sys
import webbrowser
from pathlib import Path

import wx

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from deeplabcut.gui import LOGO_PATH


class Train_network(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        # variable initialization
        self.method = "automatic"
        self.config = cfg
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Step 5. Train network")
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

        vbox1 = wx.BoxSizer(wx.VERTICAL)
        self.pose_cfg_text = wx.Button(self, label="Click to open the pose config file")
        self.pose_cfg_text.Bind(wx.EVT_BUTTON, self.edit_pose_config)
        vbox1.Add(self.pose_cfg_text, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.update_params_text = wx.Button(self, label="Update the parameters")
        self.update_params_text.Bind(wx.EVT_BUTTON, self.update_params)
        vbox1.Add(self.update_params_text, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.pose_cfg_text.Hide()
        self.update_params_text.Hide()

        sb = wx.StaticBox(self, label="Optional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        shuffles_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffles_text_boxsizer = wx.StaticBoxSizer(shuffles_text, wx.VERTICAL)
        self.shuffles = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffles_text_boxsizer.Add(self.shuffles, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        #trainingindex = wx.StaticBox(self, label="Specify the trainingset index")
        #trainingindex_boxsizer = wx.StaticBoxSizer(trainingindex, wx.VERTICAL)
        #self.trainingindex = wx.SpinCtrl(self, value="0", min=0, max=100)
        #trainingindex_boxsizer.Add(
        #    self.trainingindex, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.pose_cfg_choice = wx.RadioBox(
            self,
            label="Want to edit pose_cfg.yaml file?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.pose_cfg_choice.Bind(wx.EVT_RADIOBOX, self.chooseOption)
        self.pose_cfg_choice.SetSelection(1)

        # use the default pose_cfg file for default values
        default_pose_cfg_path = os.path.join(
            Path(deeplabcut.__file__).parent, "pose_cfg.yaml"
        )
        pose_cfg = auxiliaryfunctions.read_plainconfig(default_pose_cfg_path)
        display_iters = str(pose_cfg["display_iters"])
        save_iters = str(pose_cfg["save_iters"])
        max_iters = str(pose_cfg["multi_step"][-1][-1])

        display_iters_text = wx.StaticBox(self, label="Display iterations")
        display_iters_text_boxsizer = wx.StaticBoxSizer(display_iters_text, wx.VERTICAL)
        self.display_iters = wx.SpinCtrl(
            self, value=display_iters, min=1, max=int(max_iters)
        )
        display_iters_text_boxsizer.Add(
            self.display_iters, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )
        # self.display_iters.Enable(False)

        save_iters_text = wx.StaticBox(self, label="Save X number of iterations")
        save_iters_text_boxsizer = wx.StaticBoxSizer(save_iters_text, wx.VERTICAL)
        self.save_iters = wx.SpinCtrl(self, value="10000", min=1, max=int(max_iters))
        save_iters_text_boxsizer.Add(
            self.save_iters, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )
        # self.save_iters.Enable(False)

        max_iters_text = wx.StaticBox(self, label="Maximum iterations")
        max_iters_text_boxsizer = wx.StaticBoxSizer(max_iters_text, wx.VERTICAL)
        self.max_iters = wx.SpinCtrl(self, value="500000", min=1, max=int(max_iters))
        max_iters_text_boxsizer.Add(
            self.max_iters, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )
        # self.max_iters.Enable(False)

        snapshots = wx.StaticBox(self, label="Number of network snapshots to keep")
        snapshots_boxsizer = wx.StaticBoxSizer(snapshots, wx.VERTICAL)
        self.snapshots = wx.SpinCtrl(self, value="10", min=1, max=100)
        snapshots_boxsizer.Add(self.snapshots, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        # self.snapshots.Enable(False)

        hbox1.Add(shuffles_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        #hbox1.Add(trainingindex_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox1.Add(self.pose_cfg_choice, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox1.Add(vbox1, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        hbox2.Add(display_iters_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        hbox2.Add(save_iters_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(max_iters_text_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox2.Add(snapshots_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        boxsizer.Add(hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(hbox2, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.sizer.Add(
            boxsizer,
            pos=(4, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(5, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Ok")
        self.sizer.Add(self.ok, pos=(5, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.train_network)

        self.cancel = wx.Button(self, label="Reset")
        self.sizer.Add(
            self.cancel, pos=(5, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.cancel.Bind(wx.EVT_BUTTON, self.cancel_train_network)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.train_network"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")

    def chooseOption(self, event):
        if self.pose_cfg_choice.GetStringSelection() == "Yes":
            self.shuffles.Enable(False)
            #self.trainingindex.Enable(False)
            self.pose_cfg_text.Show()
            self.update_params_text.Show()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
        else:
            self.shuffles.Enable(True)
            #self.trainingindex.Enable(True)
            self.pose_cfg_text.Hide()
            self.update_params_text.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)

    def select_config(self, event):
        """
        """
        self.config = self.sel_config.GetPath()

    def edit_pose_config(self, event):
        """
        """
        self.shuffles.Enable(True)
        #self.trainingindex.Enable(True)
        self.display_iters.Enable(True)
        self.save_iters.Enable(True)
        self.max_iters.Enable(True)
        self.snapshots.Enable(True)
        # Read the pose config file

        cfg = auxiliaryfunctions.read_config(self.config)
        trainFraction = cfg["TrainingFraction"]
        #print(trainFraction[-1])
        #        print(os.path.join(cfg['project_path'],auxiliaryfunctions.get_model_folder(trainFraction, self.shuffles.GetValue(),cfg),'train','pose_cfg.yaml'))
        self.pose_cfg_path = os.path.join(
            cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                trainFraction[-1], self.shuffles.GetValue(), cfg
            ),
            "train",
            "pose_cfg.yaml",
        )
        # let the user open the file with default text editor. Also make it mac compatible
        if sys.platform == "darwin":
            self.file_open_bool = subprocess.call(["open", self.pose_cfg_path])
            self.file_open_bool = True
        else:
            self.file_open_bool = webbrowser.open(self.pose_cfg_path)
        if self.file_open_bool:
            self.pose_cfg = auxiliaryfunctions.read_plainconfig(self.pose_cfg_path)
        else:
            raise FileNotFoundError("File not found!")

    def update_params(self, event):
        # update the variables with the edited values in the pose config file
        if self.file_open_bool:
            self.pose_cfg = auxiliaryfunctions.read_plainconfig(self.pose_cfg_path)
            display_iters = str(self.pose_cfg["display_iters"])
            save_iters = str(self.pose_cfg["save_iters"])
            max_iters = str(self.pose_cfg["multi_step"][-1][-1])
            self.display_iters.SetValue(display_iters)
            self.save_iters.SetValue(save_iters)
            self.max_iters.SetValue(max_iters)
            self.shuffles.Enable(True)
            #self.trainingindex.Enable(True)
            self.display_iters.Enable(True)
            self.save_iters.Enable(True)
            self.max_iters.Enable(True)
            self.snapshots.Enable(True)
        else:
            raise FileNotFoundError("File not found!")

    def train_network(self, event):
        if self.shuffles.Children:
            shuffle = int(self.shuffles.Children[0].GetValue())
        else:
            shuffle = int(self.shuffles.GetValue())

        if self.snapshots.Children:
            max_snapshots_to_keep = int(self.snapshots.Children[0].GetValue())
        else:
            max_snapshots_to_keep = int(self.snapshots.GetValue())

        if self.display_iters.Children:
            displayiters = int(self.display_iters.Children[0].GetValue())
        else:
            displayiters = int(self.display_iters.GetValue())

        if self.save_iters.Children:
            saveiters = int(self.save_iters.Children[0].GetValue())
        else:
            saveiters = int(self.save_iters.GetValue())

        if self.max_iters.Children:
            maxiters = int(self.max_iters.Children[0].GetValue())
        else:
            maxiters = int(self.max_iters.GetValue())

        deeplabcut.train_network(
            self.config,
            shuffle,
            gputouse=None,
            max_snapshots_to_keep=max_snapshots_to_keep,
            autotune=None,
            displayiters=displayiters,
            saveiters=saveiters,
            maxiters=maxiters,
        )

    def cancel_train_network(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.pose_cfg_text.Hide()
        self.update_params_text.Hide()
        self.pose_cfg_choice.SetSelection(1)
        self.display_iters.SetValue(1000)
        self.save_iters.SetValue(10000)
        self.max_iters.SetValue(50000)
        self.snapshots.SetValue(5)
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
