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

import wx

import deeplabcut
from deeplabcut.gui import LOGO_PATH
from deeplabcut.utils import auxiliaryfunctions


class Evaluate_network(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initialization
        self.config = cfg
        self.bodyparts = []
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Step 6. Evaluate Network")
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

        sb = wx.StaticBox(self, label="Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox3 = wx.BoxSizer(wx.HORIZONTAL)

        config_file = auxiliaryfunctions.read_config(self.config)

        shuffles_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffles_text_boxsizer = wx.StaticBoxSizer(shuffles_text, wx.VERTICAL)
        self.shuffles = wx.SpinCtrl(self, value="1", min=0, max=100)
        shuffles_text_boxsizer.Add(self.shuffles, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        #trainingset = wx.StaticBox(self, label="Specify the trainingset index")
        #trainingset_boxsizer = wx.StaticBoxSizer(trainingset, wx.VERTICAL)
        #self.trainingset = wx.SpinCtrl(self, value="0", min=0, max=100)
        #trainingset_boxsizer.Add(
        #    self.trainingset, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        #)

        self.plot_choice = wx.RadioBox(
            self,
            label="Want to plot predictions (as in standard DLC projects)?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.plot_choice.SetSelection(0)

        self.plot_scoremaps = wx.RadioBox(
            self,
            label="Want to plot maps (ALL images): scoremaps, PAFs, locrefs?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.plot_scoremaps.SetSelection(1)

        self.bodypart_choice = wx.RadioBox(
            self,
            label="Compare all bodyparts?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.bodypart_choice.Bind(wx.EVT_RADIOBOX, self.chooseOption)

        if config_file.get("multianimalproject", False):
            bodyparts = config_file["multianimalbodyparts"]
        else:
            bodyparts = config_file["bodyparts"]
        self.bodyparts_to_compare = wx.CheckListBox(
            self, choices=bodyparts, style=0, name="Select the bodyparts"
        )
        self.bodyparts_to_compare.Bind(wx.EVT_CHECKLISTBOX, self.getbp)
        self.bodyparts_to_compare.Hide()

        self.hbox1.Add(shuffles_text_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        #self.hbox1.Add(trainingset_boxsizer, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.hbox1.Add(self.plot_scoremaps, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.hbox2.Add(self.plot_choice, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.hbox2.Add(self.bodypart_choice, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.hbox2.Add(self.bodyparts_to_compare, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        boxsizer.Add(self.hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(self.hbox2, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.sizer.Add(
            boxsizer,
            pos=(3, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(4, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="RUN: Evaluate Network")
        self.sizer.Add(self.ok, pos=(4, 3))
        self.ok.Bind(wx.EVT_BUTTON, self.evaluate_network)

        self.ok = wx.Button(self, label="Optional: Plot 3 test maps")
        self.sizer.Add(self.ok, pos=(5, 3))
        self.ok.Bind(wx.EVT_BUTTON, self.plot_maps)

        self.cancel = wx.Button(self, label="Reset")
        self.sizer.Add(
            self.cancel, pos=(4, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.cancel.Bind(wx.EVT_BUTTON, self.cancel_evaluate_network)

        if config_file.get("multianimalproject", False):
            self.inf_cfg_text = wx.Button(self, label="Edit the inference_config.yaml")
            self.inf_cfg_text.Bind(wx.EVT_BUTTON, self.edit_inf_config)
            self.sizer.Add(
                self.inf_cfg_text,
                pos=(4, 2),
                span=(1, 1),
                flag=wx.BOTTOM | wx.RIGHT,
                border=10,
            )

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.evaluate_network"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")

    def chooseOption(self, event):
        if self.bodypart_choice.GetStringSelection() == "No":
            self.bodyparts_to_compare.Show()
            self.getbp(event)
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
        if self.bodypart_choice.GetStringSelection() == "Yes":
            self.bodyparts_to_compare.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
            self.bodyparts = "all"

    def getbp(self, event):
        self.bodyparts = list(self.bodyparts_to_compare.GetCheckedStrings())

    def select_config(self, event):
        self.config = self.sel_config.GetPath()

    def plot_maps(self, event):
        shuffle = self.shuffles.GetValue()
        # if self.plot_scoremaps.GetStringSelection() == "Yes":
        deeplabcut.extract_save_all_maps(
            self.config, shuffle=shuffle, Indices=[0, 1, 5]
        )

    def edit_inf_config(self, event):
        # Read the infer config file
        cfg = auxiliaryfunctions.read_config(self.config)
        #trainingsetindex = self.trainingset.GetValue()
        trainFraction = cfg["TrainingFraction"]
        self.inf_cfg_path = os.path.join(
            cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                trainFraction[-1], self.shuffles.GetValue(), cfg
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

    def evaluate_network(self, event):

        # shuffle = self.shuffle.GetValue()
        #trainingsetindex = self.trainingset.GetValue()

        Shuffles = [self.shuffles.GetValue()]
        if self.plot_choice.GetStringSelection() == "Yes":
            plotting = True
        else:
            plotting = False

        if self.plot_scoremaps.GetStringSelection() == "Yes":
            for shuffle in Shuffles:
                deeplabcut.extract_save_all_maps(self.config, shuffle=shuffle)

        if len(self.bodyparts) == 0:
            self.bodyparts = "all"
        deeplabcut.evaluate_network(
            self.config,
            Shuffles=Shuffles,
            #trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=True,
            comparisonbodyparts=self.bodyparts,
        )

    def cancel_evaluate_network(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.plot_choice.SetSelection(1)
        self.bodypart_choice.SetSelection(0)
        self.shuffles.SetValue(1)
        if self.bodyparts_to_compare.IsShown():
            self.bodyparts_to_compare.Hide()
