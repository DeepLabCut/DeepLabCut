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

from deeplabcut.generate_training_dataset import extract_frames
from deeplabcut.gui import LOGO_PATH


class Extract_frames(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initialization
        self.method = "automatic"
        self.config = cfg
        # design the panel
        sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Step 2. Extract Frames")
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

        sb = wx.StaticBox(self, label="Optional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)

        self.method_choice = wx.RadioBox(
            self,
            label="Choose the extraction method",
            choices=["automatic", "manual"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.method_choice.Bind(wx.EVT_RADIOBOX, self.select_extract_method)
        hbox1.Add(self.method_choice, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.crop_choice = wx.RadioBox(
            self,
            label="Want to crop the frames?",
            choices=["False", "True (read from config file)", "GUI"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        hbox1.Add(self.crop_choice, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.feedback_choice = wx.RadioBox(
            self,
            label="Need user feedback?",
            choices=["No", "Yes"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        hbox1.Add(self.feedback_choice, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.opencv_choice = wx.RadioBox(
            self,
            label="Want to use openCV?",
            choices=["No", "Yes"],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.opencv_choice.SetSelection(1)
        hbox1.Add(self.opencv_choice, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        algo_text = wx.StaticBox(self, label="Select the algorithm")
        algoboxsizer = wx.StaticBoxSizer(algo_text, wx.VERTICAL)
        self.algo_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        options = ["kmeans", "uniform"]
        self.algo_choice.Set(options)
        self.algo_choice.SetValue("kmeans")
        algoboxsizer.Add(self.algo_choice, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        cluster_step_text = wx.StaticBox(self, label="Specify the cluster step")
        cluster_stepboxsizer = wx.StaticBoxSizer(cluster_step_text, wx.VERTICAL)
        self.cluster_step = wx.SpinCtrl(self, value="1")
        cluster_stepboxsizer.Add(
            self.cluster_step, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )

        slider_width_text = wx.StaticBox(self, label="Specify the GUI slider width")
        slider_widthboxsizer = wx.StaticBoxSizer(slider_width_text, wx.VERTICAL)
        self.slider_width = wx.SpinCtrl(self, value="25")
        slider_widthboxsizer.Add(
            self.slider_width, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10
        )

        hbox3.Add(algoboxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox3.Add(cluster_stepboxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        hbox3.Add(slider_widthboxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        boxsizer.Add(hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(hbox2, 5, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(hbox3, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        sizer.Add(
            boxsizer,
            pos=(3, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

        self.help_button = wx.Button(self, label="Help")
        sizer.Add(self.help_button, pos=(4, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Ok")
        sizer.Add(self.ok, pos=(4, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.extract_frames)

        self.reset = wx.Button(self, label="Reset")
        sizer.Add(
            self.reset, pos=(4, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.reset.Bind(wx.EVT_BUTTON, self.reset_extract_frames)

        sizer.AddGrowableCol(2)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.extract_frames"
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

    def select_extract_method(self, event):
        self.method = self.method_choice.GetStringSelection()
        if self.method == "manual":
            self.crop_choice.Enable(False)
            self.feedback_choice.Enable(False)
            self.opencv_choice.Enable(False)
            self.algo_choice.Enable(False)
            self.cluster_step.Enable(False)
            self.slider_width.Enable(False)
        else:
            self.crop_choice.Enable(True)
            self.feedback_choice.Enable(True)
            self.opencv_choice.Enable(True)
            self.algo_choice.Enable(True)
            self.cluster_step.Enable(True)
            self.slider_width.Enable(True)

    def extract_frames(self, event):
        mode = self.method
        algo = self.algo_choice.GetValue()
        if self.crop_choice.GetStringSelection() == "True (read from config file)":
            crop = True
        elif self.crop_choice.GetStringSelection() == "GUI":
            crop = "GUI"
        else:
            crop = False

        if self.feedback_choice.GetStringSelection() == "Yes":
            userfeedback = True
        else:
            userfeedback = False

        if self.opencv_choice.GetStringSelection() == "Yes":
            opencv = True
        else:
            opencv = False

        slider_width = self.slider_width.GetValue()
        extract_frames(
            self.config,
            mode,
            algo,
            crop=crop,
            userfeedback=userfeedback,
            cluster_step=self.cluster_step.GetValue(),
            cluster_resizewidth=30,
            cluster_color=False,
            opencv=opencv,
            slider_width=slider_width,
        )

    def reset_extract_frames(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.method_choice.SetStringSelection("automatic")
        self.crop_choice.Enable(True)
        self.feedback_choice.Enable(True)
        self.opencv_choice.Enable(True)
        self.algo_choice.Enable(True)
        self.cluster_step.Enable(True)
        self.slider_width.Enable(True)
        self.crop_choice.SetStringSelection("False")
        self.feedback_choice.SetStringSelection("No")
        self.opencv_choice.SetStringSelection("Yes")
        self.algo_choice.SetValue("kmeans")
        self.cluster_step.SetValue(1)
        self.slider_width.SetValue(25)
