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

from deeplabcut.generate_training_dataset import check_labels
from deeplabcut.gui import LOGO_PATH
from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path


def label_frames(
    config,
    multiple_individualsGUI=False,
    imtypes=["*.png"],
    config3d=None,
    sourceCam=None,
    jump_unlabeled=False,
):
    """Manually label/annotate the extracted frames.

    Update the list of body parts you want to localize in the config.yaml file first.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    multiple_individualsGUI: bool, optional, default=False
        If ``True``, a user can label multiple individuals. Note for
        ``"multianimalproject=True"`` this is automatically used.

    imtypes: list[str], optional, default=["*.png"]
        Image types to look for in the folder. By default only png images are labeled.

    config3d: str or None, optional, default=None
        Full path of the config file in the 3D project. Include when epipolar lines
        would be helpful for labeling additional camera angles.

    sourceCam: str or None, optional, default=None
        The camera name from which to pull labeling data to generate epipolar lines.
        This must match the pattern in ``'camera_names'`` in the 3D config file.
        If no value is entered, data will be pulled from either cam1 or cam2.

    jump_unlabeled: bool, optional, default=False
        Aumatically jump to the next folder containing unlabeled images.

    Returns
    -------
    None

    Examples
    --------
    Standard use case

    >>> deeplabcut.label_frames('/myawesomeproject/reaching4thestars/config.yaml')

    To label multiple individuals (without having a multiple individuals project);
    otherwise this GUI is loaded automatically

    >>> deeplabcut.label_frames(
            '/analysis/project/reaching-task/config.yaml',
            multiple_individualsGUI=True,
        )

    To label non-default image types

    >>> label_frames(config, multiple=False, imtypes=['*.jpg','*.jpeg'])

    To label with epipolar lines projected from labels in another camera angle.

    >>> label_frames(
        config,
        config3d='/analysis/project/reaching-task/reaching-task-3d/config.yaml',
        sourceCam='cam1',
    )
    """
    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if cfg.get("multianimalproject", False) or multiple_individualsGUI:
        from deeplabcut.gui import multiple_individuals_labeling_toolbox

        multiple_individuals_labeling_toolbox.show(config, config3d, sourceCam, jump_unlabeled)
    else:
        from deeplabcut.gui import labeling_toolbox

        labeling_toolbox.show(config, config3d, sourceCam, imtypes, jump_unlabeled)

    os.chdir(startpath)


class Label_frames(wx.Panel):
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

        text = wx.StaticText(self, label="DeepLabCut - Step 3. Label Frames")
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
        self.sel_config.Bind(wx.EVT_BUTTON, self.select_config)

        self.help_button = wx.Button(self, label="Help")
        sizer.Add(self.help_button, pos=(4, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.check = wx.Button(self, label="Check Labels!")
        sizer.Add(self.check, pos=(5, 4), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.check.Bind(wx.EVT_BUTTON, self.check_labelF)
        self.check.Enable(True)

        # self.build = wx.Button(self, label="Build skeleton")
        # sizer.Add(self.build, pos=(4, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
        # self.build.Bind(wx.EVT_BUTTON, self.build_skeleton)
        # self.build.Enable(True)

        self.cfg = auxiliaryfunctions.read_config(self.config)
        if self.cfg.get("multianimalproject", False):

            self.check = wx.Button(self, label="Check Labels Individuals")
            sizer.Add(self.check, pos=(5, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
            self.check.Bind(wx.EVT_BUTTON, self.check_labelInd)
            self.check.Enable(True)

        self.ok = wx.Button(self, label="Label Frames")
        sizer.Add(self.ok, pos=(4, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.label_frames)

        self.reset = wx.Button(self, label="Reset")
        sizer.Add(
            self.reset, pos=(4, 1), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10
        )
        self.reset.Bind(wx.EVT_BUTTON, self.reset_label_frames)

        sizer.AddGrowableCol(2)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.label_frames"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")

    def check_labelF(self, event):
        dlg = wx.MessageDialog(
            None,
            "This will now plot the labeled frames after you have finished labeling!",
        )
        result = dlg.ShowModal()
        check_labels(self.config, visualizeindividuals=False)

    def check_labelInd(self, event):
        dlg = wx.MessageDialog(
            None,
            "This will now plot the labeled frames after you have finished labeling!",
        )
        result = dlg.ShowModal()
        check_labels(self.config, visualizeindividuals=True)

    # def build_skeleton(self, event):
    #    skeleton.SkeletonBuilder(self.config)

    def select_config(self, event):
        """
        """
        self.config = self.sel_config.GetPath()

    def label_frames(self, event):
        label_frames(self.config)

    def reset_label_frames(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
