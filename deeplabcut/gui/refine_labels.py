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

from deeplabcut.gui import LOGO_PATH
from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path


def refine_labels(config, multianimal=False, jump_unlabeled=False):
    """Refines the labels of the outlier frames extracted from the analyzed videos.

    Helps in augmenting the training dataset. Use ``deeplabcut.analyze_videos`` to
    analyze a video and extracts the outlier frames using
    ``deeplabcut.extract_outlier_frames`` before refining the labels.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    multianimal, bool, optional, default=False
        If ``True``, a user can label multiple individuals. Note for
        ``"multianimalproject=True"`` this is automatically used.

    jump_unlabeled: bool, optional, default=False
        Aumatically jump to the next folder containing unlabeled images.

    Examples
    --------
    >>> deeplabcut.refine_labels('/analysis/project/reaching-task/config.yaml')
    """

    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if not multianimal and not cfg.get("multianimalproject", False):
        from deeplabcut.gui import refinement

        refinement.show(config, jump_unlabeled)
    else:  # loading multianimal labeling GUI
        from deeplabcut.gui import multiple_individuals_refinement_toolbox

        multiple_individuals_refinement_toolbox.show(config, jump_unlabeled)

    os.chdir(startpath)


class Refine_labels(wx.Panel):
    """
    """

    def __init__(self, parent, gui_size, cfg, page):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initialization
        self.method = "automatic"
        self.config = cfg
        self.page = page
        # design the panel
        sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Step 9. Refine labels")
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

        sizer.Add(
            self.sel_config, pos=(2, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5
        )
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        self.help_button = wx.Button(self, label="Help")
        sizer.Add(self.help_button, pos=(4, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="LAUNCH")
        sizer.Add(self.ok, pos=(4, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.refine_labels)

        self.merge = wx.Button(self, label="Merge dataset")
        sizer.Add(self.merge, pos=(4, 3), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.merge.Bind(wx.EVT_BUTTON, self.merge_dataset)
        self.merge.Enable(False)

        self.reset = wx.Button(self, label="Reset")
        sizer.Add(self.reset, pos=(4, 1), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_refine_labels)

        sizer.AddGrowableCol(2)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.refine_labels"
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
        self.sel_config.SetPath("")
        self.merge.Enable(False)
