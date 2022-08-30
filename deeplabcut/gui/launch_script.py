"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""
import os
import wx
import matplotlib as mpl

from deeplabcut.gui.create_new_project import Create_new_project
from deeplabcut.gui.welcome import Welcome
from deeplabcut.gui.widgets import BaseFrame
from deeplabcut.utils import auxiliaryfunctions


class MainFrame(BaseFrame):
    def __init__(self):
        super(MainFrame, self).__init__("DeepLabCut")
        self.statusbar.SetStatusText("www.deeplabcut.org")
        dlcparent_path = auxiliaryfunctions.get_deeplabcut_path()
        media_path = os.path.join(dlcparent_path, "gui", "media")
        logo = os.path.join(media_path, "logo.png")
        self.SetIcon(wx.Icon(logo))
        # Here we create a panel and a notebook on the panel
        self.panel = wx.Panel(self)
        self.nb = wx.Notebook(self.panel)
        # create the page windows as children of the notebook and add the pages to the notebook with the label to show on the tab
        page1 = Welcome(self.nb, self.gui_size)
        self.nb.AddPage(page1, "Welcome")

        page2 = Create_new_project(self.nb, self.gui_size)
        self.nb.AddPage(page2, "Manage Project")

        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.panel.SetSizer(self.sizer)


def launch_dlc():
    mpl.use("WxAgg")

    app = wx.App()
    app.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
    frame = MainFrame().Show()
    app.MainLoop()
