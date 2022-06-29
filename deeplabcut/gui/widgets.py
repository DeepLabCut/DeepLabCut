"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import numpy as np
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class BasePanel(wx.Panel):
    def __init__(self, parent, config, gui_size, **kwargs):
        h = gui_size[0] / 2
        w = gui_size[1] / 3
        super(BasePanel, self).__init__(parent, -1, style=wx.SUNKEN_BORDER, size=(h, w))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.orig_xlim = None
        self.orig_ylim = None
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return self.figure, self.axes, self.canvas

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)


class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)


class BaseFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, frame_title="", parent=None, imtypes=None):
        # Setting the GUI size and panels design
        displays = (
            wx.Display(i) for i in range(wx.Display.GetCount())
        )  # Gets the number of displays
        screenSizes = [
            display.GetGeometry().GetSize() for display in displays
        ]  # Gets the size of each display
        index = 0  # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (int(screenWidth * 0.8), int(screenHeight * 0.75))
        self.imtypes = imtypes  # imagetypes to look for in folder e.g. *.png

        wx.Frame.__init__(
            self,
            parent,
            id=wx.ID_ANY,
            title=frame_title,
            size=wx.Size(self.gui_size),
            pos=wx.DefaultPosition,
            style=wx.RESIZE_BORDER | wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
        )

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")

        # This sets the minimum size of the GUI. It can scale now!
        self.SetSizeHints(wx.Size(800, 600))

    @staticmethod
    def calc_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def homeButton(self, event):
        self.image_panel.resetView()
        self.figure.canvas.draw()
        self.updateZoomPan(self)
        self.zoom.SetValue(False)
        self.pan.SetValue(False)
        self.statusbar.SetStatusText("")

    def updateZoomPan(self, *args):
        # Checks if zoom/pan button is ON
        if self.pan.GetValue():
            self.toolbar.pan()
            self.pan.SetValue(False)
        if self.zoom.GetValue():
            self.toolbar.zoom()
            self.zoom.SetValue(False)

    def panButton(self, event):
        if self.pan.GetValue():
            self.toolbar.pan()
            self.statusbar.SetStatusText("Pan On")
            self.zoom.SetValue(False)
        else:
            self.toolbar.pan()
            self.statusbar.SetStatusText("Pan Off")

    def zoomButton(self, event):
        if self.zoom.GetValue():
            # Save pre-zoom xlim and ylim values
            self.prezoom_xlim = self.axes.get_xlim()
            self.prezoom_ylim = self.axes.get_ylim()
            self.toolbar.zoom()
            self.statusbar.SetStatusText("Zoom On")
            self.pan.SetValue(False)
        else:
            self.toolbar.zoom()
            self.statusbar.SetStatusText("Zoom Off")

    def onZoom(self, ax):
        # See if axis limits have actually changed
        curr_xlim = self.axes.get_xlim()
        curr_ylim = self.axes.get_ylim()
        if self.zoom.GetValue() and not (
            self.prezoom_xlim[0] == curr_xlim[0]
            and self.prezoom_xlim[1] == curr_xlim[1]
            and self.prezoom_ylim[0] == curr_ylim[0]
            and self.prezoom_ylim[1] == curr_ylim[1]
        ):
            self.updateZoomPan()
            self.statusbar.SetStatusText("Zoom Off")

    def lockChecked(self, event):
        self.cb = event.GetEventObject()
        self.view_locked = self.cb.GetValue()
