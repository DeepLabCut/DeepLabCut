"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse

import wx
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from deeplabcut.gui.widgets import BasePanel, WidgetPanel, BaseFrame


class MainFrame(BaseFrame):
    def __init__(self, parent, config, image):
        super(MainFrame, self).__init__(
            "DeepLabCut2.0 - Select Crop Parameters", parent
        )

        ###################################################################################################################################################
        # Splitting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!
        topSplitter = wx.SplitterWindow(self)

        self.image_panel = BasePanel(topSplitter, config, self.gui_size)
        self.widget_panel = WidgetPanel(topSplitter)

        topSplitter.SplitHorizontally(
            self.image_panel, self.widget_panel, sashPosition=self.gui_size[1] * 0.83
        )  # 0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

        ###################################################################################################################################################
        # Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)

        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        widgetsizer.Add(self.help, 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)

        self.quit = wx.Button(
            self.widget_panel, id=wx.ID_ANY, label="Save parameters and Quit"
        )
        widgetsizer.Add(self.quit, 1, wx.ALL, 15)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)

        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        self.widget_panel.Layout()

        # Variables initialization
        self.image = image
        self.coords = []
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        # self.cfg = auxiliaryfunctions.read_config(config)
        MainFrame.show_image(self)

    def quitButton(self, event):
        """
        Quits the GUI
        """
        # self.statusbar.SetStatusText("")
        # dlg = wx.MessageDialog(None,"Are you sure?", "Quit!",wx.YES_NO | wx.ICON_WARNING)
        # result = dlg.ShowModal()
        # if result == wx.ID_YES:
        self.Destroy()

    def show_image(self):
        self.figure, self.axes, self.canvas = self.image_panel.getfigure()
        self.ax = self.axes.imshow(self.image)
        self.figure.canvas.draw()
        self.cid = RectangleSelector(
            self.axes,
            self.line_select_callback,
            drawtype="box",
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas.mpl_connect("key_press_event", self.cid)

    def line_select_callback(self, eclick, erelease):
        "eclick and erelease are the press and release events"
        new_x1, new_y1 = eclick.xdata, eclick.ydata
        new_x2, new_y2 = erelease.xdata, erelease.ydata
        coords = [
            str(int(new_x1)),
            str(int(new_x2)),
            str(int(new_y1)),
            str(int(new_y2)),
        ]
        self.coords = coords

    def helpButton(self, event):
        """
        Opens Instructions
        """
        wx.MessageBox(
            "1. Use left click to select the region of interest. A red box will be drawn around the selected region. \n\n2. Use the corner points to expand the box and center to move the box around the image. \n\n3. Click "
            "Save parameters and Quit"
            " to save the croppeing parameters and close the GUI. \n\n Click OK to continue",
            "Instructions to use!",
            wx.OK | wx.ICON_INFORMATION,
        )


def show(config, image):
    app = wx.App()
    main = MainFrame(None, config, image)
    main.Show()
    app.MainLoop()
    return main.coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", "image")
    cli_args = parser.parse_args()
