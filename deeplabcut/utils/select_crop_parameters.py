"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""
from __future__ import print_function
import wx
import cv2
import matplotlib
import argparse
from deeplabcut.utils import auxiliaryfunctions
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(wx.Panel):

    def __init__(self, parent,config,gui_size,**kwargs):
        h=gui_size[0]/2
        w=gui_size[1]/3
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER,size=(h,w))

        self.figure = matplotlib.figure.Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return(self.figure,self.axes,self.canvas)

class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)

class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent,config,image):
# Settting the GUI size and panels design
        displays = (wx.Display(i) for i in range(wx.Display.GetCount())) # Gets the number of displays
        screenSizes = [display.GetGeometry().GetSize() for display in displays] # Gets the size of each display
        index = 0 # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (screenWidth*0.7,screenHeight*0.85)

        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'DeepLabCut2.0 - Select Crop Parameters',
                            size = wx.Size(self.gui_size), pos = wx.DefaultPosition, style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")

        self.SetSizeHints(wx.Size(self.gui_size)) #  This sets the minimum size of the GUI. It can scale now!
        
###################################################################################################################################################
# Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!
        topSplitter = wx.SplitterWindow(self)

        self.image_panel = ImagePanel(topSplitter, config,self.gui_size)
        self.widget_panel = WidgetPanel(topSplitter)
        
        topSplitter.SplitHorizontally(self.image_panel, self.widget_panel,sashPosition=self.gui_size[1]*0.83)#0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

###################################################################################################################################################
# Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)

        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        widgetsizer.Add(self.help , 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)

        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Save parameters and Quit")
        widgetsizer.Add(self.quit , 1, wx.ALL, 15)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)

        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        self.widget_panel.Layout()
        
# Variables initialization
        self.image = image
        self.coords = []
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.cfg = auxiliaryfunctions.read_config(config)
        MainFrame.show_image(self)

    def quitButton(self, event):
        """
        Quits the GUI
        """
        self.statusbar.SetStatusText("")
        dlg = wx.MessageDialog(None,"Are you sure?", "Quit!",wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            self.Destroy()

    def show_image(self):
        self.figure,self.axes,self.canvas = self.image_panel.getfigure()
        frame=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.ax = self.axes.imshow(frame)
        self.figure.canvas.draw()
        self.cid = RectangleSelector(self.axes, self.line_select_callback,drawtype='box', useblit=False,button=[1], minspanx=5, minspany=5,spancoords='pixels',interactive=True)
        self.canvas.mpl_connect('key_press_event', self.cid)
        
        
    def line_select_callback(self,eclick, erelease):
        'eclick and erelease are the press and release events'
        global coords
        new_x1, new_y1 = eclick.xdata, eclick.ydata
        new_x2, new_y2 = erelease.xdata, erelease.ydata
        coords = [str(int(new_x1)),str(int(new_x2)),str(int(new_y1)),str(int(new_y2))]
        self.coords = coords
        return(self.coords)
        
    def helpButton(self,event):
        """
        Opens Instructions
        """
        wx.MessageBox('1. Use left click to select the region of interest. A red box will be drawn around the selected region. \n\n2. Use the corner points to expand the box and center to move the box around the image. \n\n3. Click ''Save parameters and Quit'' to save the croppeing parameters and close the GUI. \n\n Click OK to continue', 'Instructions to use!', wx.OK | wx.ICON_INFORMATION)

def show(config,image):
    import imageio
    imageio.plugins.ffmpeg.download()
    app = wx.App()
    MainFrame(None,config,image).Show()
    app.MainLoop()
    return(coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config','image')
    cli_args = parser.parse_args()
