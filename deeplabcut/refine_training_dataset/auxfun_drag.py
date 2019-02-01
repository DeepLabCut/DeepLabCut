"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

"""
Class to drag,annotate and remove the data points. Use left click to drag the data points
around. Use right click to remove any unwanted data points. Internally the corresponding data
point is set to nan. When the user hovers the mouse over any data point, each data point is annotated with the labels and its corresponding likelihood. These annotations also move with the drag!
"""


import numpy as np
import wx

class DraggablePoint():
    lock = None #only one can be animated at a time
    def __init__(self, point,bodyParts,likelihood):
        self.point = point
        self.bodyParts = bodyParts
        self.likelihood = likelihood
        self.press = None
        self.background = None
        self.final_point = (0.0, 0.0)
        self.annot = self.point.axes.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.coords = []
#        self.adjust_original_labels = adjust_original_labels


    def connect(self):
        'connect to all the events we need'

        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidhover = self.point.figure.canvas.mpl_connect("motion_notify_event", self.on_hover)



    def on_press(self, event):
        """
        Define the event for the button press!
        """

        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        if event.button == 1:
            """
            This button press corresponds to the left click
            """
            self.press = (self.point.center), event.xdata, event.ydata
            DraggablePoint.lock = self
            canvas = self.point.figure.canvas
            axes = self.point.axes
            self.point.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.point.axes.bbox)
            axes.draw_artist(self.point)
            canvas.blit(axes.bbox)
        elif event.button == 3:
            """
            To remove a predicted label. Internally, the coordinates of the selected predicted label is replaced with nan. The user needs to right click for the event.After right
            click the data point is removed from the plot.
            """
            msg = wx.MessageBox('Do you want to remove %s ? You cannot undo this step!'%self.bodyParts, 'Remove!', wx.YES_NO | wx.ICON_WARNING)
            if msg == 2:
                self.press = None
                DraggablePoint.lock = None
                self.point.set_animated(False)
                self.background = None
                self.final_point = (np.nan,np.nan)
                self.point.center = (np.nan,np.nan)
                self.coords.append(self.final_point)
                self.point.figure.canvas.draw()

    def on_motion(self, event):
        """
        During the drag!
        """
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return

        if event.button == 1:
            self.point.center, xpress, ypress = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
            canvas = self.point.figure.canvas
            axes = self.point.axes
            # restore the background region
            canvas.restore_region(self.background)
            axes.draw_artist(self.point)
            canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return
        if event.button == 1:
            self.press = None
            DraggablePoint.lock = None
            self.point.set_animated(False)
            self.background = None
            self.point.figure.canvas.draw()
            self.final_point = (event.xdata, event.ydata,self.bodyParts)
            self.coords.append(self.final_point)

    def on_hover(self,event):
        """
        Annotate the lables and likelihood when the user hovers over the data points.
        """
        vis = self.annot.get_visible()
        if event.inaxes == self.point.axes:
            contains, attrd = self.point.contains(event)
            if contains:
                self.annot.xy = (self.point.center[0],self.point.center[1])
#                if self.adjust_original_labels == True:
#                    text = str(self.bodyParts)
#                else:
                text = str(self.bodyParts+',p='+ str("{0:.2f}".format(self.likelihood)))
                self.annot.set_text(text)
                self.annot.get_bbox_patch().set_alpha(0.4)
                self.annot.set_visible(True)
                self.point.figure.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)


    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        self.point.figure.canvas.mpl_disconnect(self.cidhover)