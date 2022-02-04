import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

class BasePanel(QtWidgets.QFrame):
    def __init__(self, parent, config, gui_size, **kwargs):
        super(BasePanel, self).__init__(parent)

        self.setWindowTitle('New Project')
        self.setMinimumSize(900, 500)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.orig_xlim = None
        self.orig_ylim = None
        self.sizer = QtWidgets.QHBoxLayout()
        self.sizer.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sizer.addWidget(self.canvas)

    def getfigure(self):
        # Returns the figure, axes and canvas
        return self.figure, self.axes, self.canvas

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)

class BaseFrame(QtWidgets.QFrame):
    """Contains the main GUI and button boxes"""

    def __init__(self, frame_title="", parent=None, imtypes=None):
        # Settting the GUI size and panels design


        index = 0  # For display 1.
        screenWidth = 500
        screenHeight = 900
        self.gui_size = (screenWidth * 0.8, screenHeight * 0.75)
        self.imtypes = imtypes  # imagetypes to look for in folder e.g. *.png




