import argparse
import glob
import os
import os.path
from pathlib import Path

import cv2
import re
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

from PyQt5 import QtWidgets

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from deeplabcut.gui import auxfun_drag
from deeplabcut.gui.widgets import BasePanel, WidgetPanel, BaseFrame
from deeplabcut.utils import auxiliaryfunctions, auxiliaryfunctions_3d


class ImagePanel(BasePanel):
    def __init__(self, parent, config, config3d, sourceCam, gui_size, **kwargs):
        super(ImagePanel, self).__init__(parent, config, gui_size, **kwargs)
        self.config = config
        self.config3d = config3d
        self.sourceCam = sourceCam
        self.toolbar = None

# class ScrollPanel(SP.ScrolledPanel):
#     def __init__(self, parent):
#         print('scroll')

class MainFrame(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('DeepLabCut2.0 - Labeling ToolBox')
        self.setMinimumSize(1500, 750)

        self.logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
        self.logo = self.logo_dir + '/pictures/logo.png'
        self.setWindowIcon(QIcon(self.logo))

        # self.statusbar = self.statusBar()
        # self.statusbar.showMessage("Looking for a folder to start labeling. Click 'Load frames' to begin.")
        # self.setStatusBar(self.statusbar)

        hbox = QHBoxLayout(self)

        topleft = QFrame()
        topleft.setFrameShape(QFrame.StyledPanel)
        topright = QFrame()
        topright.setFrameShape(QFrame.StyledPanel)

        bottom = QFrame()
        bottom.setFrameShape(QFrame.StyledPanel)

        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(topleft)
        splitter1.addWidget(topright)
        splitter1.setStretchFactor(1, 10)
        #splitter1.setSizes([500, 500])

        splitter2 = QSplitter(Qt.Vertical)
        splitter2.setStretchFactor(1, 10)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)

        hbox.addWidget(splitter2)
       # hbox.setAlignment(Qt.AlignTop | Qt.AlignRight)
        self.setLayout(hbox)

        # self.image_panel = ImagePanel(
        #     splitter2
        # )
        #, config, config3d, sourceCam, self.gui_size
        # self.choice_panel = ScrollPanel(vSplitter)
        # vSplitter.SplitVertically(
        #     self.image_panel, self.choice_panel, sashPosition=self.gui_size[0] * 0.8
        # )



        #self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

def show(): #config, config3d, sourceCam, imtypes=["*.png"]
    print('show')

    frame = MainFrame() # config, imtypes, config3d, sourceCam
    #frame.setWindowModality(Qt.ApplicationModal)
    frame.exec_()




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config")
#     parser.add_argument("config3d")
#     parser.add_argument("sourceCam")
#     cli_args = parser.parse_args()