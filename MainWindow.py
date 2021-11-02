import sys

from createStatus import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QTabWidget
from PyQt5.QtWidgets import QWidget, QLabel, QRadioButton, QFormLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QGridLayout, QFrame, QGraphicsDropShadowEffect

from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QSizePolicy, QStatusBar

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider
)
from PyQt5.QtGui import QPixmap
from MainApp import *
#import deeplabcut
import os


class MainWindow(QtWidgets.QMainWindow):
    config_loaded = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(MainApp, self).__init__(*args, **kwargs)
        self.setWindowTitle('DeepLabCut')