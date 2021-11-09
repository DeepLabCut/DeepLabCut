import sys

from createStatus import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QTabWidget
from PyQt5.QtWidgets import QWidget, QLabel, QRadioButton, QFormLayout, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QGridLayout, QFrame, QGraphicsDropShadowEffect, QMenu

from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QSizePolicy, QStatusBar

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider
)
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui

from MainApp import *
from CreateProject import *
import deeplabcut
#from deeplabcut.gui import canvas, widgets
#from deeplabcut.utils import auxiliaryfunctions, video_reader
import os
# import Dark Mode Theme
import qdarkstyle
#import breeze_resources

#import qrc_resources


class MainWindow(QtWidgets.QMainWindow):
    #config_loaded = QtCore.pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()

        desktop = QtWidgets.QDesktopWidget().screenGeometry(0)
        self.screen_width = desktop.width()
        self.screen_height = desktop.height()

        self.welcome_page('Welcome.png')
        self.window_set()
        self.default_set()

        names = ['new_project.png', 'open.png', 'help.png']
        self.createActions(names)
        self.createMenuBar()
        self.createToolBars(0)

        self.config = None
        self.cfg = dict()


        #self.canvas = canvas.Canvas(self.main_panel)
        #self.figtitle = widgets.ClickableLabel()
        #self.figtitle.setAlignment(QtCore.Qt.AlignCenter)
        #self.figtitle.setEnabled(False)


    def window_set(self):
        self.setWindowTitle("DeepLabCut")
        #self.setGeometry(300,150,1500,750)
        self.setMinimumSize(1500, 750)

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
        logo = logo_dir + '/pictures/logo.png'
        self.setWindowIcon(QIcon(logo))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName('Status Bar')

    def set_pic(self, name):
        pic_dir = os.path.dirname(os.path.realpath(name)) + os.path.sep
        file = pic_dir + '/pictures/' + name
        pixmap = QPixmap(file)  # C:\Users\User\PycharmProjects
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lbl.setScaledContents(True)
        lbl.setMaximumWidth(1565)
        return lbl

    def welcome_page(self, name):
        layout = QVBoxLayout()

        layout.addWidget(self.set_pic(name))
        layout.setAlignment(Qt.AlignTop)

        #pol = QSizePolicy()
        #pol.setHorizontalPolicy(QSizePolicy.Ignored)
        #lbl.setSizePolicy(pol)

        lbl_welcome1 = QLabel("Welcome to the DeepLabCut Project Manager GUI!")
        lbl_welcome1.setAlignment(Qt.AlignCenter)
        lbl_welcome2 = QLabel("To get started, please click on the 'File'")
        lbl_welcome2.setAlignment(Qt.AlignCenter)
        lbl_welcome3 = QLabel("tab to cteate or load an existing project.")
        lbl_welcome3.setAlignment(Qt.AlignCenter)

        layout.addWidget(lbl_welcome1)
        layout.addWidget(lbl_welcome2)
        layout.addWidget(lbl_welcome3)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        #desktop = QtWidgets.QDesktopWidget().screenGeometry(0)
        #self.screen_width = desktop.width()
        #self.screen_height = desktop.height()
        #self.setFocus()
        #self.activateWindow()

    def project_folder(self):
        return self.cfg.get('project_path', os.path.expanduser('~/Desktop'))

    def default_set(self):
        self.name_default = ''
        self.proj_default = ''
        self.exp_default = ''
        self.loc_default = 'C:/'


    def createActions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")
        #self.newAction.setToolTip('Create new project')

        self.newAction.setIcon(QIcon("icons/"+names[0]))
        self.newAction.setShortcut('Ctrl+N')

        self.newAction.triggered.connect(self._create)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(QIcon("icons/"+names[1]))
        self.openAction.setShortcut('Ctrl+O')
        #self.openAction.triggered.connect(self.parent.open_project)

        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.lightmodeAction = QAction("&Light theme", self)
        self.lightmodeAction.triggered.connect(self.lightmode)
        self.darkmodeAction = QAction("&Dark theme", self)
        self.darkmodeAction.triggered.connect(self.darkmode)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(QIcon("icons/"+names[2]))

        self.aboutAction = QAction("&Learn DLC", self)


    def createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        self.fileMenu = QMenu("&File", self)
        menuBar.addMenu(self.fileMenu)


        self.fileMenu.addAction(self.newAction)
        self.fileMenu.addAction(self.openAction)

        findMenu = self.fileMenu.addMenu("Open Recent")
        findMenu.addAction("File 1")
        findMenu.addAction("File 2")
        self.fileMenu.addAction(self.saveAction)
        self.fileMenu.addAction(self.exitAction)
        # View menu
        viewMenu = QMenu("&View", self)
        mode = viewMenu.addMenu("Appearance")
        menuBar.addMenu(viewMenu)
        mode.addAction(self.lightmodeAction)
        mode.addAction(self.darkmodeAction)

        # Help menu
        helpMenu = QMenu("&Help", self)
        menuBar.addMenu(helpMenu)
        helpMenu.addAction(self.helpAction)
        helpMenu.adjustSize()
        helpMenu.addAction(self.aboutAction)

    def updateMenuBar(self):
        self.fileMenu.removeAction(self.newAction)
        self.fileMenu.removeAction(self.openAction)
        #self.fileMenu.addAction(self.helpAction)
        #helpMenu.adjustSize()

        #self.fileMenu.addAction(self.newAction)
        #self.fileMenu.addAction(self.openAction)

    def createToolBars(self, flag):
        # File toolbar
        if flag == 0:
            self.fileToolBar = self.addToolBar("File")

        self.fileToolBar.addAction(self.newAction)
        self.fileToolBar.addAction(self.openAction)
        self.fileToolBar.addAction(self.helpAction)
    def remove_action(self):
        self.fileToolBar.removeAction(self.newAction)
        self.fileToolBar.removeAction(self.openAction)
        self.fileToolBar.removeAction(self.helpAction)


    @QtCore.pyqtSlot()
    def _create(self):
        # action = self.sender()
        self.class_Second = CreateProject(self)
        self.class_Second.show()

    def open_project(self):
        config = QFileDialog.getOpenFileName(self, caption='Select a configuration file',
                                                       directory=self.project_folder, filter='Config files (*.yaml)')[0]
        if not config:
            return
        self.load_config(config)

    def load_config(self, config):
        self.config = config
        self.cfg = auxiliaryfunctions.read_config(config)
        self.config_loaded.emit()
        print(f'Project "{self.cfg["Task"]}" successfully loaded.')

    def darkmode(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.welcome_page('Welcome2.png')

        names = ['new_project2.png', 'open2.png', 'help2.png']
        self.remove_action()
        self.createActions(names)
        self.updateMenuBar()
        self.createToolBars(1)
    def lightmode(self):
        self.setStyleSheet('Fusion')
        self.welcome_page('Welcome.png')

        names = ['new_project.png', 'open.png', 'help.png']
        self.remove_action()
        self.createActions(names)
        self.createToolBars(1)
        self.updateMenuBar()


