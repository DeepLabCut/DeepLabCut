import glob
import os
import os.path
from pathlib import Path

import cv2
import re
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QFrame,
    QVBoxLayout,
    QScrollArea,
    QHBoxLayout,
    QSlider,
    QLabel,
    QCheckBox,
    QButtonGroup,
    QStatusBar,
    QSplitter,
    QMessageBox,
    QPushButton,
)

from PyQt5.QtWidgets import QWidget

from mpl_toolkits.axes_grid1 import make_axes_locatable

from deeplabcut.gui import auxfun_drag
from deeplabcut.utils import auxiliaryfunctions, auxiliaryfunctions_3d


class ImagePanel(QFrame):
    def __init__(self, parent, config, config3d, sourceCam, dir=None):
        super(ImagePanel, self).__init__(parent)
        self.config = config
        self.config3d = config3d
        self.sourceCam = sourceCam
        self.toolbar = None
        self.dir = dir

        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.figure)
        self.orig_xlim = None
        self.orig_ylim = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return self.figure, self.axes, self.canvas

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)

    def retrieveData_and_computeEpLines(self, img, imNum):

        # load labeledPoints and fundamental Matrix

        if self.config3d is not None:
            cfg_3d = auxiliaryfunctions.read_config(self.config3d)
            cams = cfg_3d["camera_names"]
            path_camera_matrix = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)[2]
            path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
            stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)

            for cam in cams:
                if cam in img:
                    labelCam = cam
                    if self.sourceCam is None:
                        sourceCam = [
                            otherCam for otherCam in cams if cam not in otherCam
                        ][
                            0
                        ]  # WHY?
                    else:
                        sourceCam = self.sourceCam

            sourceCamIdx = np.where(np.array(cams) == sourceCam)[0][0]
            labelCamIdx = np.where(np.array(cams) == labelCam)[0][0]

            if sourceCamIdx < labelCamIdx:
                camera_pair = cams[sourceCamIdx] + "-" + cams[labelCamIdx]
                sourceCam_numInPair = 1
            else:
                camera_pair = cams[labelCamIdx] + "-" + cams[sourceCamIdx]
                sourceCam_numInPair = 2

            fundMat = stereo_file[camera_pair]["F"]
            sourceCam_path = os.path.split(img.replace(labelCam, sourceCam))[0]

            cfg = auxiliaryfunctions.read_config(self.config)
            scorer = cfg["scorer"]

            try:
                dataFrame = pd.read_hdf(
                    os.path.join(sourceCam_path, "CollectedData_" + scorer + ".h5")
                )
                dataFrame.sort_index(inplace=True)
            except IOError:
                print(
                    "source camera images have not yet been labeled, or you have opened this folder in the wrong mode!"
                )
                return None, None, None

            # Find offset terms for drawing epipolar Lines
            # Get crop params for camera being labeled
            foundEvent = 0
            eventSearch = re.compile(os.path.split(os.path.split(img)[0])[1])
            cropPattern = re.compile("[0-9]{1,4}")
            with open(self.config, "rt") as config:
                for line in config:
                    if foundEvent == 1:
                        crop_labelCam = np.int32(re.findall(cropPattern, line))
                        break
                    if eventSearch.search(line) != None:
                        foundEvent = 1
            # Get crop params for other camera
            foundEvent = 0
            eventSearch = re.compile(os.path.split(sourceCam_path)[1])
            cropPattern = re.compile("[0-9]{1,4}")
            with open(self.config, "rt") as config:
                for line in config:
                    if foundEvent == 1:
                        crop_sourceCam = np.int32(re.findall(cropPattern, line))
                        break
                    if eventSearch.search(line) != None:
                        foundEvent = 1

            labelCam_offsets = [crop_labelCam[0], crop_labelCam[2]]
            sourceCam_offsets = [crop_sourceCam[0], crop_sourceCam[2]]

            sourceCam_pts = np.asarray(dataFrame, dtype=np.int32)
            sourceCam_pts = sourceCam_pts.reshape(
                (sourceCam_pts.shape[0], int(sourceCam_pts.shape[1] / 2), 2)
            )
            sourceCam_pts = np.moveaxis(sourceCam_pts, [0, 1, 2], [1, 0, 2])
            sourceCam_pts[..., 0] = sourceCam_pts[..., 0] + sourceCam_offsets[0]
            sourceCam_pts[..., 1] = sourceCam_pts[..., 1] + sourceCam_offsets[1]

            sourcePts = sourceCam_pts[:, imNum, :]

            epLines_source2label = cv2.computeCorrespondEpilines(
                sourcePts, int(sourceCam_numInPair), fundMat
            )
            epLines_source2label.reshape(-1, 3)

            return epLines_source2label, sourcePts, labelCam_offsets

        else:
            return None, None, None

    def drawEpLines(self, drawImage, lines, sourcePts, offsets, colorIndex, cmap):
        height, width, depth = drawImage.shape
        for line, pt, cIdx in zip(lines, sourcePts, colorIndex):
            if pt[0] > -1000:
                coeffs = line[0]
                x0, y0 = map(int, [0 - offsets[0], -coeffs[2] / coeffs[1] - offsets[1]])
                x1, y1 = map(
                    int,
                    [
                        width,
                        -(coeffs[2] + coeffs[0] * (width + offsets[0])) / coeffs[1]
                        - offsets[1],
                    ],
                )
                cIdx = cIdx / 255
                color = cmap(cIdx, bytes=True)[:-1]
                color = tuple([int(x) for x in color])
                drawImage = cv2.line(drawImage, (x0, y0), (x1, y1), color, 1)

        return drawImage

    def drawplot(
        self, img, img_name, itr, index, bodyparts, cmap, keep_view=False, dir=None
    ):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.clear()
        # convert the image to RGB as you are showing the image with matplotlib

        image = dir + "/" + img.split("/")[-1]

        im = cv2.imread(image)[..., ::-1]

        colorIndex = np.linspace(np.max(im), np.min(im), len(bodyparts))
        # draw epipolar lines
        epLines, sourcePts, offsets = self.retrieveData_and_computeEpLines(img, itr)

        if epLines is not None:
            im = self.drawEpLines(
                im.copy(), epLines, sourcePts, offsets, colorIndex, cmap
            )
        ax = self.axes.imshow(im, cmap=cmap)

        self.orig_xlim = self.axes.get_xlim()
        self.orig_ylim = self.axes.get_ylim()
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.figure.colorbar(
            ax, cax=cax, spacing="proportional", ticks=colorIndex
        )
        cbar.set_ticklabels(bodyparts)

        self.axes.set_title(str(str(itr) + "/" + str(len(index) - 1) + " " + img_name))
        if keep_view:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        if self.toolbar is None:
            self.toolbar = NavigationToolbar(self.canvas, self)

        return (self.figure, self.axes, self.canvas, self.toolbar)

    def getColorIndices(self, img, bodyparts, dir):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        image = dir + "/" + img.split("/")[-1]
        im = cv2.imread(image)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(im))
        ticks = np.linspace(0, np.max(im), len(bodyparts))[::-1]

        return norm, ticks


class ScrollPanel(QFrame):
    def __init__(self, parent):
        super(ScrollPanel, self).__init__(parent)
        scrollArea = QScrollArea()

    def addRadioButtons(self, bodyparts, fileIndex, markersize):
        """
        Adds radio buttons for each bodypart on the right panel
        """
        self.choiceBox = QVBoxLayout(self)
        self.choiceBox.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.checkBox = QCheckBox("Adjust marker size")
        self.choiceBox.addWidget(self.checkBox)

        slider_vbox = QVBoxLayout(self)
        slider_vbox.setContentsMargins(0, 0, 0, 20)
        slider_hbox = QHBoxLayout(self)
        label_minimum = QLabel("1")
        self.label = QLabel(str(markersize), alignment=Qt.AlignCenter)
        label_maximum = QLabel(str(markersize * 3))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(markersize * 3)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setSliderPosition(markersize)

        self.label = QLabel(str(self.slider.value()), alignment=Qt.AlignCenter)
        self.slider.valueChanged.connect(self.setvalue)

        slider_hbox.addWidget(label_minimum)
        slider_hbox.addWidget(self.slider)
        slider_hbox.addWidget(label_maximum)
        slider_vbox.addLayout(slider_hbox)
        slider_vbox.addWidget(self.label)

        self.slider.setEnabled(False)
        self.choiceBox.addLayout(slider_vbox)

        self.btngroup = QButtonGroup()
        fieldrbns_text = QtWidgets.QLabel("Select a bodypart to label")
        fieldrbns_text.setContentsMargins(0, 20, 0, 0)
        self.choiceBox.addWidget(fieldrbns_text)
        self.fieldradiobox = dict()
        for l in bodyparts:
            self.fieldradiobox[l] = QtWidgets.QRadioButton(l)
            self.btngroup.addButton(self.fieldradiobox[l])
            self.choiceBox.addWidget(self.fieldradiobox[l])
        self.fieldradiobox[bodyparts[0]].setChecked(True)
        return (self.choiceBox, self.btngroup, self.slider, self.checkBox)

    def setvalue(self):
        self.label.setText(str(self.slider.value()))

    def clearBoxer(self):
        self.choiceBox.Clear(True)


class MainFrame(QMainWindow):
    def __init__(self, parent, config, imtypes, config3d, sourceCam):
        super(MainFrame, self).__init__(parent)

        self.setWindowTitle("DeepLabCut2.0 - Labeling ToolBox")
        self.setMinimumSize(1600, 750)
        size = self.size()
        self.imtypes = imtypes

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(
            "Looking for a folder to start labeling. Click 'Load frames' to begin."
        )

        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        self.setWindowIcon(QIcon(self.logo))

        centralWidget = QWidget(self)
        hbox = QHBoxLayout()

        self.image_panel = ImagePanel(self, config, config3d, sourceCam)

        self.image_panel.setFrameShape(QFrame.StyledPanel)
        self.image_panel.setMinimumWidth(size.width() * 0.75)
        self.image_panel.setMinimumHeight(size.height() * 0.75)

        self.choice_panel = ScrollPanel(self)
        self.choice_panel.setFrameShape(QFrame.StyledPanel)

        bottom = QFrame()
        bottom.setFrameShape(QFrame.StyledPanel)

        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(self.image_panel)
        splitter1.addWidget(self.choice_panel)

        splitter2 = QSplitter(Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)

        hbox.addWidget(splitter2)
        centralWidget.setLayout(hbox)
        self.setCentralWidget(centralWidget)

        self.adjustSize()

        ###################################################################################################################################################
        # Add Buttons to the WidgetPanel and bind them to their respective functions.

        self.l_btns = QHBoxLayout(bottom)
        self.l_btns.setSpacing(40)
        self.l_btns.setContentsMargins(20, 10, 20, 10)

        self.load = QtWidgets.QPushButton("Load frames")
        self.l_btns.addWidget(self.load, alignment=Qt.AlignCenter)
        self.load.setDefault(True)
        self.load.clicked.connect(self.browseDir)

        self.prev = QtWidgets.QPushButton("<<Previous")
        self.l_btns.addWidget(self.prev, alignment=Qt.AlignCenter)
        self.prev.setEnabled(False)
        self.prev.clicked.connect(self.prevImage)

        self.next = QtWidgets.QPushButton("Next>>")
        self.l_btns.addWidget(self.next, alignment=Qt.AlignCenter)
        self.next.setEnabled(False)
        self.next.clicked.connect(self.nextImage)

        self.help = QtWidgets.QPushButton("Help")
        self.l_btns.addWidget(self.help, alignment=Qt.AlignCenter)
        self.help.clicked.connect(self.helpButton)

        self.zoom = QtWidgets.QPushButton("Zoom")
        self.l_btns.addWidget(self.zoom, alignment=Qt.AlignCenter)
        self.zoom.setCheckable(True)
        self.zoom.setEnabled(False)
        self.zoom.clicked.connect(self.zoomButton)

        self.home = QtWidgets.QPushButton("Home")
        self.l_btns.addWidget(self.home, alignment=Qt.AlignCenter)
        self.home.setEnabled(False)
        self.home.clicked.connect(self.homeButton)

        self.pan = QtWidgets.QPushButton("Pan")
        self.l_btns.addWidget(self.pan, alignment=Qt.AlignCenter)
        self.pan.setCheckable(True)
        self.pan.setEnabled(False)
        self.pan.clicked.connect(self.panButton)

        self.lock = QCheckBox("Lock View")
        self.l_btns.addWidget(self.lock, alignment=Qt.AlignCenter)
        self.lock.stateChanged.connect(self.lockChecked)

        self.save = QtWidgets.QPushButton("Save")
        self.l_btns.addWidget(self.save, alignment=Qt.AlignCenter)
        self.save.setEnabled(False)
        self.save.clicked.connect(self.saveDataSet)

        self.quit = QtWidgets.QPushButton("Quit")
        self.l_btns.addWidget(self.quit, alignment=Qt.AlignCenter)
        self.quit.clicked.connect(self.quitButton)

        ###############################################################################################################################
        # Variables initialization
        self.currentDirectory = os.getcwd()
        self.index = []
        self.iter = []
        self.file = 0
        self.updatedCoords = []
        self.dataFrame = None
        self.config_file = config
        self.new_labels = False
        self.axes = None
        self.buttonCounter = []
        self.bodyparts2plot = []
        self.drs = []
        self.num = []
        self.view_locked = False
        # Workaround for MAC - xlim and ylim changed events seem to be
        # triggered too often so need to make sure that the
        # xlim and ylim have actually changed before turning zoom off
        self.prezoom_xlim = []
        self.prezoom_ylim = []

    #################################################################

    def updateZoomPan(self, *args):
        # Checks if zoom/pan button is ON

        if self.pan.isChecked():
            self.toolbar.pan()
            self.pan.setChecked(False)

        if self.zoom.isChecked():
            self.toolbar.zoom()
            self.zoom.setChecked(False)

    def lockChecked(self, event):
        print("lockChecked")

        # self.cb = event.GetEventObject()
        # self.view_locked = self.cb.GetValue()
        # TODO: finish

    def onZoom(self, ax):
        # See if axis limits have actually changed
        curr_xlim = self.axes.get_xlim()
        curr_ylim = self.axes.get_ylim()

        if self.zoom.text() and not (
            self.prezoom_xlim[0] == curr_xlim[0]
            and self.prezoom_xlim[1] == curr_xlim[1]
            and self.prezoom_ylim[0] == curr_ylim[0]
            and self.prezoom_ylim[1] == curr_ylim[1]
        ):
            self.updateZoomPan()
            self.statusBar.showMessage("Zoom Off")

    def panButton(self):

        if self.pan.isChecked():
            self.toolbar.pan()
            self.statusBar.showMessage("Pan On")
            self.zoom.setChecked(False)
        else:
            self.toolbar.pan()
            self.statusBar.showMessage("Pan Off")

    ###############################################################################################################################
    # BUTTONS FUNCTIONS FOR HOTKEYS

    def OnSliderScroll(self, event):
        """
        Adjust marker size for plotting the annotations
        """
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
        self.buttonCounter = []
        self.markerSize = self.slider.value()
        img_name = Path(self.index[self.iter]).name
        self.figure.delaxes(self.figure.axes[1])
        self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(
            self.img,
            img_name,
            self.iter,
            self.index,
            self.bodyparts,
            self.colormap,
            keep_view=True,
            dir=self.dir,
        )

        self.axes.callbacks.connect("xlim_changed", self.onZoom)
        self.axes.callbacks.connect("ylim_changed", self.onZoom)
        self.buttonCounter = MainFrame.plot(self, self.img)

    def activateSlider(self):
        """
        Activates the slider to increase the markersize
        """

        if self.checkBox.isChecked():
            self.activate_slider = True
            self.slider.setEnabled(True)
            MainFrame.updateZoomPan(self)
        else:
            self.slider.setEnabled(False)

    def onButtonRelease(self, event):
        if self.pan.isChecked():
            self.updateZoomPan()
            self.statusBar.showMessage("Pan Off")

    def onClick(self, event):
        """
        This function adds labels and auto advances to the next label.
        """
        x1 = event.xdata
        y1 = event.ydata

        rbn_id = abs(self.rdb.checkedId()) - 2
        if event.button == 3:
            if rbn_id in self.buttonCounter:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText(
                    "%s is already annotated. \nSelect another body part to annotate."
                    % (str(self.bodyparts[rbn_id]))
                )
                msg.setWindowTitle("Error")
                msg.setMinimumWidth(300)
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()

            else:
                color = self.colormap(self.norm(self.colorIndex[rbn_id]))
                circle = [
                    patches.Circle(
                        (x1, y1), radius=self.markerSize, fc=color, alpha=self.alpha
                    )
                ]
                self.num.append(circle)
                self.axes.add_patch(circle[0])
                self.dr = auxfun_drag.DraggablePoint(circle[0], self.bodyparts[rbn_id])
                self.dr.connect()
                self.buttonCounter.append(rbn_id)
                self.dr.coords = [
                    [
                        x1,
                        y1,
                        self.bodyparts[rbn_id],
                        self.rdb.checkedId(),
                    ]
                ]
                self.drs.append(self.dr)
                self.updatedCoords.append(self.dr.coords)
                if rbn_id < len(self.bodyparts) - 1:
                    # self.rdb.setId(self.rdb.button(self.rdb.checkedId()),self.rdb.checkedId() - 1)
                    self.rdb.button(self.rdb.checkedId() - 1).setChecked(True)
                self.figure.canvas.draw()

        self.canvas.mpl_disconnect(self.onClick)
        self.canvas.mpl_disconnect(self.onButtonRelease)

    def browseDir(self):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """
        self.statusBar.showMessage("Looking for a folder to start labeling...")

        path = self.config_file.split("/")[0:-1]
        path = "\\".join(path)

        cwd = os.path.join(path, "labeled-data")
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the directory where your extracted frames are saved:", cwd
        )
        if not dirname:
            return
        dirname = QtCore.QDir.toNativeSeparators(dirname)
        self.dir = dirname.replace("\\", "/")

        self.load.setEnabled(False)
        self.next.setEnabled(True)
        self.save.setEnabled(True)

        # Enabling the zoom, pan and home buttons
        self.zoom.setEnabled(True)
        self.home.setEnabled(True)
        self.pan.setEnabled(True)
        self.lock.setEnabled(True)

        # Reading config file and its variables
        self.cfg = auxiliaryfunctions.read_config(self.config_file)
        self.scorer = self.cfg["scorer"]
        self.bodyparts = self.cfg["bodyparts"]
        self.videos = self.cfg["video_sets"].keys()
        self.markerSize = self.cfg["dotsize"]
        self.alpha = self.cfg["alphavalue"]
        self.colormap = plt.get_cmap(self.cfg["colormap"])
        self.colormap = self.colormap.reversed()
        self.project_path = self.cfg["project_path"]

        imlist = []

        for imtype in self.imtypes:
            imlist.extend(
                [
                    fn.replace("\\", "/")
                    for fn in glob.glob(self.dir + "/" + imtype)
                    if ("labeled.png" not in fn)
                ]
            )
        if len(imlist) == 0:
            print("No images found!!")

        self.index = np.sort(imlist)
        self.statusBar.showMessage(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        self.relativeimagenames = [
            "labeled" + n.split("labeled")[1] for n in self.index
        ]  # [n.split(self.project_path+'/')[1] for n in self.index]

        # Reading the existing dataset,if already present
        try:
            self.dataFrame = pd.read_hdf(
                os.path.join(self.dir, "CollectedData_" + self.scorer + ".h5")
            )
            self.dataFrame.sort_index(inplace=True)
            self.prev.setEnabled(True)

            # Finds the first empty row in the dataframe and sets the iteration to that index
            for idx, j in enumerate(self.dataFrame.index):
                values = self.dataFrame.loc[j, :].values
                if np.prod(np.isnan(values)) == 1:
                    self.iter = idx
                    break
                else:
                    self.iter = 0

        except:
            a = np.empty((len(self.index), 2))
            a[:] = np.nan
            for bodypart in self.bodyparts:
                index = pd.MultiIndex.from_product(
                    [[self.scorer], [bodypart], ["x", "y"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                frame = pd.DataFrame(a, columns=index, index=self.relativeimagenames)
                self.dataFrame = pd.concat([self.dataFrame, frame], axis=1)
            self.iter = 0

        # Reading the image name
        self.img = self.dataFrame.index[self.iter]
        self.img = self.img.replace("\\", "/")

        img_name = Path(self.img).name
        self.norm, self.colorIndex = self.image_panel.getColorIndices(
            self.img, self.bodyparts, self.dir
        )

        # Checking for new frames and adding them to the existing dataframe
        old_imgs = np.sort(list(self.dataFrame.index))
        self.newimages = list(set(self.relativeimagenames) - set(old_imgs))

        if not self.newimages:
            pass
        else:
            print("Found new frames..")
            # Create an empty dataframe with all the new images and then merge this to the existing dataframe.
            self.df = None
            a = np.empty((len(self.newimages), 2))
            a[:] = np.nan

            for bodypart in self.bodyparts:
                index = pd.MultiIndex.from_product(
                    [[self.scorer], [bodypart], ["x", "y"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                frame = pd.DataFrame(a, columns=index, index=self.newimages)
                self.df = pd.concat([self.df, frame], axis=1)
            self.dataFrame = pd.concat([self.dataFrame, self.df], axis=0)
            # Sort it by the index values
            self.dataFrame.sort_index(inplace=True)

        # checks for unique bodyparts
        if len(self.bodyparts) != len(set(self.bodyparts)):
            print(
                "Error - bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again. Quitting for now!"
            )
            self.close()

        # Extracting the list of new labels
        oldBodyParts = self.dataFrame.columns.get_level_values(1)
        _, idx = np.unique(oldBodyParts, return_index=True)
        oldbodyparts2plot = list(oldBodyParts[np.sort(idx)])
        self.new_bodyparts = [x for x in self.bodyparts if x not in oldbodyparts2plot]

        # Checking if user added a new label

        if not self.new_bodyparts:  # i.e. no new label
            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.bodyparts,
                self.colormap,
                dir=self.dir,
            )

            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)

            (
                self.choiceBox,
                self.rdb,
                self.slider,
                self.checkBox,
            ) = self.choice_panel.addRadioButtons(
                self.bodyparts, self.file, self.markerSize
            )

            self.buttonCounter = MainFrame.plot(self, self.img)
            self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)
            self.canvas.mpl_connect("button_release_event", self.onButtonRelease)

        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setText(
                "New label found in the config file. \nDo you want to see all the other labels?"
            )
            msg.setWindowTitle("New label found")
            msg.setMinimumWidth(300)
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            result = msg.exec_()

            if result == QMessageBox.No:
                self.bodyparts = self.new_bodyparts
                self.norm, self.colorIndex = self.image_panel.getColorIndices(
                    self.img, self.bodyparts, self.dir
                )
            a = np.empty((len(self.index), 2))
            a[:] = np.nan
            for bodypart in self.new_bodyparts:
                index = pd.MultiIndex.from_product(
                    [[self.scorer], [bodypart], ["x", "y"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                frame = pd.DataFrame(a, columns=index, index=self.relativeimagenames)
                self.dataFrame = pd.concat([self.dataFrame, frame], axis=1)

            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.bodyparts,
                self.colormap,
                dir=self.dir,
            )
            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)

            (
                self.choiceBox,
                self.rdb,
                self.slider,
                self.checkBox,
            ) = self.choice_panel.addRadioButtons(
                self.bodyparts, self.file, self.markerSize
            )
            self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)
            self.canvas.mpl_connect("button_release_event", self.onButtonRelease)
            self.buttonCounter = MainFrame.plot(self, self.img)

        self.checkBox.stateChanged.connect(self.activateSlider)
        self.slider.valueChanged.connect(self.OnSliderScroll)

    def quitButton(self, event):
        """
        Asks user for its inputs and then quits the GUI
        """
        self.statusBar.showMessage("Qutting now!")

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Do you want to label another data set?")
        msg.setWindowTitle("Repeat?")
        msg.setMinimumWidth(300)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        result = msg.exec_()

        if result == QMessageBox.Yes:
            self.file = 1
            self.buttonCounter = []
            self.updatedCoords = []
            self.dataFrame = None
            self.bodyparts = []
            self.new_labels = self.new_labels
            if self.axes != None:
                self.axes.clear()
                self.figure.delaxes(self.figure.axes[1])

            # self.choiceBox.Clear(True) #TODO: find the same function in pyqt5
            MainFrame.updateZoomPan(self)
            MainFrame.browseDir(self)
            self.save.setEnabled(True)

        else:
            self.destroy()
            print(
                "You can now check the labels, using 'check_labels' before proceeding. Then, you can use the function 'create_training_dataset' to create the training dataset."
            )

    def helpButton(self, event):
        """
        Opens Instructions
        """
        MainFrame.updateZoomPan(self)
        self.statusBar.showMessage("Help")
        self.help_info = QtWidgets.QDialog()
        self.help_info.adjustSize()
        info = QLabel(
            "1. Select an individual and one of the body parts from the radio buttons to add a label \n (if necessary change config.yaml first to edit the label names). \n\n2. Right clicking on the image will add the selected label and the next available label will be selected from the radio button. \n The label will be marked as circle filled with a unique color (and individual ID a unique color on the rim).\n\n3. To change the marker size, mark the checkbox and move the slider, then uncheck the box. \n\n4. Hover your mouse over this newly added label to see its name. \n\n5. Use left click and drag to move the label position.  \n\n6. Once you are happy with the position, right click to add the next available label. \n You can always reposition the old labels, if required. \n You can delete a label with the middle button mouse click (or click 'delete' key). \n\n7. Click Next/Previous to move to the next/previous image (or hot-key arrows left and right).\n User can also re-label a delete point by going to a previous/next image then returning to the current image. \n NOTE: the user cannot add a label if the label is already present. \n \n8. You can click Cntrl+C to copy+paste labels from a previous image into the current image. \n\n9. When finished labeling all the images, click 'Save' to save all the labels as a .h5 file. \n\n10. Click OK to continue using the labeling GUI. For more tips and hotkeys: see docs!!"
        )
        self.help_info.setWindowTitle("User instructions")

        ok_btn = QPushButton("Ok")
        ok_btn.clicked.connect(self.close_help)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        layout.addWidget(info)
        layout.addWidget(ok_btn, alignment=Qt.AlignRight | Qt.AlignBottom)
        self.help_info.setLayout(layout)

        self.help_info.exec()

    def close_help(self):
        self.help_info.close()

    def zoomButton(self, event):
        if self.zoom.isChecked():
            # Save pre-zoom xlim and ylim values
            self.prezoom_xlim = self.axes.get_xlim()
            self.prezoom_ylim = self.axes.get_ylim()
            self.toolbar.zoom()
            self.statusBar.showMessage("Zoom On")
            self.pan.setChecked(False)
        else:
            self.toolbar.zoom()
            self.statusBar.showMessage("Zoom Off")

    def homeButton(self, event):
        self.image_panel.resetView()
        self.figure.canvas.draw()
        self.updateZoomPan(self)
        self.zoom.setCheckable(False)
        self.pan.setCheckable(False)
        self.statusBar.showMessage("")

    def saveDataSet(self, event):
        """
        Saves the final dataframe
        """
        self.statusBar.showMessage("File saved")
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)

        # Windows compatible
        self.dataFrame.sort_index(inplace=True)
        self.dataFrame = self.dataFrame.reindex(
            self.cfg["bodyparts"],
            axis=1,
            level=self.dataFrame.columns.names.index("bodyparts"),
        )
        self.dataFrame.to_csv(
            os.path.join(self.dir, "CollectedData_" + self.scorer + ".csv")
        )
        self.dataFrame.to_hdf(
            os.path.join(self.dir, "CollectedData_" + self.scorer + ".h5"),
            "df_with_missing",
            format="table",
            mode="w",
        )

    def nextImage(self, event):
        """
        Moves to next image
        """
        #  Checks for the last image and disables the Next button
        if len(self.index) - self.iter == 1:
            self.next.setEnabled(False)
            return
        self.prev.setEnabled(True)

        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)

        self.statusBar.showMessage(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )

        self.rdb.setId(self.rdb.button(-2), -2)
        self.rdb.button(-2).setChecked(True)

        self.file = 1
        # Refreshing the button counter
        self.buttonCounter = []

        MainFrame.saveEachImage(self)
        self.iter = self.iter + 1

        if len(self.index) >= self.iter:

            self.updatedCoords = MainFrame.getLabels(self, self.iter)
            self.img = self.index[self.iter]
            img_name = Path(self.index[self.iter]).name
            self.figure.delaxes(
                self.figure.axes[1]
            )  # Removes the axes corresponding to the colorbar
            (
                self.figure,
                self.axes,
                self.canvas,
                self.toolbar,
            ) = self.image_panel.drawplot(
                self.img,
                img_name,
                self.iter,
                self.index,
                self.bodyparts,
                self.colormap,
                keep_view=self.view_locked,
                dir=self.dir,
            )
            self.axes.callbacks.connect("xlim_changed", self.onZoom)
            self.axes.callbacks.connect("ylim_changed", self.onZoom)

            self.buttonCounter = MainFrame.plot(self, self.img)
            self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)
            self.canvas.mpl_connect("button_release_event", self.onButtonRelease)

    def prevImage(self, event):
        """
        Checks the previous Image and enables user to move the annotations.
        """
        # Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.prev.setEnabled(False)
            return
        else:
            self.next.setEnabled(True)
        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        self.statusBar.showMessage(
            "Working on folder: {}".format(os.path.split(str(self.dir))[-1])
        )
        MainFrame.saveEachImage(self)

        self.buttonCounter = []
        self.iter = self.iter - 1

        self.rdb.setId(self.rdb.button(-2), -2)
        self.rdb.button(-2).setChecked(True)

        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name
        self.figure.delaxes(
            self.figure.axes[1]
        )  # Removes the axes corresponding to the colorbar
        self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(
            self.img,
            img_name,
            self.iter,
            self.index,
            self.bodyparts,
            self.colormap,
            keep_view=self.view_locked,
            dir=self.dir,
        )
        self.axes.callbacks.connect("xlim_changed", self.onZoom)
        self.axes.callbacks.connect("ylim_changed", self.onZoom)

        self.buttonCounter = MainFrame.plot(self, self.img)
        self.cidClick = self.canvas.mpl_connect("button_press_event", self.onClick)
        self.canvas.mpl_connect("button_release_event", self.onButtonRelease)
        MainFrame.saveEachImage(self)

    def getLabels(self, img_index):
        """
        Returns a list of x and y labels of the corresponding image index
        """
        self.previous_image_points = []

        for bpindex, bp in enumerate(self.bodyparts):
            image_points = [
                [
                    self.dataFrame[self.scorer][bp]["x"].values[self.iter],
                    self.dataFrame[self.scorer][bp]["y"].values[self.iter],
                    bp,
                    bpindex,
                ]
            ]
            self.previous_image_points.append(image_points)
        return self.previous_image_points

    def saveEachImage(self):
        """
        Saves data for each image
        """

        for idx, bp in enumerate(self.updatedCoords):
            self.dataFrame.loc[self.relativeimagenames[self.iter]][
                self.scorer, bp[0][-2], "x"
            ] = bp[-1][0]
            self.dataFrame.loc[self.relativeimagenames[self.iter]][
                self.scorer, bp[0][-2], "y"
            ] = bp[-1][1]

    def plot(self, img):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """
        self.drs = []
        self.updatedCoords = []
        for bpindex, bp in enumerate(self.bodyparts):
            color = self.colormap(self.norm(self.colorIndex[bpindex]))
            self.points = [
                self.dataFrame[self.scorer][bp]["x"].values[self.iter],
                self.dataFrame[self.scorer][bp]["y"].values[self.iter],
            ]
            circle = [
                patches.Circle(
                    (self.points[0], self.points[1]),
                    radius=self.markerSize,
                    fc=color,
                    alpha=self.alpha,
                )
            ]
            self.axes.add_patch(circle[0])
            self.dr = auxfun_drag.DraggablePoint(circle[0], self.bodyparts[bpindex])
            self.dr.connect()
            self.dr.coords = MainFrame.getLabels(self, self.iter)[bpindex]
            self.drs.append(self.dr)
            self.updatedCoords.append(self.dr.coords)
            if not np.isnan(self.points)[0]:
                self.buttonCounter.append(bpindex)
        self.figure.canvas.draw()

        return self.buttonCounter


def show(self, config, config3d, sourceCam, imtypes=["*.png"]):

    frame = MainFrame(self, config, imtypes, config3d, sourceCam)
    frame.show()
