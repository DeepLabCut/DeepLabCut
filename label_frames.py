import os
import pydoc
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


from deeplabcut.generate_training_dataset import check_labels
from deeplabcut.utils import auxiliaryfunctions, skeleton
from pathlib import Path

def label_frames(
    config,
    multiple_individualsGUI=False,
    imtypes=["*.png"],
    config3d=None,
    sourceCam=None,
):
    """
    Manually label/annotate the extracted frames. Update the list of body parts you want to localize in the config.yaml file first.

    Parameter
    ----------
    config : string
        String containing the full path of the config file in the project.

    multiple_individualsGUI: bool, optional
        If this is set to True, a user can label multiple individuals. Note for "multianimalproject=True" this is automatically used.
        The default is ``False``; if provided it must be either ``True`` or ``False``.

    imtypes: list of imagetypes to look for in folder to be labeled.
        By default only png images are considered.

    config3d: string, optional
        String containing the full path of the config file in the 3D project. Include when epipolar lines would be helpful for labeling additional camera angles.

    sourceCam: string, optional
        String containing the camera name from which to pull labeling data to generate epipolar lines. This must match the pattern in 'camera_names' in the 3D config file.
        If no value is entered, data will be pulled from either cam1 or cam2

    Example
    --------
    Standard use case:
    >>> deeplabcut.label_frames('/myawesomeproject/reaching4thestars/config.yaml')

    To label multiple individuals (without having a multiple individuals project); otherwise this GUI is loaded automatically
    >>> deeplabcut.label_frames('/analysis/project/reaching-task/config.yaml',multiple_individualsGUI=True)

    To label other image types
    >>> label_frames(config,multiple=False,imtypes=['*.jpg','*.jpeg'])

    To label with epipolar lines projected from labels in another camera angle #+++
    >>> label_frames(config, config3d='/analysis/project/reaching-task/reaching-task-3d/config.yaml', sourceCam='cam1')
    --------

    """
    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if cfg.get("multianimalproject", False) or multiple_individualsGUI:
        import multiple_individuals_labeling_toolbox

        #multiple_individuals_labeling_toolbox.show(config, config3d, sourceCam)
    else:
        print('labeling_toolbox')
        import labeling_toolbox

        labeling_toolbox.show() #config, config3d, sourceCam, imtypes=imtypes

    os.chdir(startpath)


class Label_page(QWidget):

    def __init__(self, parent, cfg):
        super(Label_page, self).__init__(parent)

        # variable initilization
        self.method = "automatic"
        self.config = cfg

        self.separatorLine = QtWidgets.QFrame()
        self.separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        self.separatorLine.setLineWidth(0)
        self.separatorLine.setMidLineWidth(1)

        inLayout = QtWidgets.QVBoxLayout(self)
        inLayout.setAlignment(Qt.AlignTop)
        inLayout.setSpacing(20)
        inLayout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(inLayout)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 3. Label Frames")
        l1_step1.setContentsMargins(20, 0, 0, 10)

        inLayout.addWidget(l1_step1)
        inLayout.addWidget(self.separatorLine)

        layout_cfg = QtWidgets.QHBoxLayout()
        layout_cfg.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout_cfg.setSpacing(20)
        layout_cfg.setContentsMargins(20, 10, 300, 0)
        cfg_text = QtWidgets.QLabel("Select the config file")
        cfg_text.setContentsMargins(0, 0, 60, 0)

        self.cfg_line = QtWidgets.QLineEdit()
        self.cfg_line.setMaximumWidth(800)
        self.cfg_line.setMinimumWidth(600)
        self.cfg_line.setMinimumHeight(30)
        self.cfg_line.setText(self.config)
        self.cfg_line.textChanged[str].connect(self.update_cfg)

        browse_button = QtWidgets.QPushButton('Browse')
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_dir)

        layout_cfg.addWidget(cfg_text)
        layout_cfg.addWidget(self.cfg_line)
        layout_cfg.addWidget(browse_button)

        inLayout.addLayout(layout_cfg)

        layout_label_btns = QtWidgets.QVBoxLayout()
        layout_label_btns.setAlignment(Qt.AlignTop)
        layout_label_btns.setSpacing(40)
        layout_label_btns.setContentsMargins(0, 0, 60, 50)
        #
        self.label_frames_btn = QtWidgets.QPushButton('Label Frames')
        self.label_frames_btn.clicked.connect(self.label_frames)
        #
        self.check_labels_btn = QtWidgets.QPushButton('Check Labels!')
        self.check_labels_btn.clicked.connect(self.check_labelF)
        self.check_labels_btn.setEnabled(True)
        #
        layout_label_btns.addWidget(self.label_frames_btn, alignment=Qt.AlignRight)
        layout_label_btns.addWidget(self.check_labels_btn, alignment=Qt.AlignRight)
        #
        inLayout.addLayout(layout_label_btns)

        self.cfg = auxiliaryfunctions.read_config(self.config)
        if self.cfg.get("multianimalproject", False):
            print('F')
            # ?
        else: print('Y')

    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text
        print(text)

    def browse_dir(self):
        print('browse_dir')
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config:
            return
        self.config = config

    def check_labelF(self, event):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("This will now plot the labeled frames after you have finished labeling!")

        msg.setWindowTitle("Info")
        msg.setMinimumWidth(1000)
        self.logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
        self.logo = self.logo_dir + '/pictures/logo.png'
        msg.setWindowIcon(QIcon(self.logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        retval = msg.exec_()

        check_labels(self.config, visualizeindividuals=False)


    def label_frames(self):
        self.frame = None
        label_frames(self.config)