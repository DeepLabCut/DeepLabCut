import os

from PySide2.QtWidgets import QWidget, QMessageBox
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon

from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path


def refine_labels(config, multianimal=False):
    """
    Refines the labels of the outlier frames extracted from the analyzed videos.\n Helps in augmenting the training dataset.
    Use the function ``analyze_video`` to analyze a video and extracts the outlier frames using the function
    ``extract_outlier_frames`` before refining the labels.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Screens : int value of the number of Screens in landscape mode, i.e. if you have 2 screens, enter 2. Default is 1.

    scale_h & scale_w : you can modify how much of the screen the GUI should occupy. The default is .9 and .8, respectively.

    img_scale : if you want to make the plot of the frame larger, consider changing this to .008 or more. Be careful though, too large and you will not see the buttons fully!

    Examples
    --------
    >>> deeplabcut.refine_labels('/analysis/project/reaching-task/config.yaml', Screens=2, imag_scale=.0075)
    --------

    """

    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if not multianimal and not cfg.get("multianimalproject", False):
        from deeplabcut.gui import refinement

        refinement.show(config)
    else:  # loading multianimal labeling GUI
        from deeplabcut.gui import multiple_individuals_refinement_toolbox

        multiple_individuals_refinement_toolbox.show(config)

    os.chdir(startpath)


class RefineLabels(QWidget):
    def __init__(self, parent, cfg):
        super(RefineLabels, self).__init__(parent)
        # variable initilization
        self.method = "automatic"
        self.config = cfg

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(0, 20, 40, 20)
        self.setLayout(self.main_layout)

        self.set_page()

    def set_page(self):
        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 9. Refine labels")
        l1_step1.setContentsMargins(20, 0, 0, 10)

        self.main_layout.addWidget(l1_step1)
        self.main_layout.addWidget(separatorLine)

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

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_dir)

        layout_cfg.addWidget(cfg_text)
        layout_cfg.addWidget(self.cfg_line)
        layout_cfg.addWidget(browse_button)

        self.main_layout.addLayout(layout_cfg)

        self.launch_button = QtWidgets.QPushButton("LAUNCH")
        self.launch_button.setContentsMargins(0, 140, 100, 40)
        self.launch_button.clicked.connect(self.refine_labels)

        self.merge_button = QtWidgets.QPushButton("Merge dataset")
        self.merge_button.clicked.connect(self.merge_dataset)
        self.merge_button.setEnabled(False)

        self.main_layout.addWidget(self.merge_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.launch_button, alignment=Qt.AlignRight)

    def update_cfg(self):
        text = self.cfg_line.text()
        self.config = text
        self.cfg = auxiliaryfunctions.read_config(self.config)


    def browse_dir(self):
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config[0]:
            return
        self.config = config[0]
        self.cfg_line.setText(self.config)

    def merge_dataset(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(
            "1. Make sure that you have refined all the labels before merging the dataset.\n\n2. If you merge the dataset, you need to re-create the training dataset before you start the training.\n\n3. Are you ready to merge the dataset?"
        )
        msg.setWindowTitle("Warning")
        msg.setMinimumWidth(1000)
        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        msg.setWindowIcon(QIcon(self.logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        result = msg.exec_()
        if result == QMessageBox.Yes:
            print("Yes")
            # TODO: finish -->
            # notebook = self.GetParent()
            # notebook.SetSelection(4)
            # deeplabcut.merge_datasets(self.config, forceiterate=None)

    def refine_labels(self):
        self.merge_button.setEnabled(True)
        # TODO: finish refine_labels part
        # deeplabcut.refine_labels(self.config)
