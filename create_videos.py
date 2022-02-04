import os
import pydoc
import sys


import deeplabcut
from deeplabcut.utils import auxiliaryfunctions, skeleton

from PyQt5.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup, QDoubleSpinBox
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class Create_videos_page(QWidget):

    def __init__(self, parent, cfg):
        super(Create_videos_page, self).__init__(parent)

        self.filelist = []
        self.config = cfg
        self.bodyparts = []
        self.draw = False
        self.slow = False

        self.inLayout = QtWidgets.QVBoxLayout(self)
        self.inLayout.setAlignment(Qt.AlignTop)
        self.inLayout.setSpacing(20)
        self.inLayout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(self.inLayout)

        self.set_page()

    def set_page(self):
        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Create Labeled Videos")
        l1_step1.setContentsMargins(20, 0, 0, 10)

        self.inLayout.addWidget(l1_step1)
        self.inLayout.addWidget(separatorLine)

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

        layout_choose_video = QtWidgets.QHBoxLayout()
        layout_choose_video.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout_choose_video.setSpacing(70)
        layout_choose_video.setContentsMargins(20, 10, 300, 0)
        choose_video_text = QtWidgets.QLabel("Choose the videos")
        choose_video_text.setContentsMargins(0, 0, 52, 0)

        self.select_video_button = QtWidgets.QPushButton('Select videos')
        self.select_video_button.setMaximumWidth(350)
        self.select_video_button.clicked.connect(self.select_video)

        layout_choose_video.addWidget(choose_video_text)
        layout_choose_video.addWidget(self.select_video_button)

        self.inLayout.addLayout(layout_cfg)
        self.inLayout.addLayout(layout_choose_video)

        self.layout_attributes = QtWidgets.QVBoxLayout()
        self.layout_attributes.setAlignment(Qt.AlignTop)
        self.layout_attributes.setSpacing(20)
        self.layout_attributes.setContentsMargins(0, 0, 40, 0)

        label = QtWidgets.QLabel('Additional Attributes')
        label.setContentsMargins(20, 20, 0, 10)
        self.layout_attributes.addWidget(label)

        self.layout_specify = QtWidgets.QHBoxLayout()
        self.layout_specify.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_specify.setSpacing(30)
        self.layout_specify.setContentsMargins(20, 0, 50, 20)

        self._layout_videotype()
        self._layout_shuffle()
        self._layout_trainingset()

        self.layout_attributes.addLayout(self.layout_specify)

        self.cfg = auxiliaryfunctions.read_config(self.config)
        if self.cfg.get("multianimalproject", False):
            print("multianimalproject")

        self.layout_include_specify = QtWidgets.QHBoxLayout()
        self.layout_include_specify.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_include_specify.setSpacing(40)
        self.layout_include_specify.setContentsMargins(20, 0, 50, 0)

        self._draw_skeleton()
        self._trail_points()
        self._video_slow()
        self.layout_attributes.addLayout(self.layout_include_specify)

        self.layout_filter = QtWidgets.QHBoxLayout()
        self.layout_filter.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_filter.setSpacing(40)
        self.layout_filter.setContentsMargins(20, 0, 50, 0)
        self._filter()
        self.layout_attributes.addLayout(self.layout_filter)

        self.layout_plot_bp = QtWidgets.QHBoxLayout()
        self.layout_plot_bp.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_plot_bp.setSpacing(40)
        self.layout_plot_bp.setContentsMargins(20, 0, 50, 0)
        self._plot_bp()
        self.layout_attributes.addLayout(self.layout_plot_bp)

        self.run_button = QtWidgets.QPushButton('RUN')
        self.run_button.setContentsMargins(0, 40, 40, 40)
        # self.step_button.clicked.connect(self.)

        self.layout_attributes.addWidget(self.run_button, alignment=Qt.AlignRight)
        self.inLayout.addLayout(self.layout_attributes)

    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text

    def browse_dir(self):
        print('browse_dir')
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config:
            return
        self.config = config

    def select_video(self):
        print('select_video')
        cwd = os.getcwd()
        videos_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video to modify", cwd, "", "*.*"
        )
        if videos_file:
            self.vids = videos_file[0]
            self.filelist.append(self.vids)
            self.select_video_button.setText("Total %s Videos selected" % len(self.filelist))
            self.select_video_button.adjustSize()

    def _layout_videotype(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Select the network")
        self.videotype = QComboBox()
        self.videotype.setMinimumWidth(350)
        self.videotype.setMinimumHeight(30)
        options = [".avi", ".mp4", ".mov"]
        self.videotype.addItems(options)
        self.videotype.setCurrentText(".avi")

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.videotype)
        self.layout_specify.addLayout(l_opt)

    def _layout_shuffle(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the shuffle")
        self.shuffles = QSpinBox()
        self.shuffles.setMaximum(100)
        self.shuffles.setValue(1)
        self.shuffles.setMinimumWidth(400)
        self.shuffles.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.shuffles)
        self.layout_specify.addLayout(l_opt)

    def _layout_trainingset(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the trainingset index")
        self.trainingset = QSpinBox()
        self.trainingset.setMaximum(100)
        self.trainingset.setValue(0)
        self.trainingset.setMinimumWidth(400)
        self.trainingset.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.trainingset)
        self.layout_specify.addLayout(l_opt)

    def _draw_skeleton(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Include the skeleton in the video?")
        self.btngroup_draw_skeleton_choice = QButtonGroup()

        self.draw_skeleton_choice1 = QtWidgets.QRadioButton('Yes')
        # self.rotate_video_choice1.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice1))

        self.draw_skeleton_choice2 = QtWidgets.QRadioButton('No')
        self.draw_skeleton_choice2.setChecked(True)
        # self.rotate_video_choice2.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice2))

        self.btngroup_draw_skeleton_choice.addButton(self.draw_skeleton_choice1)
        self.btngroup_draw_skeleton_choice.addButton(self.draw_skeleton_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.draw_skeleton_choice1)
        l_opt.addWidget(self.draw_skeleton_choice2)
        self.layout_include_specify.addLayout(l_opt)

    def _trail_points(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(122, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the number of trail points")
        self.shuffles = QSpinBox()
        self.shuffles.setValue(0)
        self.shuffles.setMinimumWidth(400)
        self.shuffles.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.shuffles)
        self.layout_include_specify.addLayout(l_opt)

    def _video_slow(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(30, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Create a higher quality video? (slow)")
        self.btngroup_video_slow_choice = QButtonGroup()

        self.video_slow_choice1 = QtWidgets.QRadioButton('Yes')
        # self.rotate_video_choice1.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice1))

        self.video_slow_choice2 = QtWidgets.QRadioButton('No')
        self.video_slow_choice2.setChecked(True)
        # self.rotate_video_choice2.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice2))

        self.btngroup_video_slow_choice.addButton(self.video_slow_choice1)
        self.btngroup_video_slow_choice.addButton(self.video_slow_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.video_slow_choice1)
        l_opt.addWidget(self.video_slow_choice2)
        self.layout_include_specify.addLayout(l_opt)

    def _filter(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Use filtered predictions?")
        self.btngroup_filter_choice = QButtonGroup()

        self.filter_choice1 = QtWidgets.QRadioButton('Yes')
        # self.rotate_video_choice1.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice1))

        self.filter_choice2 = QtWidgets.QRadioButton('No')
        self.filter_choice2.setChecked(True)
        # self.rotate_video_choice2.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice2))

        self.btngroup_filter_choice.addButton(self.filter_choice1)
        self.btngroup_filter_choice.addButton(self.filter_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.filter_choice1)
        l_opt.addWidget(self.filter_choice2)
        self.layout_filter.addLayout(l_opt)

    def _plot_bp(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(40, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Plot all bodyparts?")
        self.btngroup_bodypart_choice = QButtonGroup()

        self.bodypart_choice1 = QtWidgets.QRadioButton('Yes')
        self.bodypart_choice1.setChecked(True)
        # self.rotate_video_choice1.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice1))

        self.bodypart_choice2 = QtWidgets.QRadioButton('No')
        self.bodypart_choice2.setChecked(True)
        # self.rotate_video_choice2.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice2))

        self.btngroup_bodypart_choice.addButton(self.bodypart_choice1)
        self.btngroup_bodypart_choice.addButton(self.bodypart_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.bodypart_choice1)
        l_opt.addWidget(self.bodypart_choice2)
        self.layout_filter.addLayout(l_opt)





