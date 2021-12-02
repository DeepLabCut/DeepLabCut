import datetime
import os
import pydoc
import sys

from PyQt5.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup, QDoubleSpinBox
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import deeplabcut

class Video_editor_page(QWidget):

    def __init__(self, parent, cfg):
        super(Video_editor_page, self).__init__(parent)

        self.method = "automatic"
        self.config = cfg

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

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Optional Video Editor")
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
        choose_video_text = QtWidgets.QLabel("Choose the video")
        choose_video_text.setContentsMargins(0, 0, 60, 0)

        self.select_video_button = QtWidgets.QPushButton('Select video')
        self.select_video_button.setMaximumWidth(150)
        self.select_video_button.clicked.connect(self.select_video)

        layout_choose_video.addWidget(choose_video_text)
        layout_choose_video.addWidget(self.select_video_button)

        self.inLayout.addLayout(layout_cfg)
        self.inLayout.addLayout(layout_choose_video)

        self.layout_attributes = QtWidgets.QVBoxLayout()
        self.layout_attributes.setAlignment(Qt.AlignTop)
        self.layout_attributes.setSpacing(20)
        self.layout_attributes.setContentsMargins(0, 0, 40, 0)

        label = QtWidgets.QLabel('Attributes')
        label.setContentsMargins(20, 20, 0, 10)
        self.layout_attributes.addWidget(label)

        self.layout_downsample = QtWidgets.QHBoxLayout()
        self.layout_downsample.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_downsample.setSpacing(200)
        self.layout_downsample.setContentsMargins(20, 0, 50, 20)

        self._layout_downsample()
        self._layout_rotate_video()

        self.layout_shorten = QtWidgets.QHBoxLayout()
        self.layout_shorten.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_shorten.setSpacing(20)
        self.layout_shorten.setContentsMargins(20, 0, 50, 0)

        self._start_time()
        self._stop_time()
        self._angle()

        self.layout_attributes.addLayout(self.layout_downsample)
        self.layout_attributes.addLayout(self.layout_shorten)

        self.crop_button = QtWidgets.QPushButton('CROP')
        self.crop_button.setContentsMargins(0, 40, 40, 40)
        #self.crop_button.clicked.connect(self.)

        self.layout_attributes.addWidget(self.crop_button, alignment=Qt.AlignRight)

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
        dlg = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video to modify", cwd, "", "*.*"
        )
        if dlg:
            self.vids = dlg[0]
            self.filelist = self.filelist + self.vids  # [0]
            self.select_video_button.setText("Total %s Videos selected"% len(self.filelist))

    def _layout_downsample(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Downsample - specify the video height (aspect ratio fixed)")
        self.video_height = QSpinBox()
        self.video_height.setMaximum(1000)

        self.video_height.setValue(256)
        self.video_height.setMinimumWidth(400)
        self.video_height.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.video_height)
        self.layout_downsample.addLayout(l_opt)

    def _layout_rotate_video(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Downsample: rotate video?")
        self.btngroup_rotate_video_choice = QButtonGroup()

        self.rotate_video_choice1 = QtWidgets.QRadioButton('Yes')
        #self.rotate_video_choice1.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice1))

        self.rotate_video_choice2 = QtWidgets.QRadioButton('No')
        self.rotate_video_choice2.setChecked(True)
        #self.rotate_video_choice2.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice2))

        self.rotate_video_choice3 = QtWidgets.QRadioButton('Arbitrary')
        #self.rotate_video_choice3.toggled.connect(lambda: self.update_rotate_video_choice(self.rotate_video_choice3))

        self.btngroup_rotate_video_choice.addButton(self.rotate_video_choice1)
        self.btngroup_rotate_video_choice.addButton(self.rotate_video_choice2)
        self.btngroup_rotate_video_choice.addButton(self.rotate_video_choice3)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.rotate_video_choice1)
        l_opt.addWidget(self.rotate_video_choice2)
        l_opt.addWidget(self.rotate_video_choice3)
        self.layout_downsample.addLayout(l_opt)

    def _start_time(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Shorten: start time (sec)")
        self.video_start = QSpinBox()
        self.video_start.setMaximum(3600)

        self.video_start.setValue(1)
        self.video_start.setMinimumWidth(400)
        self.video_start.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.video_start)
        self.layout_shorten.addLayout(l_opt)

    def _stop_time(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Shorten: stop time (sec)")
        self.video_stop = QSpinBox()
        self.video_stop.setMaximum(3600)
        self.video_stop.setMinimum(1)

        self.video_stop.setValue(30)
        self.video_stop.setMinimumWidth(400)
        self.video_stop.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.video_stop)
        self.layout_shorten.addLayout(l_opt)
    def _angle(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Angle for arbitrary rotation (deg)")
        self.angle = QDoubleSpinBox()
        self.angle.setMaximum(360.0)
        self.angle.setMinimum(-360.0)
        self.angle.setDecimals(2)

        self.angle.setValue(0.0)
        self.angle.setMinimumWidth(400)
        self.angle.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.angle)
        self.layout_shorten.addLayout(l_opt)





