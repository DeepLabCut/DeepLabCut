import time

from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from deeplabcut.gui.components import (
    DefaultTab,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.gui.widgets import FrameCropper
from deeplabcut.utils import auxfun_videos


class VideoEditor(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(VideoEditor, self).__init__(root, parent, h1_description)

        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):

        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.trim_button = QtWidgets.QPushButton("Trim")
        self.trim_button.setMinimumWidth(150)
        self.trim_button.clicked.connect(self.trim_videos)
        self.main_layout.addWidget(self.trim_button, alignment=Qt.AlignRight)

        self.down_button = QtWidgets.QPushButton("Downsample")
        self.down_button.setMinimumWidth(150)
        self.down_button.clicked.connect(self.downsample_videos)
        self.main_layout.addWidget(self.down_button, alignment=Qt.AlignRight)

        self.crop_button = QtWidgets.QPushButton("Crop")
        self.crop_button.setMinimumWidth(150)
        self.crop_button.clicked.connect(self.crop_videos)
        self.main_layout.addWidget(self.crop_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):

        videoheight_label = QtWidgets.QLabel("Video height (aspect ratio fixed)")
        self.video_height = QtWidgets.QSpinBox()
        self.video_height.setMaximum(1000)
        self.video_height.setValue(256)
        self.video_height.valueChanged.connect(self.log_video_height)

        rotate_label = QtWidgets.QLabel("Rotate video")
        self.video_rotation = QtWidgets.QComboBox()
        self.video_rotation.addItems(["no", "clockwise", "specific angle"])
        self.video_rotation.currentTextChanged.connect(self.update_video_rotation)

        trim_start_label = QtWidgets.QLabel("Trim start (sec)")
        self.video_start = QtWidgets.QSpinBox()
        self.video_start.setMaximum(3600)
        self.video_start.setValue(1)
        self.video_start.valueChanged.connect(self.log_video_start)

        trim_end_label = QtWidgets.QLabel("Trim end (sec)")
        self.video_stop = QtWidgets.QSpinBox()
        self.video_stop.setMaximum(3600)
        self.video_stop.setMinimum(1)
        self.video_stop.setValue(30)
        self.video_stop.valueChanged.connect(self.log_video_stop)

        angle_label = QtWidgets.QLabel("Rotation angle (deg)")
        self.rotation_angle = QtWidgets.QDoubleSpinBox()
        self.rotation_angle.setMaximum(360.0)
        self.rotation_angle.setMinimum(-360.0)
        self.rotation_angle.setDecimals(2)
        self.rotation_angle.setValue(0.0)
        self.rotation_angle.setEnabled(False)
        self.rotation_angle.valueChanged.connect(self.log_rotation_angle)

        downsample_title = QtWidgets.QLabel("Downsample and rotate:")
        trim_title = QtWidgets.QLabel("Shorten video (trim):")

        layout.addWidget(downsample_title, 0, 0)
        layout.addWidget(trim_title, 0, 4)
        layout.addWidget(videoheight_label, 1, 0)
        layout.addWidget(self.video_height, 1, 1)
        layout.addWidget(rotate_label, 1, 2)
        layout.addWidget(self.video_rotation, 1, 3)
        layout.addWidget(angle_label, 2, 2)
        layout.addWidget(self.rotation_angle, 2, 3)
        layout.addWidget(trim_start_label, 1, 4)
        layout.addWidget(self.video_start, 1, 5)
        layout.addWidget(trim_end_label, 2, 4)
        layout.addWidget(self.video_stop, 2, 5)
        # layout.addWidget()

    def update_video_rotation(self, option):
        self.root.logger.info(f"Video rotation set to {option.upper()}")
        if option == "specific angle":
            self.rotation_angle.setEnabled(True)
        else:
            self.rotation_angle.setEnabled(False)

    def log_video_height(self, value):
        self.root.logger.info(f"Video height set to {value}")

    def log_video_start(self, value):
        start_time = time.strftime("%H:%M:%S", time.gmtime(value))
        self.root.logger.info(f"Video start time set to {start_time}")

    def log_video_stop(self, value):
        stop_time = time.strftime("%H:%M:%S", time.gmtime(value))
        self.root.logger.info(f"Video start time set to {stop_time}")

    def log_rotation_angle(self, value):
        self.root.logger.info(f"Rotation angle set to {value}")

    def trim_videos(self):
        start = time.strftime("%H:%M:%S", time.gmtime(self.video_start.value()))
        stop = time.strftime("%H:%M:%S", time.gmtime(self.video_stop.value()))
        if self.files:
            for video in self.files:
                auxfun_videos.ShortenVideo(video, start, stop)
        else:
            self.root.logger.error("No videos selected...")

    def downsample_videos(self):
        if self.files:
            for video in self.files:
                auxfun_videos.DownSampleVideo(
                    video,
                    width=-1,
                    height=self.video_height.value(),
                    rotatecw=self.video_rotation.currentData(),
                    angle=self.rotation_angle.value(),
                )
        else:
            self.root.logger.error("No videos selected...")

    def crop_videos(self):
        if self.files:
            for video in self.files:
                _ = _crop_video(video)
        else:
            self.root.logger.error("No videos selected...")


def _crop_video(video_path):
    fc = FrameCropper(video_path)
    coords = fc.draw_bbox()
    if not coords:
        return
    origin_x, origin_y = coords[:2]
    width = int(coords[2]) - int(coords[0])
    height = int(coords[3]) - int(coords[1])
    writer = auxfun_videos.VideoWriter(video_path)
    writer.set_bbox(origin_x, origin_x + width, origin_y, origin_y + height)
    return writer.crop("cropped", None)
