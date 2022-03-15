import logging

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from PySide2.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup
from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from widgets import ConfigEditor


class AnalyzeVideos(QWidget):
    def __init__(self, parent, cfg):
        super(AnalyzeVideos, self).__init__(parent)

        self.logger = logging.getLogger("GUI")

        self.filelist = []
        self.picklelist = []
        self.bodyparts = []
        self.config = cfg
        self.cfg = auxiliaryfunctions.read_config(self.config)
        self.draw = False

        self.csv = False
        self.dynamic = False
        self.trajectory = False
        self.filter = False
        self.showfigs = True

        # if self.cfg.get("multianimalproject", False):
        #     self.bodyparts = self.cfg["multianimalbodyparts"]
        # else:
        #     self.bodyparts = self.cfg["bodyparts"]

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

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 7. Analyze Videos ....")
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

        browse_button = QtWidgets.QPushButton("Browse")
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

        self.select_video_button = QtWidgets.QPushButton("Select videos to analyze")
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

        label = QtWidgets.QLabel("Attributes")
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

        if self.cfg.get("multianimalproject", False):
            print("multianimalproject")
            # TODO: finish multianimal part:
            # self.robust = wx.RadioBox(
            #                 self,
            #                 label="Use ffprobe to read video metadata (slow but robust)",
            #                 choices=["Yes", "No"],
            #                 majorDimension=1,
            #                 style=wx.RA_SPECIFY_COLS,
            #             )
            #             self.robust.SetSelection(1)
            #             self.hbox2.Add(self.robust, 1, 1)
            #
            #             self.create_video_with_all_detections = wx.RadioBox(
            #                 self,
            #                 label="Create video for checking detections",
            #                 choices=["Yes", "No"],
            #                 majorDimension=1,
            #                 style=wx.RA_SPECIFY_COLS,
            #             )
            #             self.create_video_with_all_detections.SetSelection(1)
            #             self.hbox2.Add(
            #                 self.create_video_with_all_detections,
            #                 1,
            #                 wx.EXPAND | wx.TOP | wx.BOTTOM,
            #                 1,
            #             )
            #
            #             tracker_text = wx.StaticBox(
            #                 self, label="Specify the Tracker Method (you can try each)"
            #             )
            #             tracker_text_boxsizer = wx.StaticBoxSizer(tracker_text, wx.VERTICAL)
            #             trackertypes = ["skeleton", "box", "ellipse"]
            #             self.trackertypes = wx.ComboBox(
            #                 self, choices=trackertypes, style=wx.CB_READONLY
            #             )
            #             self.trackertypes.SetValue("ellipse")
            #             tracker_text_boxsizer.Add(
            #                 self.trackertypes, 1,wx.EXPAND | wx.TOP | wx.BOTTOM,1
            #             )
            #             self.hbox3.Add(tracker_text_boxsizer, 1,  1)
            #
            #             self.overwrite = wx.RadioBox(
            #                 self,
            #                 label="Overwrite tracking files (set to yes if you edit inference parameters)",
            #                 choices=["Yes", "No"],
            #                 majorDimension=1,
            #                 style=wx.RA_SPECIFY_COLS,
            #             )
            #             self.overwrite.SetSelection(1)
            #             self.hbox3.Add(self.overwrite, 1, 1)
            #
            #             self.calibrate = wx.RadioBox(
            #                 self,
            #                 label="Calibrate animal assembly?",
            #                 choices=["Yes", "No"],
            #                 majorDimension=1,
            #                 style=wx.RA_SPECIFY_COLS,
            #             )
            #             self.calibrate.SetSelection(1)
            #             self.hbox4.Add(self.calibrate, 1, 1)
            #
            #             self.identity_toggle = wx.RadioBox(
            #                 self,
            #                 label="Assemble with identity only?",
            #                 choices=["Yes", "No"],
            #                 majorDimension=1,
            #                 style=wx.RA_SPECIFY_COLS,
            #             )
            #             self.identity_toggle.SetSelection(1)
            #             self.hbox4.Add(self.identity_toggle, 1, 1)
            #
            #             winsize_text = wx.StaticBox(self, label="Prioritize past connections over a window of size:")
            #             winsize_sizer = wx.StaticBoxSizer(winsize_text, wx.VERTICAL)
            #             self.winsize = wx.SpinCtrl(self, value="0")
            #             winsize_sizer.Add(self.winsize, 1, wx.EXPAND | wx.TOP | wx.BOTTOM,1)
            #             self.hbox4.Add(winsize_sizer, 1, 1)
        else:
            self.layout_save_filter = QtWidgets.QHBoxLayout()
            self.layout_save_filter.setAlignment(Qt.AlignLeft | Qt.AlignTop)

            self.layout_save_filter.setSpacing(200)
            self.layout_save_filter.setContentsMargins(20, 0, 50, 20)

            self._layout_save()
            self._layout_filter()
            self._layout_plot()
            self.layout_attributes.addLayout(self.layout_save_filter)

            self.layout_crop_plot = QtWidgets.QHBoxLayout()
            self.layout_crop_plot.setAlignment(Qt.AlignLeft | Qt.AlignTop)

            self.layout_crop_plot.setSpacing(170)
            self.layout_crop_plot.setContentsMargins(20, 0, 50, 20)

            self._layout_crop()
            self._layout_trajectories()

            self.layout_attributes.addLayout(self.layout_crop_plot)

        self.analye_videos_btn = QtWidgets.QPushButton("Analyze Videos")
        self.analye_videos_btn.setContentsMargins(0, 40, 40, 40)
        self.analye_videos_btn.clicked.connect(self.analyze_videos)

        self.edit_config_file_btn = QtWidgets.QPushButton("Edit config.yaml")
        self.edit_config_file_btn.clicked.connect(self.edit_config_file)

        self.layout_attributes.addWidget(self.analye_videos_btn, alignment=Qt.AlignRight)
        self.layout_attributes.addWidget(self.edit_config_file_btn, alignment=Qt.AlignRight)
        self.inLayout.addLayout(self.layout_attributes)

    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text

    def browse_dir(self):
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config[0]:
            return
        self.config = config[0]
        self.cfg_line.setText(self.config)

    def select_video(self):
        cwd = self.config.split("/")[0:-1]
        cwd = "\\".join(cwd)
        videos_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video to modify", cwd, "", "*.*"
        )
        if videos_file[0]:
            self.vids = videos_file[0]
            self.filelist.append(self.vids)
            self.select_video_button.setText(
                "Total %s Videos selected" % len(self.filelist)
            )
            self.select_video_button.adjustSize()

    def _layout_videotype(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Videotype")
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
        l_opt.setContentsMargins(0, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(1)
        self.shuffle.setMinimumWidth(400)
        self.shuffle.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.shuffle)
        self.layout_specify.addLayout(l_opt)

    def _layout_trainingset(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingset = QSpinBox()
        self.trainingset.setMaximum(100)
        self.trainingset.setValue(0)
        self.trainingset.setMinimumWidth(400)
        self.trainingset.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.trainingset)
        self.layout_specify.addLayout(l_opt)

    def _layout_save(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        self.save_as_csv = QtWidgets.QCheckBox("Save result(s) as csv")
        self.save_as_csv.setCheckState(Qt.Unchecked)
        self.save_as_csv.stateChanged.connect(self.update_csv_choice)

        l_opt.addWidget(self.save_as_csv)
        self.layout_save_filter.addLayout(l_opt)

    def _layout_filter(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        self.filter_predictions = QtWidgets.QCheckBox("Filter predictions")
        self.filter_predictions.setCheckState(Qt.Unchecked)
        self.filter_predictions.stateChanged.connect(self.update_filter_choice)

        l_opt.addWidget(self.filter_predictions)
        self.layout_save_filter.addLayout(l_opt)

    def _layout_plot(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        self.show_plots = QtWidgets.QCheckBox("Show plots")
        self.show_plots.setCheckState(Qt.Checked)
        self.show_plots.stateChanged.connect(self.update_showfigs_choice)

        l_opt.addWidget(self.show_plots)
        self.layout_save_filter.addLayout(l_opt)

    def _layout_crop(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        self.crop_bodyparts = QtWidgets.QCheckBox("Dynamically crop bodyparts")
        self.crop_bodyparts.setCheckState(Qt.Unchecked)
        self.crop_bodyparts.stateChanged.connect(self.update_crop_choice)

        l_opt.addWidget(self.crop_bodyparts)
        self.layout_crop_plot.addLayout(l_opt)

    def _layout_trajectories(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)

        # TODO: add CheckListBox:

        #   self.trajectory_to_plot = wx.CheckListBox(
        #       self, choices=bodyparts, style=0, name="Select the bodyparts"
        #   )
        #   self.trajectory_to_plot.Bind(wx.EVT_CHECKLISTBOX, self.getbp)
        #   self.trajectory_to_plot.SetCheckedItems(range(len(bodyparts)))
        #   self.trajectory_to_plot.Hide()

        self.plot_trajectories = QtWidgets.QCheckBox("Plot trajectories")
        self.plot_trajectories.setCheckState(Qt.Unchecked)
        self.plot_trajectories.stateChanged.connect(self.update_plot_trajectory_choice)

        l_opt.addWidget(self.plot_trajectories)
        self.layout_crop_plot.addLayout(l_opt)

    def update_csv_choice(self, s):
        if s == Qt.Checked:
            self.csv = True
            self.logger.info("Save results as CSV ENABLED")
        else:
            self.csv = False
            self.logger.info("Save results as CSV DISABLED")

    def update_filter_choice(self, s):
        if s == Qt.Checked:
            self.filter = True
            self.logger.info("Filtering predictions ENABLED")
        else:
            self.filter = False
            self.logger.info("Filtering predictions DISABLED")

    def update_showfigs_choice(self, s):
        if s == Qt.Checked:
            self.showfigs = True
            self.logger.info("Plots will show as pop ups.")
        else:
            self.showfigs = False
            self.logger.info("Plots will not show up.")

    def update_crop_choice(self, s):
        if s == Qt.Checked:
            self.dynamic = True
            self.logger.info("Bodyparts will be cropped dynamically.")
        else:
            self.dynamic = False
            self.logger.info("Bodyparts will not be cropped.")


    def update_plot_trajectory_choice(self, s):
        if s == Qt.Checked:
            self.trajectory = True
            # TODO: finish functionality
            # self.trajectory_to_plot.Show()
            # self.getbp()
        else:
            self.trajectory = False
            # self.trajectory_to_plot.Hide()
            # self.bodyparts = []

    def edit_config_file(self):
        
        if not self.config:
            return
        editor = ConfigEditor(self.config)
        editor.show()

    def getbp(self, event):
        self.bodyparts = list(self.trajectory_to_plot.GetCheckedStrings())

    def analyze_videos(self):
        shuffle = self.shuffle.value()
        trainingsetindex = self.trainingset.value()

        if self.cfg.get("multianimalproject", False):
            print("Analyzing ... ")
        else:
            save_as_csv = self.csv
            if self.dynamic:
                dynamic = (True, 0.5, 10)
            else:
                dynamic = (False, 0.5, 10)

            _filter = self.filter

        if self.cfg["cropping"] == "True":
            crop = self.cfg["x1"], self.cfg["x2"], self.cfg["y1"], self.cfg["y2"]
        else:
            crop = None
        if self.cfg.get("multianimalproject", False):
            print("multianimalproject")
        else:

            scorername = deeplabcut.analyze_videos(
                self.config,
                self.filelist,
                videotype=self.videotype.currentText(),
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                gputouse=None,
                save_as_csv=save_as_csv,
                cropping=crop,
                dynamic=dynamic,
            )
            if _filter:
                deeplabcut.filterpredictions(
                    self.config,
                    self.filelist,
                    videotype=self.videotype.currentText(),
                    shuffle=shuffle,
                    trainingsetindex=trainingsetindex,
                    filtertype="median",
                    windowlength=5,
                    save_as_csv=save_as_csv,
                )

            if self.trajectory:
                showfig = self.showfigs
                deeplabcut.plot_trajectories(
                    self.config,
                    self.filelist,
                    displayedbodyparts=self.bodyparts,
                    videotype=self.videotype.currentText(),
                    shuffle=shuffle,
                    trainingsetindex=trainingsetindex,
                    filtered=_filter,
                    showfigures=showfig,
                )
