#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""
Author: Hao Wu
hwu01@g.harvard.edu
You can find the directory for your ffmpeg bindings by: "find / | grep ffmpeg" and then setting it.

This is the helper class for video reading and saving in DeepLabCut.
Updated by AM

You can set various codecs below,
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
i.e. 'XVID'
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class VideoProcessor(ABC):
    """Base class for a video processing unit, implementation is required for video
    loading and saving.

    sh and sw are the output height and width respectively.
    """

    def __init__(self, fname="", sname="", nframes=-1, fps=None, codec="X264", sh="", sw=""):
        self.fname = fname
        self.sname = sname
        self.nframes = nframes
        self.codec = codec
        self.h = 0
        self.w = 0
        self.nc = 3
        self.i = 0

        try:
            if self.fname != "":
                self.vid = self.get_video()
                self.get_info()
                self.sh = 0
                self.sw = 0
            if self.sname != "":
                if sh == "" and sw == "":
                    self.sh = self.h
                    self.sw = self.w
                else:
                    self.sw = sw
                    self.sh = sh
                self.svid = self.create_video()

        except Exception as ex:
            print("Error: %s", ex)

        if fps is not None:  # Overwrite the video's FPS
            self.FPS = fps

    def load_frame(self):
        frame = self._read_frame()
        if frame is not None:
            self.i += 1
        return frame

    @property
    def height(self):
        return self.h

    @property
    def width(self):
        return self.w

    @property
    def fps(self):
        return self.FPS

    @property
    def counter(self):
        return self.i

    @property
    def frame_count(self):
        return self.nframes

    @abstractmethod
    def get_video(self):
        raise NotImplementedError("Implement your own get_video method.")

    @abstractmethod
    def get_info(self):
        raise NotImplementedError("Implement your own get_info method.")

    @abstractmethod
    def create_video(self):
        raise NotImplementedError("Implement your own create_video method.")

    @abstractmethod
    def _read_frame(self):
        raise NotImplementedError("Implement your own _read_frame method.")

    @abstractmethod
    def save_frame(self, frame):
        raise NotImplementedError("Implement your own save_frame method.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Implement your own close method.")


class VideoProcessorCV(VideoProcessor):
    """OpenCV implementation of VideoProcessor requires opencv-python==3.4.0.12."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_video(self):
        return cv2.VideoCapture(self.fname)

    def get_info(self):
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes > all_frames:
            self.nframes = all_frames

    def create_video(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        return cv2.VideoWriter(self.sname, fourcc, self.FPS, (self.sw, self.sh), True)

    def _read_frame(self):  # return RGB (rather than BGR)!
        # return cv2.cvtColor(np.flip(self.vid.read()[1],2), cv2.COLOR_BGR2RGB)
        success, frame = self.vid.read()
        if not success:
            return frame
        return np.flip(frame, 2)

    def save_frame(self, frame):
        if frame is not None:
            self.svid.write(np.flip(frame, 2))

    def close(self):
        if hasattr(self, "svid") and self.svid is not None:
            self.svid.release()
        if hasattr(self, "vid") and self.vid is not None:
            self.vid.release()
