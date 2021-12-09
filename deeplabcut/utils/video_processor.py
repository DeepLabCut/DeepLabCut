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

import cv2
import numpy as np


class VideoProcessor(object):
    """
    Base class for a video processing unit, implementation is required for video loading and saving

    sh and sw are the output height and width respectively.
    """

    def __init__(
        self, fname="", sname="", nframes=-1, fps=None, codec="X264", sh="", sw=""
    ):
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

    def height(self):
        return self.h

    def width(self):
        return self.w

    def fps(self):
        return self.FPS

    def counter(self):
        return self.i

    def frame_count(self):
        return self.nframes

    def get_video(self):
        """
        implement your own
        """
        pass

    def get_info(self):
        """
        implement your own
        """
        pass

    def create_video(self):
        """
        implement your own
        """
        pass

    def _read_frame(self):
        """
        implement your own
        """
        pass

    def save_frame(self, frame):
        """
        implement your own
        """
        pass

    def close(self):
        """
        implement your own
        """
        pass


class VideoProcessorCV(VideoProcessor):
    """
    OpenCV implementation of VideoProcessor
    requires opencv-python==3.4.0.12
    """

    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)

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
        self.svid.write(np.flip(frame, 2))

    def close(self):
        if hasattr(self, "svid") and self.svid is not None:
            self.svid.release()
        if hasattr(self, "vid") and self.vid is not None:
            self.vid.release()
