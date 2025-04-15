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
"ffmpeg -encoders | grep -e '^ V'"
i.e. 'h264', 'libx265', 'mjpeg', 'mpeg4'
"""

import ffmpegcv


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
    FFmpegCV implementation of VideoProcessor
    """

    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)

    def get_video(self):
        return ffmpegcv.VideoCapture(self.fname, pix_fmt='rgb24')

    def get_info(self):
        self.w = self.vid.width
        self.h = self.vid.height
        all_frames = len(self.vid)
        self.FPS = self.vid.fps
        self.nc = 3
        if self.nframes == -1 or self.nframes > all_frames:
            self.nframes = all_frames

    def create_video(self):
        return ffmpegcv.VideoWriter(self.sname, self.codec, self.FPS, pix_fmt="rgb24")

    def _read_frame(self):
        success, frame = self.vid.read()
        if not success:
            return frame
        return frame.copy()

    def save_frame(self, frame):
        if frame is not None:
            self.svid.write(frame)

    def close(self):
        if hasattr(self, "svid") and self.svid is not None:
            self.svid.release()
        if hasattr(self, "vid") and self.vid is not None:
            self.vid.release()
