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

import logging
from abc import ABC, abstractmethod
from typing import Literal

import cv2
import numpy as np

from deeplabcut.utils.deprecation import deprecated

logger = logging.getLogger(__name__)


class VideoProcessor(ABC):
    """Abstract base class for video reading and writing.

    Subclasses implement backend-specific video loading, metadata extraction,
    output video creation, frame reading, frame writing, and cleanup.

    Args:
        fname (str): Path to the input video. If empty, no input video is opened.
        sname (str): Path to the output video. If empty, no output video is created.
        nframes (int): Number of frames to process. ``-1`` means all frames.
        fps (float | None): Optional FPS override.
        codec (str): FourCC codec string used for output videos.
        sh (int | Literal[""] | None): Output video height. ``""`` and ``None``
            mean use the input height.
        sw (int | Literal[""] | None): Output video width. ``""`` and ``None``
            mean use the input width.

    Attributes:
        fname (str): Input video path.
        sname (str): Output video path.
        nframes (int): Number of frames to process.
        video_fps (float | None): Video frame rate.
        FPS (float | None): Legacy alias for ``video_fps``.
        h (int): Input video height.
        w (int): Input video width.
        nc (int): Number of channels.
        i (int): Number of successfully loaded frames.
        vid: Backend-specific input video object.
        svid: Backend-specific output video object.
        sh (int): Output video height.
        sw (int): Output video width.

    Notes:
        ``height()``, ``width()``, ``fps()``, ``counter()``, and
        ``frame_count()`` are retained as methods for backwards compatibility.
    """

    def __init__(
        self,
        fname: str = "",
        sname: str = "",
        nframes: int = -1,
        fps: float | None = None,
        codec: str = "X264",
        sh: int | Literal[""] | None = "",
        sw: int | Literal[""] | None = "",
    ):
        self._fname = None
        self._sname = None
        self._fps = None
        self._nframes = None
        self._h = 0
        self._w = 0
        self.vid = None
        self.svid = None
        self.sh = 0
        self.sw = 0
        self.fname = fname
        self.sname = sname
        self.codec = codec
        self.nframes = nframes
        self.nc = 3
        self.i = 0

        try:
            if self.fname != "":
                self.vid = self.get_video()
                self.get_info()
                self.sh = 0
                self.sw = 0
            if self.sname != "":
                if sh in ("", None) and sw in (None, ""):
                    self.sh = self._h
                    self.sw = self._w
                else:
                    self.sw = sw
                    self.sh = sh
                self.svid = self.create_video()

        except Exception as ex:
            logger.exception("VideoProcessor initialization failed: %s", ex)

        if fps is not None:  # Overwrite the video's FPS
            # NOTE @C-Achard 2026-06-09 improving checks here might break old API
            # same for raising on missing FPS
            self.video_fps = fps

    def load_frame(self):
        frame = self._read_frame()
        if frame is not None:
            self.i += 1
        return frame

    @property
    def fname(self):
        return self._fname

    @fname.setter
    def fname(self, value):
        self._fname = "" if value in (None, "") else str(value)

    @property
    def sname(self):
        return self._sname

    @sname.setter
    def sname(self, value):
        self._sname = "" if value in (None, "") else str(value)

    @property
    def h(self):
        return self._h

    @property
    def w(self):
        return self._w

    @h.setter
    def h(self, value):
        self._h = int(value)

    @w.setter
    def w(self, value):
        self._w = int(value)

    @property
    def video_fps(self):
        return self._fps

    @video_fps.setter
    def video_fps(self, value):
        self._fps = None if value is None else float(value)

    @property
    def nframes(self):
        return self._nframes

    @nframes.setter
    def nframes(self, value):
        self._nframes = int(value)

    ### Legacy compatibility methods
    @deprecated(replacement="VideoProcessor.i", since="3.1")
    def counter(self):
        return self.i

    @deprecated(replacement="VideoProcessor.h", since="3.1")
    def height(self):
        return self._h

    @deprecated(replacement="VideoProcessor.w", since="3.1")
    def width(self):
        return self._w

    @deprecated(replacement="VideoProcessor.video_fps", since="3.1")
    def fps(self):
        return self._fps

    @deprecated(replacement="VideoProcessor.video_fps", since="3.1")
    @property
    def FPS(self):
        return self._fps

    @deprecated(replacement="VideoProcessor.video_fps", since="3.1")
    @FPS.setter
    def FPS(self, value):
        self.video_fps = None if value is None else float(value)

    def frame_count(self):
        return self._nframes

    ###

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
    """OpenCV-backed video reader and writer.

    This implementation uses cv2.VideoCapture for reading videos and
    cv2.VideoWriter for writing videos. Frames returned by
    `load_frame` are converted from OpenCV's native BGR channel order to
    RGB channel order. Frames passed to `save_frame` are expected to be in
    RGB order and are converted back to BGR before writing.

    Attributes:
        fname (str): Path to the input video. If empty, no input video is opened.
        sname (str): Path to the output video. If empty, no output video is created.
        nframes (int): Number of frames to process. If initialized as ``-1``,
            it is replaced by the total number of frames reported by OpenCV.
        codec (str): FourCC codec string used when creating the output video.
        h (int): Input video height in pixels.
        w (int): Input video width in pixels.
        nc (int): Number of channels. This implementation uses ``3``.
        i (int): Number of frames successfully loaded through ``load_frame()``.
        FPS (float): Frames per second reported by OpenCV, or the user-provided
            override.
        sh (int): Output video height in pixels.
        sw (int): Output video width in pixels.
        vid (cv2.VideoCapture | None): OpenCV video reader.
        svid (cv2.VideoWriter | None): OpenCV video writer.
    """

    def get_video(self):
        """Open the input video with OpenCV.

        Returns:
            cv2.VideoCapture: OpenCV video capture object for ``self.fname``.
        """
        return cv2.VideoCapture(self.fname)

    def get_info(self):
        """Populate metadata from the OpenCV video reader.

        Sets:
            self.w: Frame width in pixels.
            self.h: Frame height in pixels.
            self.nframes: Number of frames to process.
            self.FPS: Frames per second reported by OpenCV.
            self.nc: Number of channels, always ``3``.

        Notes:
            If ``self.nframes`` is ``-1`` or greater than the total number of
            frames reported by OpenCV, it is replaced by OpenCV's frame count.
        """
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3

        if self.nframes == -1 or self.nframes > all_frames:
            self.nframes = all_frames

    def create_video(self):
        """Create an OpenCV video writer.

        Returns:
            cv2.VideoWriter: OpenCV video writer for ``self.sname``.

        Notes:
            ``self.sw`` and ``self.sh`` are expected to be set by the base class
            before this method is called. The codec is interpreted as a FourCC
            string, preserving the historical OpenCV behavior.
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        return cv2.VideoWriter(self.sname, fourcc, self.video_fps, (self.sw, self.sh), True)

    def _read_frame(self):
        """Read the next video frame.

        Returns:
            numpy.ndarray | None: The next frame in RGB channel order, or
            ``None`` if no frame could be read.

        Notes:
            OpenCV returns BGR frames. This method converts them to RGB using
            ``np.flip(frame, 2)`` to preserve legacy behavior.
        """
        if self.vid is None:
            return None

        success, frame = self.vid.read()
        if not success:
            return frame

        return np.flip(frame, 2)

    def save_frame(self, frame):
        """Write one RGB frame to the output video.

        Args:
            frame (numpy.ndarray | None): RGB frame to write. ``None`` is ignored.

        Notes:
            This method preserves the historical behavior of silently ignoring
            ``None`` frames. Non-``None`` frames are converted from RGB to BGR
            before being passed to OpenCV.
        """
        if frame is None:
            return
        if self.svid is not None:
            self.svid.write(np.flip(frame, 2))
        else:
            logger.warning(f"Could not write video because no output video writer is open for {self.sname}")

    def close(self):
        """Release OpenCV reader and writer resources.

        This method is safe to call multiple times. After release, ``self.svid``
        and ``self.vid`` are set to ``None`` to avoid accidental reuse of closed
        OpenCV handles.
        """
        if hasattr(self, "svid") and self.svid is not None:
            self.svid.release()
            self.svid = None

        if hasattr(self, "vid") and self.vid is not None:
            self.vid.release()
            self.vid = None
