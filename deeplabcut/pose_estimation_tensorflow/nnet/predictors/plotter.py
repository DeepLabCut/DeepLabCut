
# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable

import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData

# Used specifically by plugin...
import math
import numpy as np
from matplotlib import pyplot
import cv2


class PlotterArgMax(Predictor):
    """
    Identical to singleargmax, but plots at most 100 probability frames to the user using matplotlib
    during processing...
    """

    def __init__(self, bodyparts: List[str], num_frames: int, settings: Dict[str, Any]):
        super().__init__(bodyparts, num_frames, settings)
        self._parts = bodyparts
        self._num_frames = num_frames

        # Keeps track of how many frames
        self._current_frame = 0
        # Determines grid size of charts
        self._grid_size = int(math.ceil(math.sqrt(len(self._parts))))
        # Stores opencv video writer...
        self._vid_writer = None

        # Name of the video file to save to
        self.VIDEO_NAME = settings["video_name"]
        # Output codec to save using
        self.OUTPUT_CODEC = cv2.VideoWriter_fourcc(*settings["codec"])
        # Frames per second to use for video
        self.OUTPUT_FPS = settings["output_fps"]
        # Determines if we are using log scaling...
        self.LOG_SCALE = settings["use_log_scale"]


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        for frame in range(scmap.get_frame_count()):
            # Plot all probability maps
            for bp in range(scmap.get_bodypart_count()):
                pyplot.subplot(self._grid_size, self._grid_size, bp + 1)
                pyplot.title(f"Bodypart: {self._parts[bp]}, Frame: {self._current_frame}")
                pyplot.pcolormesh(np.log(scmap.get_prob_table(frame, bp)) if (self.LOG_SCALE) else
                                  scmap.get_prob_table(frame, bp))
                # This reverses the y-axis data, so as probability maps match the video...
                pyplot.ylim(pyplot.ylim()[::-1])

            # Save chart to the buffer.
            fig = pyplot.gcf()
            pyplot.tight_layout()
            pyplot.gcf().canvas.draw()

            # Convert plot to cv2 image, then plot it...
            img = cv2.imdecode(np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                               .reshape(fig.canvas.get_width_height()[::-1] + (3,)), cv2.IMREAD_COLOR)

            # Clear plotting done by pyplot....
            pyplot.clf()

            # If the video writer does not exist, create it now...
            if (self._vid_writer is None):
                height, width, layers = img.shape
                self._vid_writer = cv2.VideoWriter(self.VIDEO_NAME, self.OUTPUT_CODEC, self.OUTPUT_FPS,
                                                   (width, height))

            # Write image to the video writer...
            self._vid_writer.write(img)
            self._current_frame += 1


        # Return argmax values for each frame...
        return scmap.get_poses_for(scmap.get_max_scmap_points())


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        # Release the video writer...
        self._vid_writer.release()
        # We are done, return None...
        return None

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("video_name", "Name of the video file that plotting data will be saved to.", "prob-dlc.mp4"),
            ("codec", "The codec to be used by the opencv library to save info to, typically a 4-byte string.", "MPEG"),
            ("output_fps", "The frames per second of the output video, as an number.", 15),
            ("use_log_scale", "Boolean, determines whether to apply log scaling to the frames in the video.", True)
        ]

    @staticmethod
    def get_name() -> str:
        return "plotterargmax"

    @staticmethod
    def get_description() -> str:
        return "Identical to singleargmax, but plots a video of probability frames using matplotlib" \
               "during processing..."

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None