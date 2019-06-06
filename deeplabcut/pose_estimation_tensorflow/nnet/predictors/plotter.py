
# For types in methods
from typing import Union, List, Tuple, Any, Dict

import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData

# Used specifically by plugin...
import math
import numpy as np
import io
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

        # Attempt to import matplotlib, if it fails set a flag to indicate so...
        try:
            from matplotlib import pyplot
            self._has_plotter = pyplot
        except ImportError:
            print("Error: Unable to import matplotlib, fall back to just computing frames...")
            self._has_plotter = None

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


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # If we managed to import numplotlib, render plots and save them to a video...
        if(self._has_plotter is not None):
            pyplot = self._has_plotter

            for frame in range(scmap.get_frame_count()):
                # Allocate buffer to store output of pyplot
                buffer = io.BytesIO()

                # Plot all probability maps
                for bp in range(scmap.get_bodypart_count()):
                    pyplot.subplot(self._grid_size, self._grid_size, bp + 1)
                    pyplot.title(f"Bodypart: {self._parts[bp]}, Frame: {self._current_frame}")
                    pyplot.pcolormesh(np.log(scmap.get_prob_table(frame, bp)))

                # Save chart to the buffer.
                pyplot.tight_layout()
                pyplot.savefig(buffer, format="png")
                pyplot.clf()
                buffer.seek(0)

                # Convert buffer cv2 image, then close the buffer
                img = cv2.imdecode(np.frombuffer(buffer.getbuffer(), dtype=np.uint8), cv2.IMREAD_COLOR)
                buffer.close()

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
            ("codec", "The codec to be used by the opencv library to save info to, typically a 4-byte string.", "X264"),
            ("output_fps", "The frames per second of the output video, as an number.", 15)
        ]

    @staticmethod
    def get_name() -> str:
        return "plotterargmax"

    @staticmethod
    def get_description() -> str:
        return "Identical to singleargmax, but plots at most 100 probability frames to the user using matplotlib \n" \
               "during processing..."