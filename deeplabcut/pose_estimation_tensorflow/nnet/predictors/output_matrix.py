
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
        self._first_run = 0


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # If we managed to import numplotlib, render plots and save them to a video...
        if(self._first_run > 360):
            self._first_run = True

            with open("scmap.txt", "w") as f:
                for y in range(scmap.get_frame_height()):
                    for x in range(scmap.get_frame_width()):
                        f.write(f"{x} {y} {scmap.get_prob_table(0, 0)[y, x]}")


        self._first_run += scmap.get_frame_count()
        # Return argmax values for each frame...
        return scmap.get_poses_for(scmap.get_max_scmap_points())

    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        # We are done, return None...
        return None

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return None

    @staticmethod
    def get_name() -> str:
        return "plotterargmax"

    @staticmethod
    def get_description() -> str:
        return "Identical to singleargmax, but plots a video of probability frames using matplotlib \n" \
               "during processing..."