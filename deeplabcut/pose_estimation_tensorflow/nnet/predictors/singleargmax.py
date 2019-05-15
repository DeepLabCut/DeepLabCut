
# For types in methods
from typing import Union, List, Tuple
from numpy import ndarray

import numpy as np

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose


class SingleArgMaxPredict(Predictor):
    """
    Default processor for DeepLabCut, and was the code originally used by DeepLabCut historically. Predicts
    the point from the probability frames simply by selecting the max probability in the source frame.
    """
    def __init__(self, bodyparts: List[str]):
        super().__init__(bodyparts)
        self.bodyparts = bodyparts
        self.num_parts = len(bodyparts)


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # Using new object library to get the max...
        return scmap.get_poses_for(scmap.get_max_scmap_points())

    def on_end(self) -> Union[None, Pose]:
        # Processing is done per frame, so return None.
        return None

    @staticmethod
    def get_name() -> str:
        return "singleargmax"


    @staticmethod
    def get_description() -> str:
        return ("Default processor for DeepLabCut, and was the code originally used by DeepLabCut \n"
                "historically. Predicts the point from the probability frames simply by selecting \n"
                "the max probability in the source frame.")