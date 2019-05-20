
# For types in methods
from typing import Union, List

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose


class SingleArgMaxPredict(Predictor):
    """
    Default processor for DeepLabCut, and the code originally used by DeepLabCut for prediction of points. Predicts
    the point from the probability frames simply by selecting the max probability in the frame.
    """
    def __init__(self, bodyparts: List[str], num_frames: int):
        super().__init__(bodyparts, num_frames)


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # Using new object library to get the max... Drastically simplified logic...
        return scmap.get_poses_for(scmap.get_max_scmap_points())

    def on_end(self, pbar) -> Union[None, Pose]:
        # Processing is done per frame, so return None.
        return None

    @staticmethod
    def get_name() -> str:
        return "singleargmax"


    @staticmethod
    def get_description() -> str:
        return ("Default processor for DeepLabCut, and was the code originally used by DeepLabCut \n"
                "historically. Predicts the point from the probability frames simply by selecting \n"
                "the max probability in the frame.")