
# For types in methods
from typing import Union, List, Tuple, Any, Callable, Dict

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose


class SingleArgMaxPredict(Predictor):
    """
    Default processor for DeepLabCut, and the code originally used by DeepLabCut for prediction of points. Predicts
    the point from the probability frames simply by selecting the max probability in the frame.
    """
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int, settings: None,
                 video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._num_outputs = num_outputs


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # Using new object library to get the max... Drastically simplified logic...
        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))

    def on_end(self, pbar) -> Union[None, Pose]:
        return None

    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return None

    @classmethod
    def get_name(cls) -> str:
        return "argmax"

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True