from typing import Union, List, Callable, Tuple, Any, Dict

import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet.predictors.fastviterbi import FastViterbi
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData

class HeadedViterbi(Predictor):
    # TODO: Finish writing...
    def __init__(self, bodyparts: List[str], num_frames: int, settings: Union[Dict[str, Any], None]):
        super().__init__(bodyparts, num_frames, settings)

        self._wrapped_viterbi = FastViterbi(["head1", "head2"] + bodyparts, num_frames, settings)


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        pass

    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        pass

    @staticmethod
    def get_name() -> str:
        return "headedviterbi"

    @staticmethod
    def get_description() -> str:
        return FastViterbi.get_description()

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return FastViterbi.get_settings() + (("num_heads", "The number of heads within each frame"),)

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None