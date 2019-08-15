from typing import Union, List, Callable, Tuple, Any, Dict

import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet.predictors.fastviterbi import FastViterbi
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData

class HeadedViterbi(Predictor):

    # TODO: Finish writing...
    def __init__(self, bodyparts: List[str], num_frames: int, settings: Union[Dict[str, Any], None]):
        super().__init__(bodyparts, num_frames, settings)

        self._bodyparts = bodyparts

        # Convert noses and tails to a list of tuples, being in the format: (nose index, tail index)
        self._noses = {part: i for i, part in enumerate(bodyparts) if (part.startswith("nose"))}
        self._tails = {part: i for i, part in enumerate(bodyparts) if (part.startswith("tail"))}

        self._body_clusters = []

        nose_str_len = len("nose")
        tail_str_len = len("tail")

        for nose in self._noses:
            for tail in self._tails:
                if(nose[nose_str_len:] == tail[tail_str_len:]):
                    self._body_clusters.append((self._noses[nose], self._tails[tail]))

        self._wrapped_viterbi = FastViterbi(self._head_list + bodyparts, num_frames, settings)


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
        return FastViterbi.get_settings()

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None