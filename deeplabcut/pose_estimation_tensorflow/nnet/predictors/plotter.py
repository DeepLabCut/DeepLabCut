
# For types in methods
from typing import Union, List

# Plugin base class
import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData


class PlotterArgMax(Predictor):
    """
    Identical to singleargmax, but plots at most 100 probability frames to the user using matplotlib
    during processing...
    """
    def __init__(self, bodyparts: List[str], num_frames: int):
        super().__init__(bodyparts, num_frames)
        self._parts = bodyparts
        self._num_frames = num_frames

        # Attempt to import matplotlib, if it fails set a flag to indicate so...
        try:
            from matplotlib import pyplot
            self._has_plotter = True
        except ImportError:
            print("Error: Unable to import matplotlib, fall back to just computing frames...")
            self._has_plotter = False

        # Keeps track of how many frames
        self._current_frame = 0
        self._last_shown_frame = 0
        self._step = self._num_frames / 100

    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # If we managed to import numplotlib, and we have gone enough frames to move a step, display plots...
        if(self._has_plotter and (self._current_frame - self._last_shown_frame >= self._step)):
            from matplotlib import pyplot

            for bp in range(scmap.get_bodypart_count()):
                pyplot.subplot(2, 2, bp + 1)
                pyplot.title(f"Bodypart: {self._parts[bp]}, Frame: {self._current_frame}")
                pyplot.pcolormesh(scmap.get_prob_table(0, bp))

            pyplot.show()

        # Otherwise increment counter and return same selections as singleargmax plugin would....
        self._current_frame += scmap.get_frame_count()
        return scmap.get_poses_for(scmap.get_max_scmap_points())


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        # We are done, return None...
        return None

    @staticmethod
    def get_name() -> str:
        return "plotterargmax"

    @staticmethod
    def get_description() -> str:
        return "Identical to singleargmax, but plots at most 100 probability frames to the user using matplotlib \n" \
               "during processing..."