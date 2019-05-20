
# For types in methods
from typing import Union, List, Tuple
from numpy import ndarray
import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import math
import numpy as np

class Viterbi(Predictor):
    """
    A predictor that applies the Viterbi algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive.
    """
    # Global values for the gaussian formula, can be adjusted for differing results...
    NORM_DIST = 1  # The normal distribution
    AMPLITUDE = 1  # The amplitude, or height of the gaussian curve


    def __init__(self, bodyparts: List[str], num_frames: int):
        super().__init__(bodyparts, num_frames)

        # Store bodyparts and num_frames for later use, they become useful
        self._bodyparts = bodyparts
        self._num_frames = num_frames
        # Used to store viterbi frames
        self._viterbi_frames: TrackingData= None
        self._current_frame = 0



    def _gaussian_formula(self, prior_x: float, x: float, prior_y: float, y: float) -> float:
        """
        Private method, computes location of point (x, y) on a gaussian curve given a prior point (x_prior, y_prior)
        to use as the center.

        :param prior_x: The x point in the prior frame
        :param x: The current frame's x point
        :param prior_y: The y point in the prior frame
        :param y: The current frame's y point
        :return: The location of x, y given gaussian curve centered at x_prior, y_prior
        """

        # Formula for 2D gaussian curve (or bump)
        inner_x_delta = ((prior_x - x) ** 2) / (2 * self.NORM_DIST ** 2)
        inner_y_delta = ((prior_y - y) ** 2) / (2 * self.NORM_DIST ** 2)
        return self.AMPLITUDE * np.exp(-(inner_x_delta + inner_y_delta))


    def _viterbi_prob_point(self, x: int, y: int, prob: ndarray, prior_frame: ndarray) -> ndarray:
        """
        Private method, computes viterbi probability for a given point

        :param x: Integer, the x location of the point
        :param y: Integer, the y location of the point
        :param prob: numpy array of floats, the probability of bodyparts at the point
        :param prior_frame: numpy array, the frame prior of this point's frame
        :return: The viterbi probability of this point, as an numpy array of floats.
        """
        # Get width and height of frame
        height = prior_frame.shape[0]
        width = prior_frame.shape[1]

        # Create a temporary array to store values
        temp = np.zeros(prior_frame.shape[2], height, width)

        # Iterating all point indexes
        for yi in prior_frame.shape[0]:
            for xi in prior_frame.shape[1]:
                # Viterbi of point = prior viterbi * gaussian curve to encourage near points * current probabilities
                temp[:, yi, xi] = prior_frame[yi, xi] + np.log(self._gaussian_formula(xi, x, yi, y)) + np.log(prob)

        # Grab the maximums of each probability...
        return np.max(temp.reshape(temp.shape[2], width * height), axis=1).squeeze()


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        """ Handles Forward part of the viterbi algorithm, allowing for faster post processing. """
        # Check if this is the first frame
        if(self._viterbi_frames is None):
            # Create the TrackingData to hold the viterbi map.
            viterbi_scmap = np.zeros((self._num_frames, scmap.get_frame_height(),
                                      scmap.get_frame_width(), len(self._bodyparts)), dtype="float32")

            viterbi_locref = np.zeros((self._num_frames, scmap.get_frame_height(),
                                      scmap.get_frame_width(), len(self._bodyparts), 2), dtype="float32")

            self._viterbi_frames = TrackingData(viterbi_scmap, viterbi_locref, scmap.get_down_scaling())

            # Set the first frame of the map to the log scaled version of the first frame in the original source
            self._viterbi_frames.get_source_map()[0] = np.log(scmap.get_source_map()[0])
            # Remove first frame from the temporary source map and increment the counter
            scmap.set_source_map(scmap.get_source_map()[1:])
            self._current_frame += 1

        # Normal frame processing for all frames past first one:
        # Iterate each frame of the batch
        for frame_i in range(scmap.get_frame_count()):
            temp_frame = np.zeros((scmap.get_frame_height(), scmap.get_frame_width(), scmap.get_bodypart_count()))

            for yi in range(scmap.get_frame_height()):
                for xi in range(scmap.get_frame_width()):
                    # Compute viterbi for this point....
                    temp_frame[yi, xi] = self._viterbi_prob_point(xi, yi, scmap.get_source_map()[frame_i, yi, xi],
                                                    self._viterbi_frames.get_source_map()[self._current_frame - 1])

            # Add frame to viterbi map.
            self._viterbi_frames.get_source_map()[self._current_frame] = temp_frame
            # Increase frames processed counter...
            self._current_frame += 1

        # Return none, as we can't return any poses yet...
        return None

    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        """ Handles backward part of viterbi, and then returns the poses """
        r_counter = self._num_frames - 1

        # Allocate pose object to store poses


        # Array to store maximums of each body part for a single frame.
        current_max_i = self._viterbi_frames.get_max_of_frame(r_counter)


        r_counter -= 1

        # Begin backtrack
        # TODO: Backtrack...
        while(r_counter > 0):



    @staticmethod
    def get_name() -> str:
        return "viterbi"

    @staticmethod
    def get_description() -> str:
        return ("A predictor that applies the Viterbi algorithm to frames in order to predict poses.\n"
                "The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but\n"
                "is also more memory intensive and computationally expensive.")