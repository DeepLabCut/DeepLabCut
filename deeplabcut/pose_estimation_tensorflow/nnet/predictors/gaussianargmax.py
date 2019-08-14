from typing import Union, List, Tuple, Any, Dict, Callable
import tqdm
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData
import numpy as np


class GaussianArgMax(Predictor):
    """
    Predictor plugin that computes the next frame's max by multiplying the current frame by a gaussian centered at the
    prior frame's maximum point location.
    """

    def __init__(self, bodyparts: List[str], num_frames: int, settings: Dict[str, Any]):
        """ Creates a new Gaussian arg max predictor... """
        super().__init__(bodyparts, num_frames, settings)
        # Store number of bodyparts and frames for later use...
        self._part_len = len(bodyparts)
        self._num_frames = num_frames

        # Will store the prior max location
        self._prior_max_loc = None
        # Stores the gaussian table
        self._gaussian_table = None
        # Global values for the gaussian formula, can be adjusted in dlc_config for differing results...
        self.NORM_DIST = settings["norm_dist"]  # The normal distribution
        self.AMPLITUDE = settings["amplitude"]  # The amplitude, or height of the gaussian curve
        self.LOWEST_VAL = settings["lowest_val"]  # Changes the lowest value that the gaussian curve can produce


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
        return self.AMPLITUDE * np.exp(-(inner_x_delta + inner_y_delta)) + self.LOWEST_VAL


    def _compute_gaussian_table(self, width: int, height: int):
        """
        Compute the gaussian table given the width and height of each probability frame. Results are stored in
        self._gaussian_table
        """
        # Compute the coordinates of the origin
        ox, oy = width - 1, height - 1

        # Compute the width and height of the gaussian table
        g_t_width, g_t_height = (width * 2) - 1, (height * 2) - 1

        # Create empty array
        self._gaussian_table = np.zeros((g_t_height, g_t_width), dtype="float32")

        # Iterate table filling in values
        for gy in range(self._gaussian_table.shape[0]):
            for gx in range(self._gaussian_table.shape[1]):
                # We add log scale now so we don't have to do it later...
                self._gaussian_table[gy, gx] = np.log(self._gaussian_formula(ox, gx, oy, gy))


    def _gaussian_table_at(self, current_x: int, current_y: int, width: int, height: int):
        """
        Get the gaussian table for a probability frame given the current point within the frame being compared to

        :param current_x: The current x location within the current frame being looked at
        :param current_y: The current y location within the current frame being looked at
        :param width: The width of the probability frame...
        :param height: The height of the probability frame...
        :return: The slice or subsection of _gaussian_table for this point in this frame.
        """
        # Get the origin location
        ox, oy = width - 1, height - 1

        # Compute offset of the probability map within the gaussian map
        off_x, off_y = ox - current_x, oy - current_y

        # Return slice of gaussian table correlated to this probability frame...
        return self._gaussian_table[off_y:off_y + height, off_x:off_x + width]


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        frame, height, width, parts = scmap.get_source_map().shape
        gaus_temp = TrackingData.empty_tracking_data(frame, parts, width, height)
        offset = 0

        # If the prior frame is not set, we have just started, just set it to the max of the first frame
        if(self._prior_max_loc is None):
            # Precompute the gaussian table
            self._compute_gaussian_table(width, height)
            # Set prior max to maximums of the first frame...
            self._prior_max_loc = np.unravel_index(
                np.argmax(scmap.get_source_map()[0].reshape(height * width, parts), axis=0), (height, width)
            )

            # Copy over first frame to the temporary gaussian modified map
            gaus_temp.get_source_map()[0] = scmap.get_source_map()[0]
            # Jump over first frame from scmap using offset, as we still need to compute the rest
            offset = 1


        for fr in range(offset, frame):
            # Get x and y tuples from max location
            y, x = self._prior_max_loc

            for bp in range(parts):
                # Set the probability table of the current frame to the current frame times the gaussian
                # centered at the max of the prior frame...
                gaus_temp.set_prob_table(
                    fr, bp, np.log(scmap.get_prob_table(fr, bp)) + self._gaussian_table_at(x[bp], y[bp], width, height)
                )

            # Update prior max location to the next frame(or set to current frame).
            self._prior_max_loc = np.unravel_index(
                np.argmax(gaus_temp.get_source_map()[fr].reshape(height * width, parts), axis=0), (height, width)
            )

        # Return all original probabilities at gaussian modified maximums.
        return scmap.get_poses_for(gaus_temp.get_max_scmap_points())



    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        # We are finished
        return None

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("norm_dist", "The normal distribution of the 2D gaussian curve used"
                          "for transition probabilities by the viterbi algorithm.", 1),
            ("amplitude", "The amplitude of the gaussian curve used by the viterbi algorithm.", 1),
            ("lowest_val", "The lowest value of the gaussian curve used by the viterbi algorithm."
                           "Really a constant that is added on the the 2D gaussian to give all points"
                           "a minimum probability.", 0)
        ]

    @staticmethod
    def get_name() -> str:
        return "gaussianargmax"

    @staticmethod
    def get_description() -> str:
        return ("Predictor plugin that computes the next frame's max by multiplying the current frame by a gaussian"
                "centered at the frame's maximum point location.")

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None