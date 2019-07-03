# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable
from numpy import ndarray
import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import numpy as np


class FastViterbi(Predictor):
    # STABLE STATE 1:
    """
    A predictor that applies the Viterbi algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive. This specific implementation
    uses sparse matrix multiplication(log addition) for massive speedup over the normal
    viterbi implementation...)
    """

    def __init__(self, bodyparts: List[str], num_frames: int, settings: Dict[str, Any]):
        """ Initialized a fastviterbi plugin for analyzing a video """
        super().__init__(bodyparts, num_frames, settings)

        # Store bodyparts and num_frames for later use, they become useful
        self._bodyparts = bodyparts
        self._num_frames = num_frames

        # Will hold the viterbi frames on the forward compute...
        # Dimension are: Frame -> Bodypart * 2 -> (y, x), (probability, loc_off_x, loc_off_y, old probability)
        # The x, y are split from other values because they are integers while others are floats...
        self._viterbi_frames: List[List[Union[ndarray, None]]] = [None] * num_frames
        # Represents precomputed gaussian values...
        self._gaussian_table: ndarray = None
        # Stores stride used in this video...
        self._down_scaling = None
        # Keeps track of current frame...
        self._current_frame = 0

        # Global values for the gaussian formula, can be adjusted in dlc_config for differing results...
        self.NORM_DIST = settings["norm_dist"]  # The normal distribution
        self.AMPLITUDE = settings["amplitude"]  # The amplitude, or height of the gaussian curve
        self.LOWEST_VAL = settings["lowest_gaussian_value"]  # Changes the lowest value that the gaussian curve can produce
        self.THRESHOLD = settings["threshold"] # The threshold for the matrix... Everything below this value is ignored.

    @staticmethod
    def log(num):
        """ Computes and returns the natural logarithm of the number or numpy array """
        return np.log(num)

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

    def _compute_gaussian_table(self, width: int, height: int) -> None:
        """
        Compute the gaussian table given the width and height of each probability frame. Results are stored in
        self._gaussian_table
        """
        # Allocate gaussian table of width x height...
        self._gaussian_table = np.zeros((height, width), dtype="float32")

        # Iterate through filling in the values, using (0, 0) as the prior coordinate
        for y in range(height):
            for x in range(width):
                self._gaussian_table[y, x] = self.log(self._gaussian_formula(0, x, 0, y))

        # Done, return...
        return

    def _gaussian_values_at(self, current_xs: ndarray, current_ys: ndarray, prior_xs: ndarray, prior_ys: ndarray) -> ndarray:
        """
        Get the gaussian values for a collection of points provided as arrays....

        :param current_xs: The array storing the current frame's x values
        :param current_ys: The array storing the current frame's y values
        :param prior_xs: The array storing the prior frame's x values
        :param prior_ys: The array storing the prior frame's y values
        :return: A 2D array representing all gaussian combinations that need to be computed for this frame...
        """
        # Compute delta x and y values. Use broadcasting to compute is for all current frames.
        # We take the absolute value since differences can be negative and gaussian mirrors over the axis...
        delta_x = np.abs(np.expand_dims(current_xs, axis=1) - np.expand_dims(prior_xs, axis=0))
        delta_y = np.abs(np.expand_dims(current_ys, axis=1) - np.expand_dims(prior_ys, axis=0))

        # Return the current_frame by prior_frame gaussian array by just plugging in y and x indexes...
        # This is the easiest way I have found to do it but I am sure there is a better way...
        return self._gaussian_table[delta_y.flatten(), delta_x.flatten()].reshape(delta_y.shape)


    def _compute_init_frame(self, bodypart: int, frame: int, scmap: TrackingData):
        """ Inserts the initial frame, or first frame to have actual points that are above the threshold. """
        # Get coordinates for all above threshold probabilities in this frame...
        coords = np.nonzero(scmap.get_prob_table(frame, bodypart) > self.THRESHOLD)

        # If no points are found above the threshold, frame for this bodypart is a dud, set it to none...
        if (len(coords[0]) == 0):
            self._viterbi_frames[self._current_frame][bodypart * 2] = None
            self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = None
            return

        # Set first attribute for this bodypart to the y, x coordinates element wise.
        self._viterbi_frames[self._current_frame][bodypart * 2] = np.transpose(coords)
        # Get the probabilities and offsets of the first frame and store them...
        prob = scmap.get_prob_table(frame, bodypart)[coords]
        off_x, off_y = np.zeros(np.transpose(coords).shape) if(scmap.get_offset_map() is None) else np.transpose(
                       scmap.get_offset_map()[frame, coords[0], coords[1], bodypart])
        self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = np.transpose((self.log(prob), off_x, off_y, prob))


    def _compute_normal_frame(self, bodypart: int, frame: int, scmap: TrackingData):
        """ Computes and inserts a frame that has occurred after the first data-full frame. """
        # Get coordinates for all above threshold probabilities in this frame...
        coords = np.nonzero(scmap.get_prob_table(frame, bodypart) > self.THRESHOLD)

        # If no points are found above the threshold, frame for this bodypart is a dud,
        # copy prior frame probabilities... Also set all old_probabilities to 0 this point can't
        # be plotted...
        if (len(coords[0]) == 0):
            self._viterbi_frames[self._current_frame][bodypart * 2] = (
                np.copy(self._viterbi_frames[self._current_frame - 1][bodypart * 2])
            )
            self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = (
                np.copy(self._viterbi_frames[self._current_frame - 1][bodypart * 2 + 1])
            )
            self._viterbi_frames[self._current_frame][(bodypart * 2) + 1][:, 3] = 0
            return

        # Otherwise:
        
        # Set coordinates for this frame
        self._viterbi_frames[self._current_frame][bodypart * 2] = np.transpose(coords)

        # Get the x and y locations for the points in this frame and the prior frame...
        cy, cx = coords
        py, px = self._viterbi_frames[self._current_frame - 1][bodypart * 2].transpose()

        # Get offset values
        off_x, off_y = np.zeros(np.array(coords).shape) if(scmap.get_offset_map() is None) else np.transpose(
                       scmap.get_offset_map()[frame, coords[0], coords[1], bodypart])
        # Grab current non-viterbi probabilities...
        current_prob = scmap.get_prob_table(frame, bodypart)[coords]
        # Grab the prior viterbi probabilities
        prior_vit_probs = self._viterbi_frames[self._current_frame - 1][bodypart * 2][:, 0]

        # Perform viterbi computation and set this frame to the final viterbi frame...
        viterbi_vals = (np.expand_dims(self.log(current_prob), axis=1) + np.expand_dims(prior_vit_probs, axis=0) +
                       self._gaussian_values_at(cx, cy, px, py))
        self._viterbi_frames[self._current_frame][bodypart * 2 + 1] = np.transpose((np.max(viterbi_vals, axis=1),
                                                                                 off_x, off_y, current_prob))

    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        """ Handles Forward part of the viterbi algorithm, allowing for faster post processing. """
        # If gaussian_table is none, we have just started, initialize all variables...
        if(self._gaussian_table is None):
            # Precompute gaussian...
            self._compute_gaussian_table(scmap.get_frame_width(), scmap.get_frame_height())
            # Create empty python list for first frame.
            self._viterbi_frames[self._current_frame] = [None] * (len(self._bodyparts) * 2)
            # Set down scaling.
            self._down_scaling = scmap.get_down_scaling()

            for bp in range(scmap.get_bodypart_count()):
                self._compute_init_frame(bp, 0, scmap)

            # Remove first frame from source map, so we can compute the rest as normal...
            scmap.set_source_map(scmap.get_source_map()[1:])
            self._current_frame += 1

        # Continue on to main loop...
        for frame in range(scmap.get_frame_count()):

            # Create a frame...
            self._viterbi_frames[self._current_frame] = [None] * (len(self._bodyparts) * 2)

            for bp in range(scmap.get_bodypart_count()):
                # If the prior frame was a dud frame and all frames before it where dud frames, try initializing
                # on this frame... Note in a dud frame all points are below threshold...
                if(self._viterbi_frames[self._current_frame - 1][bp * 2] is None):
                    self._compute_init_frame(bp, frame, scmap)
                # Otherwise we can do full viterbi on this frame...
                else:
                    self._compute_normal_frame(bp, frame, scmap)

            # Increment frame counter
            self._current_frame += 1

        # Still processing frames, return None to indicate that...
        return None


    def _get_prior_frame(self, prior_frame: List[Union[ndarray, None]], current_point: Tuple[int, int, float]) -> Union[Tuple[int, Tuple[int, int, float]], Tuple[None, None]]:
        """
        Performs the viterbi back computation, given prior frame and current predicted point,
        returns the predicted point for this frame... (for single bodypart...)
        """
        # If the point data is none, return None
        if((prior_frame[0] is None)):
            return None, None

        cx, cy, cprob = current_point
        prior_viterbi = (cprob + self._gaussian_values_at(np.array(cx), np.array(cy), prior_frame[0][:, 1],
                                                          prior_frame[0][:, 0]).flatten() + prior_frame[1][:, 0])

        max_loc: int = np.argmax(prior_viterbi)

        return (max_loc, (prior_frame[0][max_loc][1], prior_frame[0][max_loc][0], prior_frame[1][max_loc][0]))


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        """ Handles backward part of viterbi, and then returns the poses """
        # Counter to keep track of current frame...
        r_counter = self._num_frames - 1
        # To eventually store all poses
        all_poses = Pose.empty_pose(self._num_frames, len(self._bodyparts))
        # Points of the 'prior' frame (really the current frame)
        prior_points: List[Tuple[int, int, float]] = []

        # Initial frame...
        for bp in range(len(self._bodyparts)):
            # If point data is None, throw error because entire video has no plotting data then...
            # This should never happen....
            if(self._viterbi_frames[r_counter][(bp * 2)] is None):
                raise ValueError("All frames contain zero points!!! No actual tracking data!!!")

            # Get the max location index
            max_loc = np.argmax(self._viterbi_frames[r_counter][(bp * 2) + 1][:, 0])

            # Gather all required fields...
            y, x = self._viterbi_frames[r_counter][(bp * 2)][max_loc]
            prob = self._viterbi_frames[r_counter][(bp * 2) + 1][max_loc, 0]
            off_x, off_y = self._viterbi_frames[r_counter][(bp * 2) + 1][max_loc, 1:3]
            output_prob = self._viterbi_frames[r_counter][(bp * 2) + 1][max_loc, 3]

            # Append point to prior points and also add it the the poses object...
            prior_points.append((x, y, prob))
            all_poses.set_at(r_counter, bp, (x, y), (off_x, off_y), output_prob, self._down_scaling)

            # Drop the counter by 1
            r_counter -= 1
            progress_bar.update()

        # Entering main loop...
        while(r_counter >= 0):
            # Create a variable to store current points, which will eventually become the prior points...
            current_points: List[Tuple[int, int, float]] = []

            for bp in range(len(self._bodyparts)):
                # Run single step of backtrack....
                max_loc, max_point = self._get_prior_frame(
                    [self._viterbi_frames[r_counter][bp * 2], self._viterbi_frames[r_counter][bp * 2 + 1]],
                    prior_points[bp]
                )
                # If this point is None, copy prior_point, output pose of (0, 0) with probability of 0 and continue.
                if(max_loc is None):
                    current_points.append(prior_points[bp])
                    all_poses.set_at(r_counter, bp, (0, 0), (0, 0), 0, 0)

                    continue

                # Add the max point to current points...
                current_points.append(max_point)
                # Add current max to pose object...
                coord_data, prob_data = self._viterbi_frames[r_counter][(bp * 2):(bp * 2) + 2]

                all_poses.set_at(r_counter, bp, tuple(reversed(coord_data[max_loc])), prob_data[max_loc, 1:3],
                                 prob_data[max_loc, 3], self._down_scaling)

            # Set prior_points to current_points...
            prior_points = current_points
            # Decrement the counter
            r_counter -= 1
            progress_bar.update()

        # Return the poses...
        return all_poses


    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("norm_dist", "The normal distribution of the 2D gaussian curve used \n"
                          "for transition probabilities by the viterbi algorithm.", 3),
            ("amplitude", "The amplitude of the gaussian curve used by the viterbi algorithm.", 1),
            ("lowest_gaussian_value", "The lowest value of the gaussian curve used by the viterbi algorithm. \n"
                                      "Really a constant that is added on the the 2D gaussian to give all points\n"
                                      "a minimum probability.", 0),
            ("threshold", "The minimum floating point value a pixel within the probability frame must have \n"
                          "in order to be kept and added to the sparse matrix.", 0.001)
        ]

    @staticmethod
    def get_name() -> str:
        return "fastviterbi"

    @staticmethod
    def get_description() -> str:
        return ("A predictor that applies the Viterbi algorithm to frames in order to predict poses. \n"
                "The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but \n"
                "is also more memory intensive and computationally expensive. This specific implementation \n"
                "uses sparse matrix multiplication(log addition) for massive speedup over the normal \n"
                "viterbi implementation...")

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None
