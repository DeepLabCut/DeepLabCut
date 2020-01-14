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
from collections import deque

# TODO: Add more test methods, disable numpy warnings....
# TODO: Add Concept of being "In the Ground..."
# TODO: Outputing 0.99 for first frame, I think it is an error in the forward compute, although may be bug in back compute

class FastViterbi(Predictor):
    """
    A predictor that applies the Viterbi algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive. This specific implementation
    uses sparse matrix multiplication for massive speedup over the normal
    viterbi implementation...)
    """
    # The amount of side block increase for the normal distribution to increase by 1...
    ND_UNIT_PER_SIDE_COUNT = 10

    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int, settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        """ Initialized a fastviterbi plugin for analyzing a video """
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)

        # Store bodyparts and num_frames for later use, they become useful
        self._bodyparts = bodyparts
        self._num_frames = num_frames

        # Will hold the viterbi frames on the forward compute...
        # Dimension are: Frame -> Bodypart * 2 -> (y, x), (probability, loc_off_x, loc_off_y, old probability)
        # The x, y are split from other values because they are integers while others are floats...
        self._viterbi_frames: List[List[Union[ndarray, None]]] = [None] * num_frames

        # Stashes edge to edge probabilities, in the form delta-edge-index -> gaussian...
        self._edge_edge_table: ndarray = None
        # Stores actual edge values
        self._edge_vals: ndarray = None
        self._edge_coords: ndarray = None

        # Represents precomputed gaussian values... Stored in the form (delta_y, delta_x) -> gaussian value.
        self._gaussian_table: ndarray = None
        # Stores stride used in this video...
        self._down_scaling = None
        # Keeps track of current frame...
        self._current_frame = 0

        # Precomputed table for computing negative impacts of body parts
        self._neg_gaussian_table = None

        # Values for the gaussian formula, can be adjusted in dlc_config for differing results...
        self.NORM_DIST_UNSCALED = settings["norm_dist"]  # The normal distribution
        self.NORM_DIST = None # To be computed below...
        self.AMPLITUDE = settings["amplitude"]  # The amplitude, or height of the gaussian curve
        self.LOWEST_VAL = settings["lowest_gaussian_value"]  # Changes the lowest value that the gaussian curve can produce
        self.THRESHOLD = settings["threshold"] # The threshold for the matrix... Everything below this value is ignored.
        self.EDGE_PROB = settings["edge_probability"] # DLC "Predicted" going off screen probability.
        self.BLOCKS_PER_EDGE = settings["edge_blocks_per_side"] # Number of blocks to allocate per edge....
        self._edge_block_value = self.EDGE_PROB / (self.BLOCKS_PER_EDGE * 4) # Probability value for every block...

        # More global variables, can also be set in dlc_config...
        self.NEGATE_ON = settings["negate_overlapping_predictions"] # Enables prior body part negation...
        self.NEG_NORM_DIST_UNSCALED = settings["negative_impact_distance"] # Normal distribution of negative 2D gaussian curve
        self.NEG_NORM_DIST = None # To be computed below....
        self.NEG_AMPLITUDE = settings["negative_impact_factor"] # Negative amplitude to use for 2D negation gaussian

        # Used for keeping track of the last n - 1 bodypart, used for bodypart negation during forward compute
        if(self.NEGATE_ON):
            self._bp_stack = deque(maxlen=len(bodyparts) - 1)

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

    def _neg_gaussian_formula(self, prior_x: float, x: float, prior_y: float, y: float) -> float:
        """
        Private method, computes location of point (x, y) on a inverted gaussian curve given a prior point
        (x_prior, y_prior) to use as the center.

        :param prior_x: The x point in the prior frame
        :param x: The current frame's x point
        :param prior_y: The y point in the prior frame
        :param y: The current frame's y point
        :return: The location of x, y given inverted gaussian curve (1 - curve) centered at x_prior, y_prior
        """
        # Formula for 2D inverted gaussian curve (or dip)
        inner_x_delta = ((prior_x - x) ** 2) / (2 * self.NEG_NORM_DIST ** 2)
        inner_y_delta = ((prior_y - y) ** 2) / (2 * self.NEG_NORM_DIST ** 2)
        return 1 - (self.NEG_AMPLITUDE * np.exp(-(inner_x_delta + inner_y_delta)))

    def _compute_gaussian_table(self, width: int, height: int) -> None:
        """
        Compute the gaussian table given the width and height of each probability frame. Results are stored in
        self._gaussian_table
        """
        # Compute the normal distribution based on how many blocks per frame there are...
        self.NORM_DIST = (np.sqrt(width * height) / self.ND_UNIT_PER_SIDE_COUNT) * self.NORM_DIST_UNSCALED
        # Allocate gaussian table of width x height...
        self._gaussian_table = np.zeros((height + 2, width + 2), dtype="float32")

        # Iterate through filling in the values, using (0, 0) as the prior coordinate
        for y in range(height + 2):
            for x in range(width + 2):
                self._gaussian_table[y, x] = self._gaussian_formula(0, x, 0, y)

        # Done, return...
        return


    def _compute_neg_gaussian_table(self, width: int, height: int) -> None:
        """
        Computes the precomputed inverted 2D gaussian curve used for providing negative impacts at prior predicted
        bodyparts. Stored in self._neg_gaussian_table.
        """
        # Scale normal distribution based on size of frames...
        self.NEG_NORM_DIST = (np.sqrt(width * height) / self.ND_UNIT_PER_SIDE_COUNT) * self.NEG_NORM_DIST_UNSCALED
        # Allocate...
        self._neg_gaussian_table = np.zeros((height, width), dtype="float32")

        # Iterate and fill values
        for y in range(height):
            for x in range(width):
                self._neg_gaussian_table[y, x] = self._neg_gaussian_formula(0, x, 0, y)

        # Done
        return


    def _compute_edge_coordinates(self, width: int, height: int, num_of_blocks: int):
        """ Computes centered coordinates for edge blocks, which are used for gaussian computations... """
        self._edge_coords = np.zeros((2, num_of_blocks * 4), dtype=int)

        # Used to get the midpoints of the blocks on a given side.....
        def get_midpoints(num_blocks, side_length):
            #      __________Block's starting point in list__________ + _____midway of a subsection_______
            return (np.arange(num_blocks) / num_blocks) * side_length + ((side_length / num_blocks) * 0.5)

        # Side 1 (Left...)
        self._edge_coords[0, 0:num_of_blocks] = -1 # X values
        self._edge_coords[1, 0:num_of_blocks] = get_midpoints(num_of_blocks, height) # Y Values

        # Side 2 (Bottom...)
        self._edge_coords[0, num_of_blocks:num_of_blocks * 2] = get_midpoints(num_of_blocks, width) # X values
        self._edge_coords[1, num_of_blocks:num_of_blocks * 2] = height # Y Values

        # Side 3 (Right...)
        self._edge_coords[0, num_of_blocks * 2:num_of_blocks * 3] = width # X values
        self._edge_coords[1, num_of_blocks * 2:num_of_blocks * 3] = get_midpoints(num_of_blocks, height) # Y Values

        # Side 4 (Top...)
        self._edge_coords[0, num_of_blocks * 3:num_of_blocks * 4] = get_midpoints(num_of_blocks, width) # X values
        self._edge_coords[1, num_of_blocks * 3:num_of_blocks * 4] = -1 # Y Values

        # Transpose so it's edge index -> x, y coordinate...
        self._edge_coords = self._edge_coords.transpose()

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
        # Compute delta x and y values. Use broadcasting to compute this for all current frames.
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

        # Set initial value for edge values...
        self._edge_vals[self._current_frame, bodypart, :] = self._edge_block_value

        # If no points are found above the threshold, frame for this bodypart is a dud, set it to none...
        if (len(coords[0]) == 0):
            self._viterbi_frames[self._current_frame][bodypart * 2] = None
            self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = None
            return

        # Set first attribute for this bodypart to the y, x coordinates element wise.
        self._viterbi_frames[self._current_frame][bodypart * 2] = np.transpose(coords)

        # Get the probabilities and offsets of the first frame and store them...
        prob = scmap.get_prob_table(frame, bodypart)[coords] * (1 - self.EDGE_PROB)
        off_x, off_y = np.zeros(np.array(coords).shape) if(scmap.get_offset_map() is None) else np.transpose(
                       scmap.get_offset_map()[frame, coords[0], coords[1], bodypart])
        self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = np.transpose((prob, off_x, off_y))


    def _compute_normal_frame(self, bodypart: int, frame: int, scmap: TrackingData):
        """ Computes and inserts a frame that has occurred after the first data-full frame. """
        # Get coordinates for all above threshold probabilities in this frame...
        coords = np.nonzero(scmap.get_prob_table(frame, bodypart) > self.THRESHOLD)

        # SPECIAL CASE: NO IN-FRAME VITERBI VALUES THAT MAKE IT ABOVE THRESHOLD...
        if (len(coords[0]) == 0):
            # In this special case, we just copy the prior frame data...
            self._edge_vals[self._current_frame, bodypart, :] = self._edge_vals[self._current_frame - 1, bodypart, :]

            self._viterbi_frames[self._current_frame][bodypart * 2] = (
                np.copy(self._viterbi_frames[self._current_frame - 1][bodypart * 2])
            )
            self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = (
                np.copy(self._viterbi_frames[self._current_frame - 1][bodypart * 2 + 1])
            )
            return

        # NORMAL CASE:

        # Get the x and y locations for the points in this frame and the prior frame...
        cy, cx = coords
        py, px = self._viterbi_frames[self._current_frame - 1][bodypart * 2].transpose()

        # Get offset values
        off_x, off_y = np.zeros(np.array(coords).shape) if(scmap.get_offset_map() is None) else np.transpose(
                       scmap.get_offset_map()[frame, coords[0], coords[1], bodypart])
        # Grab current non-viterbi probabilities...
        current_prob = scmap.get_prob_table(frame, bodypart)[coords] * (1 - self.EDGE_PROB)
        # Grab the prior viterbi probabilities
        prior_vit_probs = self._viterbi_frames[self._current_frame - 1][(bodypart * 2) + 1][:, 0]


        # Get all of the same data for the edge values...
        edge_x, edge_y = self._edge_coords[:, 0], self._edge_coords[:, 1]
        prior_edge_probs = self._edge_vals[self._current_frame - 1, bodypart, :]
        current_edge_probs = np.array([self._edge_block_value] * (self.BLOCKS_PER_EDGE * 4))

        # COMPUTE IN-FRAME VITERBI VALUES:
        # Compute probabilities of transferring from the prior frame to this frame...
        frame_to_frame = (np.expand_dims(current_prob, axis=1) * np.expand_dims(prior_vit_probs, axis=0)
                          * self._gaussian_values_at(cx, cy, px, py))
        # Compute the probabilities of transferring from the prior edge to this frame.
        edge_to_frame = (np.expand_dims(current_prob, axis=1) * self._gaussian_values_at(cx, cy, edge_x, edge_y)
                         * np.expand_dims(prior_edge_probs, axis=0))
        # Merge probabilities of going from the edge to the frame or frame to frame, selecting the max of the two for
        # each point in this frame.
        viterbi_vals = np.maximum(np.max(frame_to_frame, axis=1), np.max(edge_to_frame, axis=1))

        # COMPUTE OFF-SCREEN VITERBI VALUES:
        # Compute the probability of transitioning from the prior frame to the current edge.....
        frame_to_edge = (np.expand_dims(current_edge_probs, axis=1) * self._gaussian_values_at(edge_x, edge_y, px, py)
                         * np.expand_dims((prior_vit_probs), axis=0))
        # Compute the probability of transitioning from the prior edge to the current edge...
        edge_to_edge = (np.expand_dims(current_edge_probs, axis=1) * np.expand_dims(prior_edge_probs, axis=0)
                        * self._gaussian_values_at(edge_x, edge_y, edge_x, edge_y))
        # Merge probabilities to produce final edge transitioning viterbi values...
        edge_vit_vals = np.maximum(np.max(frame_to_edge, axis=1), np.max(edge_to_edge, axis=1))

        # NORMALIZE PROBABILITIES:
        # If we were to leave the probabilities as-is then we would experience floating point underflow after enough
        # frames, so we normalize the probabilities before storing them to avoid this issue. Since relations and scaling
        # between the probabilities is still the same, normalizing has no effect on the results...
        total_sum = np.sum(edge_vit_vals) + np.sum(viterbi_vals)
        viterbi_vals = viterbi_vals / total_sum
        edge_vit_vals = edge_vit_vals / total_sum

        # POST FILTER PHASE:

        # Filter out any zero or NaN values from our matrix:
        post_filter = (~np.isnan(viterbi_vals)) | (viterbi_vals > self.THRESHOLD)
        coords = coords[0][post_filter], coords[1][post_filter]
        viterbi_vals = viterbi_vals[post_filter]

        if (len(coords[0]) == 0):
            # In this special case, we just copy the prior frame data...
            self._edge_vals[self._current_frame, bodypart, :] = self._edge_vals[self._current_frame - 1, bodypart, :]

            self._viterbi_frames[self._current_frame][bodypart * 2] = (
                np.copy(self._viterbi_frames[self._current_frame - 1][bodypart * 2])
            )
            self._viterbi_frames[self._current_frame][(bodypart * 2) + 1] = (
                np.copy(self._viterbi_frames[self._current_frame - 1][bodypart * 2 + 1])
            )
            return

        # Set coordinates for this frame
        self._viterbi_frames[self._current_frame][bodypart * 2] = np.transpose(coords)

        # SAVE NEW VITERBI FRAMES:
        self._edge_vals[self._current_frame, bodypart] = edge_vit_vals
        self._viterbi_frames[self._current_frame][bodypart * 2 + 1] = np.transpose((viterbi_vals, off_x, off_y))



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

            # Compute negative gaussian if negative impact setting is switched on...
            if(self.NEGATE_ON):
                self._compute_neg_gaussian_table(scmap.get_frame_width(), scmap.get_frame_height())

            # Adjust the blocks-per edge to avoid having it greater then the length of one of the sides of the frame.
            self.BLOCKS_PER_EDGE = min(scmap.get_frame_height(), scmap.get_frame_width(), self.BLOCKS_PER_EDGE)
            self._edge_block_value = self.EDGE_PROB / (self.BLOCKS_PER_EDGE * 4)  # Must be recomputed...

            # Create off edge point table of gaussian values for off-edge/on-edge transitions...
            self._compute_edge_coordinates(scmap.get_frame_width(), scmap.get_frame_height(), self.BLOCKS_PER_EDGE)
            self._edge_vals = np.zeros((self._num_frames, len(self._bodyparts), self.BLOCKS_PER_EDGE * 4), dtype="float32")

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


    def _get_prior_location(self, prior_frame: List[Union[ndarray, None]], prior_edge_probs: ndarray, current_point: Tuple[int, int, float]) -> Union[Tuple[bool, int, Tuple[int, int, float]], Tuple[None, None, None]]:
        """
        Performs the viterbi back computation, given prior frame and current predicted point,
        returns the predicted point for this frame... (for single bodypart...)
        """
        # If the point data is none, return None
        if((prior_frame[0] is None) or (current_point[0] is None)):
            return None, None, None

        # Unpack the point
        cx, cy, cprob = current_point
        # Get prior frame points/probabilities....
        px, py, pprob = prior_frame[0][:, 1], prior_frame[0][:, 0], prior_frame[1][:, 0]
        # Get prior edge points/probabilities....
        edge_x, edge_y = self._edge_coords[:, 0], self._edge_coords[:, 1]

        # Compute probability of going from current point to some where in the prior frame...
        prior_frame_viterbi = (cprob * self._gaussian_values_at(np.array(cx), np.array(cy), px, py).flatten() * pprob)

        # Compute the probability of going from the current point to somewhere outside of the prior frame...
        prior_edge_viterbi = (cprob * self._gaussian_values_at(np.array(cx), np.array(cy), edge_x, edge_y).flatten()
                              * prior_edge_probs)

        # Get max probability in list of possible transitions, for the frame and edge... Also normalize them...
        max_frame_i: int = np.argmax(prior_frame_viterbi)
        max_edge_i: int = np.argmax(prior_edge_viterbi)

        # Compute total sum of probabilities so we can normalize when returning values...
        total_sum = np.sum(prior_frame_viterbi) + np.sum(prior_edge_viterbi)

        # If the in frame selection is greater, return it, otherwise return edge prediction
        if(prior_frame_viterbi[max_frame_i] > prior_edge_viterbi[max_edge_i]):
            return (True, max_frame_i, (px[max_frame_i], py[max_frame_i], prior_frame_viterbi[max_frame_i] / total_sum))
        else:
            return (False, max_edge_i, (edge_x[max_edge_i], edge_y[max_edge_i],
                    prior_edge_viterbi[max_edge_i] / total_sum))


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        """ Handles backward part of viterbi, and then returns the poses """
        # Counter to keep track of current frame...
        r_counter = self._num_frames - 1
        # To eventually store all poses
        all_poses = Pose.empty_pose(self._num_frames, len(self._bodyparts))
        # Points of the 'prior' frame (really the current frame)
        prior_points: List[Tuple[int, int, float]] = []
        # Keeps track last body part count - 1 body parts...
        bp_queue = deque(maxlen=(len(self._bodyparts) - 1))

        # Initial frame...
        for bp in range(len(self._bodyparts)):
            # If point data is None, throw error because entire video has no plotting data then...
            # This should never happen....
            if(self._viterbi_frames[r_counter][(bp * 2)] is None):
                raise ValueError("All frames contain zero points!!! No actual tracking data!!!")

            viterbi_data = self._viterbi_frames[r_counter][(bp * 2) + 1][:, 0]
            coord_y, coord_x = self._viterbi_frames[r_counter][(bp * 2)][:].transpose()

            # Perform negation of prior body parts if enabled
            if(self.NEGATE_ON):
                for bpx, bpy, bpprob in bp_queue:
                    if (bpx is None):
                        continue
                    viterbi_data = viterbi_data * self._neg_gaussian_table[np.abs(coord_y - bpy), np.abs(coord_x - bpx)]

            # Get the max location index
            max_frame_loc = np.argmax(viterbi_data)
            max_edge_loc = np.argmax(self._edge_vals[r_counter, bp])

            # Gather all required fields...
            y, x = self._viterbi_frames[r_counter][(bp * 2)][max_frame_loc]
            off_x, off_y = self._viterbi_frames[r_counter][(bp * 2) + 1][max_frame_loc][1:]
            prob = viterbi_data[max_frame_loc]
            edge_x, edge_y = self._edge_coords[max_edge_loc]
            edge_prob = self._edge_vals[r_counter, bp, max_edge_loc]

            # If the edge is greater then the max point in frame, set prior point to (-1, -1) and pose output
            # probability to 0.
            if(prob < edge_prob):
                prior_points.append((edge_x, edge_y, edge_prob))
                all_poses.set_at(r_counter, bp, (-1, -1), (0, 0), 0, 1)

                if(self.NEGATE_ON):
                    bp_queue.append((None, None, None))
            else:
                # Normalize the viterbi probability...
                normalized_prob = prob / (np.sum(self._edge_vals[r_counter, bp]) + np.sum(viterbi_data))

                # Append point to prior points and also add it the the poses object...
                prior_points.append((x, y, normalized_prob))
                all_poses.set_at(r_counter, bp, (x, y), (off_x, off_y), normalized_prob, self._down_scaling)

                if(self.NEGATE_ON):
                    bp_queue.append((x, y, normalized_prob))

        # Drop the counter by 1
        r_counter -= 1
        progress_bar.update()

        # Entering main loop...
        while(r_counter >= 0):
            # Create a variable to store current points, which will eventually become the prior points...
            current_points: List[Tuple[int, int, float]] = []

            for bp in range(len(self._bodyparts)):
                # Run single step of backtrack....
                viterbi_data = self._viterbi_frames[r_counter][bp * 2 + 1][:]
                coord_y, coord_x = self._viterbi_frames[r_counter][bp * 2][:].transpose()

                # If negate switch is on, perform negation of all prior bodyparts from this one...
                if(self.NEGATE_ON):
                    for bpx, bpy, bpprob in bp_queue:
                        if(bpx is None):
                            continue
                        viterbi_data[:, 0] = viterbi_data[:, 0] * self._neg_gaussian_table[
                                             np.abs(coord_y - bpy), np.abs(coord_x - bpx)]


                is_in_frame, max_loc, max_point = self._get_prior_location(
                    [self._viterbi_frames[r_counter][bp * 2], viterbi_data],
                    self._edge_vals[r_counter, bp],
                    prior_points[bp]
                )
                # If point is None, plot an unplotable, copy prior point, continue
                if(is_in_frame is None):
                    all_poses.set_at(r_counter, bp, (-1, -1), (0, 0), 0, 1)
                    current_points.append((None, None, None))
                    if(self.NEGATE_ON):
                        bp_queue.append((None, None, None))
                    continue


                # Based on if the point is in frame or not, decide how to plot it.
                if(is_in_frame):
                    max_x, max_y, max_normalized_prob = max_point
                    off_x, off_y = self._viterbi_frames[r_counter][(bp * 2) + 1][max_loc, 1:]

                    all_poses.set_at(r_counter, bp, (max_x, max_y), (off_x, off_y),
                                     max_normalized_prob,
                                     self._down_scaling)

                    if (self.NEGATE_ON):
                        bp_queue.append(max_point)
                else:
                    all_poses.set_at(r_counter, bp, (-1, -1), (0, 0), 0, 1)

                    if (self.NEGATE_ON):
                        bp_queue.append((None, None, None))

                # Append point to current points....
                current_points.append(max_point)

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
            ("norm_dist", "The normal distribution of the 2D gaussian curve used"
                          "for transition probabilities by the viterbi algorithm.", 1),
            ("amplitude", "The amplitude of the gaussian curve used by the viterbi algorithm.", 1),
            ("lowest_gaussian_value", "The lowest value of the gaussian curve used by the viterbi algorithm."
                                      "Really a constant that is added on the the 2D gaussian to give all points"
                                      "a minimum probability.", 0),
            ("threshold", "The minimum floating point value a pixel within the probability frame must have"
                          "in order to be kept and added to the sparse matrix.", 0.001),
            ("edge_probability", "A constant float between 0 and 1 that determines the probability that a point goes"
                                 "off the screen. This probability is divided among edge blocks", 0.2),
            ("edge_blocks_per_side", "Number of edge blocks to have per side.", 4),
            ("negate_overlapping_predictions", "If enabled, predictor will discourage a bodypart from being in the same"
                                               "location as prior predicted body parts.", True),
            ("negative_impact_factor", "The height of the upside down 2D gaussian curve used for negating locations"
                                       "of prior predicted body parts.", 0.99),
            ("negative_impact_distance", "The normal distribution of the 2D gaussian curve used for negating locations"
                                         "of prior predicted body parts.", 0.5)
        ]

    @staticmethod
    def get_name() -> str:
        return "fastviterbi"

    @staticmethod
    def get_description() -> str:
        return ("A predictor that applies the Viterbi algorithm to frames in order to predict poses. "
                "The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but "
                "is also more memory intensive and computationally expensive. This specific implementation "
                "uses sparse matrix multiplication for massive speedup over the normal "
                "viterbi implementation...")

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return [cls.test_plotting]


    @classmethod
    def test_plotting(cls) -> Tuple[bool, str, str]:
        # Make tracking data...
        track_data = TrackingData.empty_tracking_data(4, 1, 3, 3, 2)

        track_data.set_prob_table(0, 0, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        track_data.set_prob_table(1, 0, np.array([[0, 1, 0], [0, 0.5, 0], [0, 0, 0]]))
        track_data.set_prob_table(2, 0, np.array([[1, 0.5, 0], [0, 0, 0], [0, 0, 0]]))
        track_data.set_prob_table(3, 0, np.array([[0.5, 0, 0], [1, 0, 0], [0, 0, 0]]))

        # Note that probabilities are scaled down due to being adjusted to equal 1 with out of bounds value included...
        expected_result = [[3, 3, 0.6621716], [3, 1, 0.6621716], [1, 1, 0.6621716], [1, 3, 0.7]]

        # Make the predictor...
        predictor = cls(["part1"], 1, track_data.get_frame_count(), {name:val for name, desc, val in cls.get_settings()}, None)

        # Pass it data...
        predictor.on_frames(track_data)

        # Check output
        poses = predictor.on_end(tqdm.tqdm(total=4)).get_all()

        if(np.allclose(poses, expected_result)):
            return (True, "\n" + str(expected_result), "\n" + str(np.array(poses)))
        else:
            return (False, "\n" + str(expected_result), "\n" + str(np.array(poses)))

    @classmethod
    def supports_multi_output(cls) -> bool:
        return False

