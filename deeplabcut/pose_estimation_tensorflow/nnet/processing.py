"""
Contains Abstract Base Class for all predictor plugins, which when provided probability frames from the neural net,
figure out where the points should be in the image. They are executed when deeplabcut.analyze_videos is run with
"predictor" argument set to a valid plugin name...
"""
# Abstract class stuff
from abc import ABC
from abc import abstractmethod

# Used for type hints
from numpy import ndarray
from typing import List, Union, Type, Tuple, Set, Sequence, Dict, Any, Callable
import tqdm

# Used by get_predictor for loading plugins
from deeplabcut.pose_estimation_tensorflow.util import pluginloader
from deeplabcut.pose_estimation_tensorflow.nnet import predictors
import inspect

import numpy as np


class TrackingData:
    """
    Represents tracking data received from the DeepLabCut neural network. Includes a source map of probabilities,
    the predicted location offsets within the source map, stride info and ect. Also provides many convenience methods
    for working with and getting info from the DLC neural network data.
    """

    """ The default image down scaling used by DeepLabCut """
    DEFAULT_SCALE: int = 8


    def __init__(self, scmap: ndarray, locref: Union[None, ndarray] = None, stride: int = DEFAULT_SCALE):
        """
        Create an new tracking data object to store DLC neural network data for one frame or a batch of frames.

        :param scmap: The probability maps produced by the neural network, a 4-dimensional numpy array containing the
                      dimensions: [frame, y location, x location, body part].
        :param locref: The "offsets" produced by DeepLabCut neural network, stored in a 5-dimensional numpy array
                       containing the dimensions:
                       [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        :param stride: Integer which stores the down scaling of the probability map relative to the size of the original
                       video. This value defaults to 8, meaning the original video is 8 times the size of the
                       probability map.
        """
        # If scmap received is only 3-dimension, it is of only 1 frame, so add the batch dimension so it works better.
        if(len(scmap.shape) == 3):
            self._scmap = np.expand_dims(scmap, axis=0)
        else:
            self._scmap = scmap

        if((locref is not None) and (len(locref.shape) == 3)):
            self._locref = np.expand_dims(locref, axis=0)
        else:
            self._locref = locref

        self._scaling = stride


    @classmethod
    def empty_tracking_data(cls, frame_amt: int, part_count: int, width: int, height: int,
                            stride: int = DEFAULT_SCALE) -> "TrackingData":
        """
        Create a new empty tracking data object with space allocated to fit the specified sizes of data.

        :param frame_amt: The amount of probability map frames to allocate space for, an Integer.
        :param part_count: The amount of body parts per frame to allocate space for, an Integer.
        :param width: The width of each probability frame, an Integer.
        :param height: The height of each probability frame, an Integer.
        :param stride: The downscaling of the probability frame relative to the original video, defaults to 8,
                       meaning the original video is 8 times the size of the probability map.
        :return: A tracking data object full of zeroes.
        """
        return cls(np.zeros((frame_amt, height, width, part_count), dtype="float32"), None, stride)


    def get_source_map(self) -> ndarray:
        """
        Gets the raw probability source map of this tracking data.

        :return: A numpy array representing the source probability map of this tracking data. It is a 4-dimensional
                array containing the dimensions: [frame, y location, x location, body part] -> probability
        """
        return self._scmap

    def get_offset_map(self) -> Union[None, ndarray]:
        """
        Returns the offset prediction map representing offset predictions for each location in the probability map.

        :return: A numpy array representing the predicted offsets for each location within the probability map, or
                 None if this TrackingData doesn't have offset predictions...
                 Indexing: [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        """
        return self._locref

    def get_down_scaling(self) -> int:
        """
        Get the down scaling performed on this source map, as an integer.

        :return: An integer representing the downscaling of the probability map compared to the original video file, in
                 terms of what the dimensions of the probability map need to multiplied by to match the dimensions of
                 the original video file.
        """
        return self._scaling

    def set_source_map(self, scmap: ndarray):
        """
        Set the raw probability map of this tracking data.

        :param scmap: A numpy array representing the probability map of this tracking data. It is a 4-dimensional
                      array containing the dimensions: [frame, y location, x location, body part]
        """
        self._scmap = scmap

    def set_offset_map(self, locref: Union[None, ndarray]):
        """
        Set the offset prediction map representing offset predictions for each location in the probability map.

        :param locref: A numpy array representing the predicted offsets within the probability map. Can also
                       be set to None to indicate this TrackingData doesn't have or support higher precision
                       predictions.
                       Dimensions: [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        """
        self._locref = locref

    def set_down_scaling(self, scale: int):
        """
        Set the down scaling performed on the probability map, as an integer

        :param scale: An integer representing the downscaling of the probability map compared to the original video file, in
                 terms of what the dimensions of the probability map need to multiplied by to match the dimensions of
                 the original video file.
        """
        self._scaling = scale

    def get_max_scmap_points(self, num_max: int = 1) -> Tuple[ndarray, ndarray]:
        """
        Get the locations with the max probabilities for each frame in this TrackingData.

        :param num_max: Specifies the number of maximums to grab for each body part from each frame. Defaults to 1.

        :return: A tuple of numpy arrays, the first numpy array being the y index for the max of each frame, the second
                 being the x index for the max of each frame.
                 Dimensions: [1 for x and 0 for y index, frame, body part] -> index
        """
        batchsize, ny, nx, num_joints = self._scmap.shape
        scmap_flat = self._scmap.reshape((batchsize, nx * ny, num_joints))

        if(num_max <= 1):
            scmap_top = np.argmax(scmap_flat, axis=1)
        else:
            # Grab top values
            scmap_top = np.argpartition(scmap_flat, -num_max, axis=1)[:, -num_max:]
            for ix in range(batchsize):
                # Sort predictions for each body part from highest to least...
                vals = scmap_flat[ix, scmap_top[ix], np.arange(num_joints)]
                arg = np.argsort(-vals, axis=0)
                scmap_top[ix] = scmap_top[ix, arg, np.arange(num_joints)]
            # Flatten out the map so arrangement is:
            # [frame] -> [joint 1 prediction 1, joint 1 prediction 2, ... , joint 2 prediction 1, ... ]
            # Note this mimics single prediction format...
            scmap_top = scmap_top.swapaxes(1, 2).reshape(batchsize, num_max * num_joints)

        # Convert to x, y locations....
        return np.unravel_index(scmap_top, (ny, nx))


    def get_max_of_frame(self, frame: int, num_outputs: int = 1) -> Tuple[ndarray, ndarray]:
        """
        Get the locations of the highest probabilities for a single frame in the array.

        :param frame: The index of the frame to get the maximum of, in form of an integer.
        :param num_outputs: Specifies the number of maximums to grab for each body part from the frame. Defaults to 1.
        :return: A tuple of numpy arrays, the first numpy array being the y index of the max probability for each body
                 part in the frame, the second being the x index of the max probability for each body part in the frame
                 Indexing: [1 for x or 0 for y index] -> [bodypart 1, bodypart 1 prediction 2, ..., bodypart 2, ...]
        """
        y_dim, x_dim, num_joints = self._scmap.shape[1:4]
        scmap_flat = self._scmap[frame].reshape((y_dim * x_dim, num_joints))

        if(num_outputs <= 1):
            # When num_outputs is 1, we just grab the single maximum...
            flat_max = np.argmax(scmap_flat, axis=0)
        else:
            # When num_outputs greater then 1, use partition to get multiple maximums...
            scmap_top = np.argpartition(scmap_flat, -num_outputs, axis=0)[-num_outputs:]
            vals = scmap_flat[scmap_top, np.arange(num_joints)]
            arg = np.argsort(-vals, axis=0)
            flat_max = scmap_top[arg, np.arange(num_joints)].swapaxes(1, 2).reshape(num_outputs * num_joints)

        return np.unravel_index(flat_max, dims=(y_dim, x_dim))


    def get_poses_for(self, points: Tuple[ndarray, ndarray]):
        """
        Return a pose object for the "maximum" predicted indexes passed in.

        :param points: A tuple of 2 numpy arrays, one representing the y indexes for each frame and body part,
                       the other being the x indexes represented the same way. (Note 'get_max_scmap_points' returns
                       maximum predictions in this exact format).
        :return: The Pose object representing all predicted maximum locations for selected points...

        NOTE: This method detects when multiple predictions(num_outputs > 1) have been made and will still work
              correctly...
        """
        y, x = points
        # Create new numpy array to store probabilities, x offsets, and y offsets...
        probs = np.zeros(x.shape)

        # Get the number of predicted values for each joint for the passed maximums... We will divide the body part
        # index by this value in order to get the correct body part in this source map...
        num_outputs = x.shape[1] / self.get_bodypart_count()

        x_offsets = np.zeros(x.shape)
        y_offsets = np.zeros(y.shape)

        # Iterate the frame and body part indexes in x and y, we just use x since both are the same size
        for frame in range(x.shape[0]):
            for bp in range(x.shape[1]):
                probs[frame, bp] = self._scmap[frame, y[frame, bp], x[frame, bp], int(bp // num_outputs)]
                # Locref is frame -> y -> x -> bodypart -> relative coordinate pair offset. if it is None, just keep
                # all offsets as 0.
                if (self._locref is not None):
                    x_offsets[frame, bp], y_offsets[frame, bp] = self._locref[frame, y[frame, bp], x[frame, bp],
                                                                              int(bp // num_outputs)]

        # Now apply offsets to x and y to get actual x and y coordinates...
        # Done by multiplying by scale, centering in the middle of the "scale square" and then adding extra offset
        x = x.astype("float32") * self._scaling + (0.5 * self._scaling) + x_offsets
        y = y.astype("float32") * self._scaling + (0.5 * self._scaling) + y_offsets

        return Pose(x, y, probs)


    @staticmethod
    def _get_count_of(val: Union[int, slice, Sequence[int]], length: int) -> int:
        """ Internal private method to get length of an index selection(as in how many indexes it selects...) """
        if (isinstance(val, Sequence)):
            return len(val)
        elif (isinstance(val, slice)):
            start, stop, step = val.indices(length)
            return len(range(start, stop - 1, step))
        elif (isinstance(val, int)):
            return 1
        else:
            raise ValueError("Value is not a slice, integer, or list...")


    def get_prob_table(self, frame: Union[int, slice, Sequence[int]], bodypart: Union[int, slice, Sequence[int]]) -> ndarray:
        """
        Get the probability map for a selection of frames and body parts or a single frame and body part.

        :param frame: The frame index, as an integer or slice.
        :param bodypart: The body part index, as an integer or slice.
        :return: The probability map(s) for a single frame or selection of frames based on indexes, as a numpy array...
                 Dimensions: [frame, body part, y location, x location] -> probability
        """
        # Compute amount of frames and body parts selected....
        frame_count = self._get_count_of(frame, self.get_frame_count())
        part_count = self._get_count_of(bodypart, self.get_bodypart_count())

        # Return the frames, reshaped to be more "frame like"...
        slicer = self._scmap[frame, :, :, bodypart]
        
        # If the part_count is greater then one, move bodypart dimension back 2 in the dimensions
        if(part_count > 1 and frame_count > 1):
            return np.transpose(slicer, (0, 3, 1, 2))
        elif(part_count > 1):
            return np.transpose(slicer, (2, 0, 1))
        # Otherwise just return the slice...
        else:
            return slicer


    def set_prob_table(self, frame: Union[int, slice, Sequence[int]], bodypart: Union[int, slice, Sequence[int]],
                           values: ndarray):
        """
        Set the probability table for a selection of frames and body parts or a single frame and body part.

        :param frame: The frame index, as an integer or slice.
        :param bodypart: The body part index, as an integer or slice.
        :param values: The probability map(s) to set in this TrackingData object based on the frame and body parts
                       specified, as a numpy array...
                       Dimensions of values: [frame, body part, y location, x location] -> probability
        """
        # Compute amount of frames and body parts selected....
        frame_count = self._get_count_of(frame, self.get_frame_count())
        part_count = self._get_count_of(bodypart, self.get_bodypart_count())
        
        # If multiple body parts were selected, rearrange dimensions to match those used by the scmap...
        if(part_count > 1 and frame_count > 1):
            values = np.transpose(self._scmap[frame, :, :, bodypart], (0, 2, 3, 1))
        elif(part_count > 1):
            values = np.transpose(self._scmap[frame, :, :, bodypart], (1, 2, 0))
        
        # Set the frames, resizing the array to fit
        self._scmap[frame, :, :, bodypart] = values


    def get_frame_count(self) -> int:
        """
        Get the number of frames stored in this TrackingData object.

        :return: The number of frames stored in this tracking data object.
        """
        return self._scmap.shape[0]

    def get_bodypart_count(self) -> int:
        """
        Get the number of body parts stored in this TrackingData object per frame.

        :return: The number of body parts per frame as an integer.
        """
        return self._scmap.shape[3]

    def get_frame_width(self) -> int:
        """
        Return the width of each probability map in this TrackingData object.

        :return: The width of each probability map as an integer.
        """
        return self._scmap.shape[2]

    def get_frame_height(self) -> int:
        """
        Return the height of each probability map in this TrackingData object.

        :return: The height of each probability map as an integer.
        """
        return self._scmap.shape[1]

    # Used for setting single poses....
    def set_pose_at(self, frame: int, bodypart: int, scmap_x: int, scmap_y: int, pose_object: "Pose", output_num: int=0):
        """
        Set a pose in the specified Pose object to the specified x and y coordinate for a provided body part and frame.
        This method will use data from this TrackingData object to correctly set the information in the Pose object.

        :param frame: The specified frame to copy from this TrackingData to the Pose object and set.
        :param bodypart: The specified body part to copy from this TrackingData to the pose object and set
        :param scmap_x: The x index of this TrackingData to set the Pose prediction to.
        :param scmap_y: The y index of this TrackingData to set the Pose prediction to.
        :param pose_object: The pose object to be modified/copied to.
        :param output_num: The output number to set in the pose object (Which prediction for this bodypart?).
                           Should only be needed if num_outputs > 1. Defaults to 0, meaning the first prediction.
        :return: Nothing, changes stored in pose_object...
        """
        # Get probability...
        prob = self._scmap[frame, scmap_y, scmap_x, bodypart]
        # Default offsets to 0
        off_x, off_y = 0, 0

        # If we are actually using locref, set offsets to it
        if(self._locref is not None):
            off_y, off_x = self._locref[frame, scmap_y, scmap_x, bodypart]

        # Compute actual x and y values in the video...
        scmap_x = float(scmap_x) * self._scaling + (0.5 * self._scaling) + off_x
        scmap_y = float(scmap_y) * self._scaling + (0.5 * self._scaling) + off_y

        # Set values...
        pose_object.set_x_at(frame, bodypart + output_num, scmap_x)
        pose_object.set_y_at(frame, bodypart + output_num, scmap_y)
        pose_object.set_prob_at(frame, bodypart + output_num, prob)


class Pose:
    """
    Class defines the Poses for given amount of frames and body parts... Note that pose has no concept of multiple
    predictions for body part, but rather simply expects the multiple predictions to be stored side-by-side as
    multiple body parts... Also it should be noted that data is stored in terms of original video coordinates, not
    probability source map indexes.
    """
    def __init__(self, x: ndarray, y: ndarray, prob: ndarray):
        """
        Create a new Pose object, or batch of poses for frames.

        :param x: All x video coordinates for these poses, in ndarray indexing format frame -> body part -> x-value
        :param y: All y video coordinates for these poses, in ndarray indexing format frame -> body part -> y-value
        :param prob: All probabilities for these poses, in ndarray indexing format frame -> body part -> p-value
        """
        self._data = np.zeros((x.shape[0], x.shape[1] * 3))

        self.set_all_x(x)
        self.set_all_y(y)
        self.set_all_prob(prob)

    # Helper Constructor methods...

    @classmethod
    def empty_pose(cls, frame_count: int, part_count: int) -> "Pose":
        """
        Returns an empty pose object, or a pose object with numpy arrays full of zeros. It will have space for
        "frame_count" frames and "part_count" body parts.

        :param frame_count: The amount of frames to allocate space for in the underlying array, an Integer.
        :param part_count: The amount of body parts to allocate space for in the underlying array, an Integer.
        :return: A new Pose object.
        """
        return cls(np.zeros((frame_count, part_count)), np.zeros((frame_count, part_count)),
                   np.zeros((frame_count, part_count)))

    # Helper Methods

    def _fix_index(self, index: Union[int, slice], value_offset: int) -> Union[int, slice]:
        """
        Fixes slice or integer indexing received by user for body part to fit the actual way it is stored.
        PRIVATE METHOD! Should not be used outside this class, for internal index correction!

        :param index: An integer or slice representing indexing
        :param value_offset: An integer representing the offset of the desired values in stored data
        :return: Slice or integer, being the fixed indexing to actually get the body parts
        """
        if(isinstance(index, (int, np.integer))):
            # Since all data is all stored together, multiply by 3 and add the offset...
            return (index * 3) + value_offset
        elif(isinstance(index, slice)):
            # Normalize the slice and adjust the indexes.
            start, end, step = index.indices(self._data.shape[1] // 3)
            return slice((start * 3) + value_offset, (end * 3) + value_offset, step * 3)
        else:
            raise ValueError(f"Index is not of type slice or integer! It is type '{type(index)}'!")

    # Represents point data, is a tuple of x, y data where x and y are numpy arrays or integers...
    PointData = Tuple[Union[int, ndarray], Union[int, ndarray]]
    # Represents point data, is a tuple of x, y data where x and y are numpy arrays or floats...
    FloatPointData = Tuple[Union[float, ndarray], Union[float, ndarray]]
    # Represents and Index, either an integer or a slice
    Index = Union[int, slice]

    def set_at(self, frame: Index, bodypart: Index, scmap_coord: PointData, offset: Union[FloatPointData, None],
               prob: Union[float, ndarray], down_scale: int):
        """
        Set the probability data at a given location or locations to the specified data.

        :param frame: The index of the frame or frames to set, an integer or a slice.
        :param bodypart: The index of the bodypart or bodyparts to set, integer or a slice
        :param scmap_coord: The source map index to set this Pose's location to, specifically the index directly
                            selected from the downscaled source map stored in the TrackingData object. It is a tuple of
                            two integer or numpy arrays representing x and y coordinates...
        :param offset: The offset of the source map point once scaled to fit the video. This data should be collected
                       using get_offset_map in the TrackingData object. Is a tuple of x and y floating point
                       coordinates, or numpy arrays of floating point coordinates.
        :param prob: The probabilities to be set in this Pose object, between 0 and 1. Is a numpy array
                     of floating point numbers or a single floating point number.
        :param down_scale: The downscale factor of the original source map relative to the video, an integer.
                                  this is typically collected from the method TrackingData.get_down_scaling().
                                  Ex. Value of 8 means TrackingData probability map is 1/8th the size of the original
                                  video.
        :return: Nothing...
        """
        offset = (0, 0) if (offset is None) else offset

        scmap_x = scmap_coord[0] * down_scale + (0.5 * down_scale) + offset[0]
        scmap_y = scmap_coord[1] * down_scale + (0.5 * down_scale) + offset[1]

        self.set_x_at(frame, bodypart, scmap_x)
        self.set_y_at(frame, bodypart, scmap_y)
        self.set_prob_at(frame, bodypart, prob)


    # Setter Methods
    def set_all_x(self, x: ndarray):
        """
        Set the x video coordinates of this batch of Poses.

        :param x: An ndarray with the same dimensions as this Pose object, providing all x video coordinates...
        """
        self._data[:, 0::3] = x

    def set_all_y(self, y: ndarray):
        """
        Sets the y video coordinates of this batch of Poses.

        :param y: An ndarray with same dimensions as this pose object, providing all y video coordinates...
        """
        self._data[:, 1::3] = y

    def set_all_prob(self, probs: ndarray):
        """
        Set the probability values of this batch of Poses

        :param probs: An ndarray with same dimensions as this Pose object, providing all probability values for given
                      x, y video coordinates...
        """
        self._data[:, 2::3] = probs

    def set_x_at(self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray):
        """
        Set the x video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The values to set this Pose's x video coordinates to, as a numpy array...
        """
        self._data[frame, self._fix_index(bodypart, 0)] = values

    def set_y_at(self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray):
        """
        Set the y video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The values to set this Pose's y video coordinates to, as a numpy array...
        """
        self._data[frame, self._fix_index(bodypart, 1)] = values

    def set_prob_at(self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray):
        """
        Set the probability values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The values to set this pose's probabilities to, as a numpy array...
        """
        self._data[frame, self._fix_index(bodypart, 2)] = values

    # Getter Methods

    def get_all(self) -> ndarray:
        """
        Returns all data combined together into a numpy array. Note method is mostly useful to DLC, not Predictor
        plugins.

        :return: A numpy array with indexing of the dimensions: [frame -> x, y or prob every 3-slots].
        """
        return self._data

    def get_all_x(self) -> ndarray:
        """
        Returns x video coordinates for all frames and body parts.

        :return: The x video coordinates for all frames and body parts...
        """
        return self._data[:, 0::3]

    def get_all_y(self) -> ndarray:
        """
        Returns y video coordinates for all frames and body parts.

        :return: The y video coordinates for all frames and body parts...
        """
        return self._data[:, 1::3]

    def get_all_prob(self) -> ndarray:
        """
        Returns probability data for all frames and body parts...

        :return: The probability data for all frames and body parts...
        """
        return self._data[:, 2::3]

    def get_x_at(self, frame: Union[int, slice], bodypart: Union[int, slice]) -> ndarray:
        """
        Get the x video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The x video coordinates for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 0)]

    def get_y_at(self, frame: Union[int, slice], bodypart: Union[int, slice]) -> ndarray:
        """
        Get the y video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The y video coordinates for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 1)]

    def get_prob_at(self, frame: Union[int, slice], bodypart: Union[int, slice]) -> ndarray:
        """
        Get the probability values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The probability values for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 2)]

    def get_frame_count(self) -> int:
        """
        Returns the amount of frames in this pose object

        :return: An integer, being the amount of total frames stored in this pose
        """
        return self._data.shape[0]

    def get_bodypart_count(self) -> int:
        """
        Gets the amount of body parts per frame in this pose object

        :return: The amount of body parts per frame, as an integer.
        """
        return (self._data.shape[1] // 3)


class Predictor(ABC):
    """
    Base plugin class for all predictor plugins.

    Predictors accept TrackingData objects as they are generated by the network and are expected to return a single or
    several Pose objects providing the predicted locations of body parts in the original video...
    """
    @abstractmethod
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        """
        Constructor for the predictor. Should be used by plugins to initialize key data structures and settings.

        :param bodyparts: The body parts for the dataset, a list of the string friendly names in order. Note that if in
                          multi-output mode, this will be a list of
        :param num_outputs: The number of expected outputs for each body part model. Note that if this plugin doesn't
                            support multi output mode, this will always be 1. When returning poses, all of the
                            outputs for a single body part should be side-by-side.
                                Ex: If the bodyparts=[Nose, Tail] and num_outputs=2, pose arrangement should be:
                                    [Nose1, Nose2, Tail1, Tail2]
        :param num_frames: The number of total frames this predictor will be processing.
        :param settings: The settings for this predictor plugin. Dictionary is a map of strings, or setting names
                         to values. The actual data within the dictionary depends on return provided by get_settings
                         and what settings the user has set in deeplabcut's config.yaml.
                         If get_settings for this predictor returns None, this method will pass None...
        :param video_metadata: The metadata information for this dlc instance. Most of these settings are primarily
                               useful to interactive plugins. Includes the keys:
                                    "fps": Original Video's frames per second
                                    "h5-file-name": The name of the original h5 file and it's path, as a string.
                                    "orig-video-path": The file path and name of the video being analyzed, as a string.
                                                       this value may be None, meaning the video could not be found, and
                                                       user is processing frames via a .dlcf file.
                                    "duration": The duration of the video in seconds
                                    "size": The x and y dimensions of the original video.
                                    "cropping-offset": The (y, x) offset of the cropped box in the video. If there is
                                                       no cropping, this value is set to None. Width/Height of cropping
                                                       box can be inferred using tracking data width and height and
                                                       multiplying it by the stride.
                                    "dotsize": The radius of dots used when outputting predictions to a video, an
                                               integer.
                                    "colormap": The colormap used when plotting points to a video, a string representing
                                                a matplotlib colormap.
                                    "alphavalue": The alpha value of the points when outputting predictions, a float
                                                  between 0 and 1.
                                    "pcutoff": The probability at which to display no point in the final plotted video
                                               if the point in the data falls below this threshold. A float between 0
                                               and 1.
        """
        pass

    @abstractmethod
    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        """
        Executed on every batch of frames in the video, plugins should process or store the probability map data and
        return the guessed max locations, or return None if it is storing the probability maps for post-processing.

        :param scmap: A TrackingData object, containing probability maps, offset maps, and all data and methods needed
                      to generate poses.

        :return: A Pose object representing a collection of predicted poses for frames and body parts, or None if
                 TrackingData objects need to be stored since this plugin requires post-processing.
        """
        pass

    @abstractmethod
    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        """
        Executed once all frames have been run through. Should be used for post-processing. Useful if a plugin needs to
        store all of the frames in order to make predictions.

        :param progress_bar: A tqdm progress bar, should be used to display post-processing progress, the max value
                             of the progress bar is set to the number of frames left.
                             (Number of total frames minus the number of frames returned in 'on_frames')...

        :return: A Pose object representing a collection of poses for frames and body parts, or None if all of the
                 predictions were made and returned as Pose object in 'on_frames'.
        """
        pass


    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the name of this predictor plugin, the name is used when selecting a predictor in the
        deeplabcut.analyze_videos method.

        :return: The name of this plugin to be used to select it, as a string.
        """
        pass


    @classmethod
    def get_description(cls) -> str:
        """
        Get the description of this plugin, the equivalent of a doc-string for this plugin, it is displayed when
        user lists available plugins. Does not have to be overridden, and defaults to returning the sanitized docstring
        of this class.

        :return: The description/summary of this plugin as a string.
        """
        if(cls.__doc__ is None):
            return "None"
        else:
            return inspect.cleandoc(cls.__doc__)


    @classmethod
    @abstractmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        """
        Get the configurable or available settings for this predictor plugin.

        :return: The settings that can be set for this plugin, in the form of a list of tuples. Each tuple will contain
                 3 items:

                    - Setting Name: A String, the name this setting will have in the config file, and also the name it
                                    will be given when passed to the constructor of this predictor.
                    - Setting Description: A String, A user friendly description of the setting. Should include info
                                           on it's default value, what it does, and what it should be set to.
                    - Setting Default Value: Any type, the default value to be assigned to this setting if it is not
                                             set explicitly in the DeepLabCut config by the user...

                  If this predictor plugin has no configurable settings, this method should return None.
        """
        pass

    @classmethod
    @abstractmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        """
        Get the test methods for this plugin.

        :return: A list of callable objects(aka. methods) or None if no test methods exist. The callables in the list
                 should accept no arguments and return a tuple of 3 items, containing the below values in order:

                    - Test Success: A Boolean, True if test was successful, otherwise False.

                    - Test Expected Results: A string, a human readable string representing the expected results of this
                                             test.
                    - Test Actual Results: A string, a human readable string representing the actual results that the
                                           the test method received. If test was successful, this should match the
                                           expected results value.
                 Another valid response from the test methods is to throw an exception, in which case the test is
                 considered a failure and the stack trace of the exception is printed instead of the expected/actual
                 results.

        """
        pass

    @classmethod
    @abstractmethod
    def supports_multi_output(cls) -> bool:
        """
        Get whether or not this plugin supports outputting multiple of the same body part (num_outputs > 1). Returning
        false here will keep the plugin from being allowed to be used when num_outputs is greater then 1.

        :return: A boolean, True if multiple outputs per body part is supported, otherwise False...
        """
        pass


def get_predictor(name: str) -> Type[Predictor]:
    """
    Get the predictor plugin by the specified name.

    :param name: The name of this plugin, should be a string
    :return: The plugin class that has a name that matches the specified name
    """
    # Load the plugins from the directory: "deeplabcut/pose_estimation_tensorflow/nnet/predictors"
    plugins = get_predictor_plugins()
    # Iterate the plugins until we find one with a matching name, otherwise throw a ValueError if we don't find one.
    for plugin in plugins:
        if(plugin.get_name() == name):
            return plugin
    else:
        raise ValueError(f"Predictor plugin {name} does not exist, try another plugin name...")


def get_predictor_plugins() -> Set[Type[Predictor]]:
    """
    Get and retrieve all predictor plugins currently available to the DeepLabCut implementation...

    :return: A Set of Predictors, being the all classes that extend the Predictor class currently loaded visible to
    the python interpreter.
    """
    return pluginloader.load_plugin_classes(predictors, Predictor)

