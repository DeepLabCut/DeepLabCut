"""
Contains 2 Utility Classes for reading and writing the DeepLabCut Frame Store format. The format allows for processing
videos using DeepLabCut and then running predictions on the probability map data later. The DLCF format is a glorified
hdf5 file, with the specified fields below:

Group "/" {
    Attribute "number_of_frames": A 64 bit integer
    Attribute "frame_height": A 64 bit integer
    Attribute "frame_width": A 64 bit integer
    Attribute "frame_rate": A 64 bit float
    Attribute "stride": The video scaling factor relative to the original probability map, A 64 bit integer
    Attribute "orig_video_height": A 64 bit integer
    Attribute "orig_video_width", A 64 bit integer
    Attribute "crop_offset_y": A 64 bit integer, if negative there is no cropping.
    Attribute "crop_offset_x": A 64 bit integer, if negative there is no cropping.
    Attribute "bodypart_names", A list of strings, being all of the body part names in order.

    Group "frame_0" {
        Group "(bodypart_names[0])" {
            Attribute "is_sparse": 8 bit integer, True or False, determines if data is stored in sparse format below.
            Attribute "offsets_included": 8 bit integer, True or False, determines if data includes offsets.

            If "is_sparse" is True: ()
                Data "x": 1-Dimensional array of 32-bit integers being the x-offsets of probabilities within the
                          probability map.
                Data "y": 1-Dimensional array of 32-bit integers being the y-offsets of probabilities within the
                          probability map.
                Data "probs": 1-Dimensional array of 32-bit floats being the probabilities within the probability map.

                if "offsets_included" is True:
                    Data "offset_x": 1-Dimensional array of 32 bit floats being x-offsets of most probable locations
                                     within the video.
                    Data "offset_y": 1-Dimensional array of 32 bit floats being y-offsets of most probable locations
                                     within the video.

            If "is_sparse" is False: (All array dimensions below are frame_height x frame_width)
                Data "probs": 2-Dimensional array of 32-bit floats being the probability map.

                if "offsets_included" is True:
                    Data "offset_x": 2-Dimensional array of 32 bit floats being x-offsets of most probable locations
                                     within the video.
                    Data "offset_y": 2-Dimensional array of 32 bit floats being y-offsets of most probable locations
                                     within the video.
        }
        Group "(bodypart_names[1])" { [...] }
        [...]
        Group "(bodypart_names[length(bodypart_names) - 1])" { [...] }
    }
    Group "frame_1" { [...] }
    [...]
    Group "frame_(number_of_frames-1)" { [...] }
}

"""
from typing import BinaryIO, Optional

from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData

# Will eventually replace this by moving the class over....
from deeplabcut.pose_estimation_tensorflow.util.frame_store_fmt import DLCFSHeader
import numpy as np
import h5py


class DLCFSConstants:
    # The usual stuff...
    MIN_SPARSE_SAVING_FACTOR = 3
    # The file type:
    FILE_TYPE_KEY = "file_type"
    FILE_TYPE = "DLCF"
    # The file version:
    FILE_VERSION_KEY = "file_version"
    FILE_VERSION = "v1.0.0"
    # Frames are prefixed using this....
    FRAME_PREFIX = "frame_"

class SubFrameKeys:
    IS_STORED_SPARSE = "is_sparse"
    INCLUDES_OFFSETS = "offsets_included"
    X = "x"
    Y = "y"
    PROBS = "probs"
    OFFSET_X = "offset_x"
    OFFSET_Y = "offset_y"


class DLCFSReader:
    """
    A DeepLabCut Frame Store Reader. Allows for reading ".dlcf" files.
    """
    def _assert_true(self, assertion: bool, error_msg: str):
        """
        Private method, if the assertion is false, throws a ValueError.
        """
        if not assertion:
            raise ValueError(error_msg)

    def __init__(self, file: BinaryIO):
        """
        Create a new DeepLabCut Frame Store Reader.

        :param file: The binary file object to read a frame store from, file opened with 'rb'.
        """
        self._file = h5py.File(file, "r")
        self._frames_processed = 0

        # Check the version and file type match...
        file_type = self._file.attrs.get(DLCFSConstants.FILE_TYPE_KEY, None)
        file_version = self._file.attrs.get(DLCFSConstants.FILE_VERSION_KEY, None)

        if((file_type != DLCFSConstants.FILE_TYPE) or (file_version != DLCFSConstants.FILE_VERSION)):
            raise ValueError("H5 File Header does not match the header of a DeepLabCut FrameStore.")

        # Dump the header, which is pretty much directly dumped into the hdf5 file...
        self._header = DLCFSHeader()
        self._header.update({key: self._file.attrs[key] for key in self._header.keys()})

        # Make sure cropping offsets land within the video if they are not None
        if self._header.crop_offset_y is not None:
            crop_end = self._header.crop_offset_y + (
                self._header.frame_height * self._header.stride
            )
            self._assert_true(
                crop_end < self._header.orig_video_height,
                "Cropping box in DLCF file is invalid!",
            )
        if self._header.crop_offset_x is not None:
            crop_end = self._header.crop_offset_x + (
                self._header.frame_width * self._header.stride
            )
            self._assert_true(
                crop_end < self._header.orig_video_width,
                "Cropping box in DLCF file is invalid!",
            )

    def get_header(self) -> DLCFSHeader:
        """
        Get the header of this DLC frame store file.

        :return: A DLCFSHeader object, contains important metadata info from this frame store.
        """
        return DLCFSHeader(*self._header.to_list())

    def has_next(self, num_frames: int = 1) -> bool:
        """
        Checks if this frame store object at least num_frames more frames to be read.

        :param num_frames: An Integer, The number of frames to check the availability of, defaults to 1.
        :return: True if there are at least num_frames more frames to be read, otherwise False.
        """
        return (self._frames_processed + num_frames) <= self._header.number_of_frames

    @classmethod
    def _init_offset_data(cls, track_data: TrackingData):
        if track_data.get_offset_map() is None:
            # If tracking data is currently None, we need to create an empty array to store all data.
            shape = (
                track_data.get_frame_count(),
                track_data.get_frame_height(),
                track_data.get_frame_width(),
                track_data.get_bodypart_count(),
                2,
            )
            track_data.set_offset_map(np.zeros(shape, dtype=np.float32))

    def read_frames(self, num_frames: int = 1) -> TrackingData:
        """
        Read the next num_frames frames from this frame store object and returns a TrackingData object.

        :param num_frames: The number of frames to read from the frame store.
        :return: A TrackingData object storing all frame info that was stored in this DLC Frame Store....

        :raises: An EOFError if more frames were requested then were available.
        """
        if not self.has_next(num_frames):
            frames_left = self._header.number_of_frames - self._frames_processed
            raise EOFError(
                f"Only '{frames_left}' were available, and '{num_frames}' were requested."
            )

        temp_frame_idx = self._frames_processed
        self._frames_processed += num_frames

        __, frame_h, frame_w, __, stride = self._header.to_list()[:5]
        bp_lst = self._header.bodypart_names

        track_data = TrackingData.empty_tracking_data(
            num_frames, len(bp_lst), frame_w, frame_h, stride
        )

        for frame_idx in range(track_data.get_frame_count()):

            frame_group = self._file[f"{DLCFSConstants.FRAME_PREFIX}{temp_frame_idx}"]

            for bp_idx in range(track_data.get_bodypart_count()):
                bp_group = frame_group[bp_lst[bp_idx]]
                is_sparse = bp_group.attrs[SubFrameKeys.IS_STORED_SPARSE]
                has_offsets = bp_group.attrs[SubFrameKeys.INCLUDES_OFFSETS]

                if(is_sparse):
                    sparse_x = bp_group[SubFrameKeys.X]
                    sparse_y = bp_group[SubFrameKeys.Y]
                    probs = bp_group[SubFrameKeys.PROBS]

                    if(has_offsets):
                        # If data has offsets, store them...
                        self._init_offset_data(track_data)

                        off_x = bp_group[SubFrameKeys.OFFSET_X]
                        off_y = bp_group[SubFrameKeys.OFFSET_Y]

                        track_data.get_offset_map()[frame_idx, sparse_y, sparse_x, bp_idx, 1] = off_y
                        track_data.get_offset_map()[frame_idx, sparse_y, sparse_x, bp_idx, 0] = off_x

                    track_data.get_source_map()[frame_idx, sparse_y, sparse_x, bp_idx] = probs  # Set probability data...
                else:
                    probs = bp_group[SubFrameKeys.PROBS]

                    if(has_offsets):
                        self._init_offset_data(track_data)
                        off_x = bp_group[SubFrameKeys.OFFSET_X]
                        off_y = bp_group[SubFrameKeys.OFFSET_Y]

                        track_data.get_offset_map()[frame_idx, :, :, bp_idx, 1] = off_y
                        track_data.get_offset_map()[frame_idx, :, :, bp_idx, 0] = off_x

                    track_data.get_source_map()[frame_idx, :, :, bp_idx] = probs

        return track_data

    def close(self):
        """
        Close this frame reader, cleaning up any resources used during reading from the file. Does not close the passed
        file handle!
        """
        self._file.close()


class DLCFSWriter:
    """
    A DeepLabCut Frame Store Writer. Allows for writing ".dlcf" files.
    """

    def __init__(
        self,
        file: BinaryIO,
        header: DLCFSHeader,
        threshold: Optional[float] = 1e6
    ):
        """
        Create a new DeepLabCut Frame Store Writer.

        :param file: The file to write to, a file opened in 'wb' mode.
        :param header: The DLCFrameStoreHeader, with all properties filled out.
        :param threshold: A float between 0 and 1, the threshold at which to filter out any probabilities which fall
                          below it. The default value is 1e6, and it can be set to None to force all frames to be
                          stored in the non-sparse format.
        """
        self._file = h5py.File(file, "w")

        self._threshold = (
            threshold if (threshold is None or 0 <= threshold <= 1) else 1e6
        )
        self._current_frame = 0

        self._file.attrs[DLCFSConstants.FILE_TYPE_KEY] = DLCFSConstants.FILE_TYPE
        self._file.attrs[DLCFSConstants.FILE_VERSION_KEY] = DLCFSConstants.FILE_VERSION

        # Dump the entire header.... Also store it for internal checking...
        tmp_data = {}
        tmp_data.update(header)
        crop_off_x, crop_off_y = tmp_data["crop_offset_x"], tmp_data["crop_offset_y"]
        tmp_data["crop_offset_x"] = crop_off_x if(crop_off_x is not None) else -1
        tmp_data["crop_offset_y"] = crop_off_y if(crop_off_y is not None) else -1

        self._file.attrs.update(tmp_data)
        self._header = DLCFSHeader(*header.to_list())


    def write_data(self, data: TrackingData):
        """
        Write the following frames to the file.

        :param data: A TrackingData object, which contains frame data.
        """
        # Some checks to make sure tracking data parameters match those set in the header:
        current_frame_tmp = self._current_frame

        self._current_frame += data.get_frame_count()
        if self._current_frame > self._header.number_of_frames:
            raise ValueError(
                f"Data Overflow! '{self._header.number_of_frames}' frames expected, tried to write "
                f"'{self._current_frame + 1}' frames."
            )

        if data.get_bodypart_count() != len(self._header.bodypart_names):
            raise ValueError(
                f"'{data.get_bodypart_count()}' body parts does not match the "
                f"'{len(self._header.bodypart_names)}' body parts specified in the header."
            )

        if(data.get_frame_width() != self._header.frame_width or data.get_frame_height() != self._header.frame_height):
            raise ValueError("Frame dimensions don't match ones specified in header!")

        for frm_idx in range(data.get_frame_count()):

            frame_grp = self._file.create_group(f"{DLCFSConstants.FRAME_PREFIX}{current_frame_tmp}")

            for bp_idx in range(data.get_bodypart_count()):
                bp_grp = frame_grp.create_group(self._header.bodypart_names[bp_idx])

                frame = data.get_prob_table(frm_idx, bp_idx)
                offsets = data.get_offset_map()

                if(offsets is not None):
                    off_y = offsets[frm_idx, :, :, bp_idx, 1]
                    off_x = offsets[frm_idx, :, :, bp_idx, 0]
                else:
                    off_x, off_y = None, None

                bp_grp.attrs[SubFrameKeys.INCLUDES_OFFSETS] = offsets is not None

                if(self._threshold is not None):
                    sparse_y, sparse_x = np.nonzero(frame > self._threshold)
                    probs = frame[(sparse_y, sparse_x)]

                    # Check if we managed to strip out at least 2/3rds of the data, and if so write the frame using the
                    # sparse format. Otherwise it is actually more memory efficient to just store the entire frame...
                    if(len(frame.flat) >= (len(sparse_y) * DLCFSConstants.MIN_SPARSE_SAVING_FACTOR)):
                        bp_grp.attrs[SubFrameKeys.IS_STORED_SPARSE] = True
                        out_x = bp_grp.create_dataset(SubFrameKeys.X, sparse_x.shape, np.uint32)
                        out_y = bp_grp.create_dataset(SubFrameKeys.Y, sparse_y.shape, np.uint32)
                        out_prob = bp_grp.create_dataset(SubFrameKeys.PROBS, probs.shape, np.float32)
                        out_x[:] = sparse_x
                        out_y[:] = sparse_y
                        out_prob[:] = probs

                        if(offsets is not None):
                            off_x, off_y = off_x[(sparse_y, sparse_x)], off_y[(sparse_y, sparse_x)]
                            out_off_x = bp_grp.create_dataset(SubFrameKeys.OFFSET_X, off_x.shape, np.float32)
                            out_off_y = bp_grp.create_dataset(SubFrameKeys.OFFSET_Y, off_y.shape, np.float32)
                            out_off_x[:] = off_x
                            out_off_y[:] = off_y

                        continue

                # User has disabled sparse optimizations or they wasted more space, stash entire frame...
                bp_grp.attrs[SubFrameKeys.IS_STORED_SPARSE] = False
                prob_data = bp_grp.create_dataset(SubFrameKeys.PROBS, frame.shape, np.float32)
                prob_data[:] = frame

                if(offsets is not None):
                    out_x = bp_grp.create_dataset(SubFrameKeys.OFFSET_X, off_x.shape, np.float32)
                    out_y = bp_grp.create_dataset(SubFrameKeys.OFFSET_Y, off_y.shape, np.float32)
                    out_x[:] = off_x
                    out_y[:] = off_y

            current_frame_tmp += 1

    def close(self):
        """
        Close this frame writer, cleaning up any resources used during writing to the file. Does not close the passed
        file handle!
        """
        self._file.flush()
        self._file.close()