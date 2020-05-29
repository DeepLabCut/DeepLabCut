"""
Contains 2 Utility Classes for reading and writing the DeepLabCut Frame Store format. The format allows for processing
videos using DeepLabCut and then running predictions on the probability map data later. Below is a specification for
the DeepLabCut Frame Store format...

DEEPLABCUT FRAMESTORE BINARY FORMAT: (All multi-byte fields are in little-endian format)
['DLCF'] -> DeepLabCut Frame store - 4 Bytes (file magic)

Header:
	['DLCH'] -> DeepLabCut Header
	[num_frames] - the number of frames. 8 Bytes (long unsigned integer)
	[num_bp] - number of bodyparts contained per frame. 4 Bytes (unsigned integer)
    [frame_height] - The height of a frame. 4 Bytes (unsigned integer)
	[frame_width] - The width of a frame. 4 Bytes (unsigned integer)
	[frame_rate] - The frame rate, in frames per second. 8 Bytes (double float).
	[stride] - The original video upscaling multiplier relative to current frame size. 4 Bytes (unsigned integer)
	[orig_video_height] - The original video height. 4 Bytes (unsigned integer)
	[orig_video_width] - The original video width. 4 Bytes (unsigned integer)
	[crop_y1] - The y offset of the cropped box, set to max value to indicate no cropping...
	[crop_x1] - The x offset of the cropped box, set to max value to indicate no cropping...

Bodypart Names:
    ['DBPN'] -> Deeplabcut Body Part Names
    (num_bp entries):
        [bp_len] - The length of the name of the bodypart. 2 Bytes (unsigned short)
        [DATA of length bp_len] - UTF8 Encoded name of the bodypart.

Frame data block:
	['FDAT'] -> Frame DATa
	Now the data (num_frames entries):
	    Each sub-frame entry (num_bp entries):

            Single Byte: 000000[offsets_included][sparse_fmt]:
                [sparse_fmt]- Single bit, whether we are using the sparse format. See difference in storage below:
                [offsets_included] - Single bis, whether we have offset data included. See difference in storage below:
            [data_length] - The length of the compressed/uncompressed frame data, 8 Bytes (long unsigned integer)

            DATA (The below is compressed in the zlib format and must be uncompressed first). Based on 'sparse_fmt' flag:

                If it is false, frames are stored as 4 byte float arrays, row-by-row, as below (x, y order below):
                    prob(1, 1), prob(2, 1), prob(3, 1), ....., prob(x, 1)
                    prob(1, 2), prob(2, 2), prob(3, 2), ....., prob(x, 2)
                    .....................................................
                    prob(1, y), prob(2, y), prob(3, y), ....., prob(x, y)
                Length of the above data will be frame height * frame width...
                if [offsets_included] == 1:
                    Then 2 more maps equivalent to the above store the offset within the map when converting back
                    to video:
                        off_y(1, 1), off_y(2, 1), off_y(3, 1), ....., off_y(x, 1)
                        off_y(1, 2), off_y(2, 2), off_y(3, 2), ....., off_y(x, 2)
                        .........................................................
                        off_y(1, y), off_y(2, y), off_y(3, y), ....., off_y(x, y)

                        off_x(1, 1), off_x(2, 1), off_x(3, 1), ....., off_x(x, 1)
                        off_x(1, 2), off_x(2, 2), off_x(3, 2), ....., off_x(x, 2)
                        .........................................................
                        off_x(1, y), off_x(2, y), off_x(3, y), ....., off_x(x, y)
                Otherwise frames are stored in the format below.

                Sparse Frame Format (num_bp entries):
                    [num_entries] - Number of sparse entries in the frame, 8 bytes, unsigned integer.
                    [arr y] - list of 4 byte unsigned integers of length num_entries. Stores y coordinates of probabilities.
                    [arr x] - list of 4 byte unsigned integers of length num_entries. Stores x coordinates of probabilities.
                    [probs] - list of 4 byte floats, Stores probabilities specified at x and y coordinates above.
                    if [offsets_included] == 1:
                        [off y] - list of 4 byte floats, stores y offset within the block of pixels.
                        [off x] - list of 4 byte floats, stores x offset within the block of pixels.
"""
from io import BytesIO
from typing import List, Any, BinaryIO, Optional, Tuple
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
import numpy as np
import zlib

# REQUIRED DATA TYPES: (With little endian encoding...)
luint8 = np.dtype(np.uint8).newbyteorder("<")
luint16 = np.dtype(np.uint16).newbyteorder("<")
luint32 = np.dtype(np.uint32).newbyteorder("<")
luint64 = np.dtype(np.uint64).newbyteorder("<")
ldouble = np.dtype(np.float64).newbyteorder("<")
lfloat = np.dtype(np.float32).newbyteorder("<")


def to_bytes(obj: Any, dtype: np.dtype) -> bytes:
    """
    Converts an object to bytes.

    :param obj: The object to convert to bytes.
    :param dtype: The numpy data type to interpret the object as when converting to bytes.
    :return: A bytes object, representing the object obj as type dtype.
    """
    return dtype.type(obj).tobytes()


def from_bytes(data: bytes, dtype: np.dtype) -> Any:
    """
    Converts bytes to a single object depending on the passed data type.

    :param data: The bytes to convert to an object
    :param dtype: The numpy data type to convert the bytes to.
    :return: An object of the specified data type passed to this method.
    """
    return np.frombuffer(data, dtype=dtype)[0]


class DLCFSConstants:
    """
    Class stores some constants for the DLC Frame Store format.
    """
    # The frame must become 1/3 or less its original size when sparsified to save space over the entire frame format,
    # so we check for this by dividing the original frame size by the sparse frame size and checking to see if it is
    # greater than this factor below...
    MIN_SPARSE_SAVING_FACTOR = 3
    # Magic...
    FILE_MAGIC = b"DLCF"
    # Chunk names...
    HEADER_CHUNK_MAGIC = b"DLCH"
    # The header length, including the 'DLCH' magic
    HEADER_LENGTH = 52
    BP_NAME_CHUNK_MAGIC = b"DBPN"
    FRAME_DATA_CHUNK_MAGIC = b"FDAT"


def string_list(lister: list):
    """
    Casts object to a list of strings, enforcing type...

    :param lister: The list
    :return: A list of strings

    :raises: ValueError if the list doesn't contain strings...
    """
    lister = list(lister)

    for item in lister:
        if(not isinstance(item, str)):
            raise ValueError("Must be a list of strings!")

    return lister


def non_max_int32(val: luint32) -> Optional[int]:
    """
    Casts an object to a non-max integer, being None if it is the maximum value.

    :param val: The value to cast...
    :return: An integer, or None if the value equals the max possible integer.
    """
    if(val is None):
        return None

    val = int(val)

    if(val == np.iinfo(luint32).max):
        return None
    else:
        return val

class DLCFSHeader():
    """
    Stores some basic info about a frame store...

    Below are the fields in order, their names, types and default values:
        ("number_of_frames", int, 0),
        ("frame_height", int, 0),
        ("frame_width", int, 0),
        ("frame_rate", float, 0),
        ("stride", int, 0),
        ("orig_video_height", int, 0),
        ("orig_video_width", int, 0),
        ("crop_offset_y", int or None if no cropping, None),
        ("crop_offset_x", int or None if no cropping, None),
        ("bodypart_names", list of strings, []),
    """
    SUPPORTED_FIELDS = [
        ("number_of_frames", int, 0),
        ("frame_height", int, 0),
        ("frame_width", int, 0),
        ("frame_rate", float, 0),
        ("stride", int, 0),
        ("orig_video_height", int, 0),
        ("orig_video_width", int, 0),
        ("crop_offset_y", non_max_int32, None),
        ("crop_offset_x", non_max_int32, None),
        ("bodypart_names", string_list, [])
    ]

    GET_VAR_CAST = {name: var_cast for name, var_cast, __ in SUPPORTED_FIELDS}

    def __init__(self, *args, **kwargs):
        """
        Make a new DLCFrameStoreHeader. Supports tuple style construction and also supports setting the fields using
        keyword arguments. Look at the class documentation for all the fields.
        """
        # Make the fields.
        self._values = {}
        for name, var_caster, def_value in self.SUPPORTED_FIELDS:
            self._values[name] = def_value

        for new_val, (key, var_caster, __) in zip(args, self.SUPPORTED_FIELDS):
            self._values[key] = var_caster(new_val)

        for key, new_val in kwargs.items():
            if(key in self._values):
                self._values[key] = new_val

    def __getattr__(self, item):
        if(item == "_values"):
            return self.__dict__[item]
        return self._values[item]

    def __setattr__(self, key, value):
        if(key == "_values"):
            self.__dict__["_values"] = value
            return
        self.__dict__["_values"][key] = self.GET_VAR_CAST[key](value)

    def __str__(self):
        return str(self._values)

    def to_list(self) -> List[Any]:
        return [self._values[key] for key, __, __ in self.SUPPORTED_FIELDS]


class DLCFSReader():
    """
    A DeepLabCut Frame Store Reader. Allows for reading ".dlcf" files.
    """

    HEADER_DATA_TYPES = [luint64, luint32, luint32, luint32, ldouble, luint32, luint32, luint32, luint32, luint32]
    HEADER_OFFSETS = np.cumsum([4] + [dtype.itemsize for dtype in HEADER_DATA_TYPES])[:-1]

    def _assert_true(self, assertion: bool, error_msg: str):
        """
        Private method, if the assertion is false, throws a ValueError.
        """
        if(not assertion):
            raise ValueError(error_msg)

    def __init__(self, file: BinaryIO):
        """
        Create a new DeepLabCut Frame Store Reader.

        :param file: The binary file object to read a frame store from, file opened with 'rb'.
        """
        self._assert_true(file.read(4) == DLCFSConstants.FILE_MAGIC, "File is not of the DLC Frame Store Format!")
        # Check for valid header...
        header_bytes = file.read(DLCFSConstants.HEADER_LENGTH)
        self._assert_true(header_bytes[0:4] == DLCFSConstants.HEADER_CHUNK_MAGIC,
                          "First Chunk must be the Header ('DLCH')!")

        # Read the header into a DLC header...
        parsed_data = [from_bytes(header_bytes[off:(off + dtype.itemsize)], dtype) for off, dtype in
                       zip(self.HEADER_OFFSETS, self.HEADER_DATA_TYPES)]
        self._header = DLCFSHeader(parsed_data[0], *parsed_data[2:])
        body_parts = [None] * parsed_data[1]

        # Make sure cropping offsets land within the video if they are not None
        if(self._header.crop_offset_y is not None):
            crop_end = self._header.crop_offset_y + (self._header.frame_height * self._header.stride)
            self._assert_true(crop_end < self._header.orig_video_height, "Cropping box in DLCF file is invalid!")
        if(self._header.crop_offset_x is not None):
            crop_end = self._header.crop_offset_x + (self._header.frame_width * self._header.stride)
            self._assert_true(crop_end < self._header.orig_video_width, "Cropping box in DLCF file is invalid!")

        # Read the body part chunk...
        self._assert_true(file.read(4) == DLCFSConstants.BP_NAME_CHUNK_MAGIC, "Body part chunk must come second!")
        for i in range(len(body_parts)):
            length = from_bytes(file.read(2), luint16)
            body_parts[i] = file.read(int(length)).decode("utf-8")
        # Add the list of body parts to the header...
        self._header.bodypart_names = body_parts

        # Now we assert that we have reached the frame data chunk
        self._assert_true(file.read(4) == DLCFSConstants.FRAME_DATA_CHUNK_MAGIC, f"Frame data chunk not found!")

        self._file = file
        self._frames_processed = 0

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
    def _parse_flag_byte(cls, byte: luint8) -> Tuple[bool, bool]:
        """ Returns if it is of the sparse format, followed by if it includes offset data... """
        return ((byte & 1) == 1, ((byte >> 1) & 1) == 1)

    @classmethod
    def _take_array(cls, data: bytes, dtype: np.dtype, count: int) -> Tuple[bytes, np.ndarray]:
        """ Reads a numpy array from the byte array, returning the leftover data and the array. """
        if(count <= 0):
            raise ValueError("Can't have a negative amount of entries....")
        return (data[dtype.itemsize * count:], np.frombuffer(data, dtype=dtype, count=count))

    @classmethod
    def _init_offset_data(cls, track_data: TrackingData):
        if (track_data.get_offset_map() is None):
            # If tracking data is currently None, we need to create an empty array to store all data.
            shape = (track_data.get_frame_count(), track_data.get_frame_height(), track_data.get_frame_width(),
                     track_data.get_bodypart_count(), 2)
            track_data.set_offset_map(np.zeros(shape, dtype=lfloat))

    def read_frames(self, num_frames: int = 1) -> TrackingData:
        """
        Read the next num_frames frames from this frame store object and returns a TrackingData object.

        :param num_frames: The number of frames to read from the frame store.
        :return: A TrackingData object storing all frame info that was stored in this DLC Frame Store....

        :raises: An EOFError if more frames were requested then were available.
        """
        if(not self.has_next(num_frames)):
            frames_left = self._header.number_of_frames - self._frames_processed
            raise EOFError(f"Only '{frames_left}' were available, and '{num_frames}' were requested.")

        self._frames_processed += num_frames
        __, frame_h, frame_w, __, stride = self._header.to_list()[:5]
        bp_lst = self._header.bodypart_names

        track_data = TrackingData.empty_tracking_data(num_frames, len(bp_lst), frame_w, frame_h, stride)

        for frame_idx in range(track_data.get_frame_count()):
            for bp_idx in range(track_data.get_bodypart_count()):
                sparse_fmt_flag, has_offsets_flag = self._parse_flag_byte(from_bytes(self._file.read(1), luint8))
                data = zlib.decompress(self._file.read(int(from_bytes(self._file.read(8), luint64))))

                if(sparse_fmt_flag):
                    entry_len = int(from_bytes(data[:8], luint64))
                    data = data[8:]
                    data, sparse_y = self._take_array(data, dtype=luint32, count=entry_len)
                    data, sparse_x = self._take_array(data, dtype=luint32, count=entry_len)
                    data, probs = self._take_array(data, dtype=lfloat, count=entry_len)

                    if(has_offsets_flag): # If offset flag is set to true, load in offset data....
                        self._init_offset_data(track_data)

                        data, off_y = self._take_array(data, dtype=lfloat, count=entry_len)
                        data, off_x = self._take_array(data, dtype=lfloat, count=entry_len)
                        track_data.get_offset_map()[frame_idx, sparse_y, sparse_x, bp_idx, 1] = off_y
                        track_data.get_offset_map()[frame_idx, sparse_y, sparse_x, bp_idx, 0] = off_x
                    else:
                        track_data.set_offset_map(None)

                    track_data.get_prob_table(frame_idx, bp_idx)[sparse_y, sparse_x] = probs # Set probability data...
                else:
                    data, probs = self._take_array(data, dtype=lfloat, count=frame_w * frame_h)
                    probs = np.reshape(probs, (frame_h, frame_w))

                    if(has_offsets_flag):
                        self._init_offset_data(track_data)

                        data, off_y = self._take_array(data, dtype=lfloat, count=frame_h * frame_w)
                        data, off_x = self._take_array(data, dtype=lfloat, count=frame_h * frame_w)
                        off_y = np.reshape(off_y, (frame_h, frame_w))
                        off_x = np.reshape(off_x, (frame_h, frame_w))

                        track_data.get_offset_map()[frame_idx, :, :, bp_idx, 1] = off_y
                        track_data.get_offset_map()[frame_idx, :, :, bp_idx, 0] = off_x

                    track_data.get_prob_table(frame_idx, bp_idx)[:] = probs

        return track_data


class DLCFSWriter():
    """
    A DeepLabCut Frame Store Writer. Allows for writing ".dlcf" files.
    """
    def __init__(self, file: BinaryIO, header: DLCFSHeader, threshold: Optional[float] = 1e6,
                 compression_level: int = 6):
        """
        Create a new DeepLabCut Frame Store Writer.

        :param file: The file to write to, a file opened in 'wb' mode.
        :param header: The DLCFrameStoreHeader, with all properties filled out.
        :param threshold: A float between 0 and 1, the threshold at which to filter out any probabilities which fall
                          below it. The default value is 1e6, and it can be set to None to force all frames to be
                          stored in the non-sparse format.
        :param compression_level: The compression of the data. 0 is no compression, 9 is max compression but is slow.
                                  The default is 6.
        """
        self._out_file = file
        self._header = header
        self._threshold = threshold if (threshold is None or 0 <= threshold <= 1) else 1e6
        self._compression_level = compression_level if(0 <= compression_level <= 9) else 6
        self._current_frame = 0

        # Write the file magic...
        self._out_file.write(DLCFSConstants.FILE_MAGIC)
        # Now we write the header:
        self._out_file.write(DLCFSConstants.HEADER_CHUNK_MAGIC)
        self._out_file.write(to_bytes(header.number_of_frames, luint64))  # The frame count
        self._out_file.write(to_bytes(len(header.bodypart_names), luint32))  # The body part count
        self._out_file.write(to_bytes(header.frame_height, luint32))  # The height of each frame
        self._out_file.write(to_bytes(header.frame_width, luint32))  # The width of each frame
        self._out_file.write(to_bytes(header.frame_rate, ldouble))  # The frames per second
        self._out_file.write(to_bytes(header.stride, luint32))  # The video upscaling factor
        # Original video height and width.
        self._out_file.write(to_bytes(header.orig_video_height, luint32))
        self._out_file.write(to_bytes(header.orig_video_width, luint32))
        # The cropping (y, x) offset, or the max integer values if there is no cropping box...
        max_val = np.iinfo(luint32).max
        self._out_file.write(to_bytes(max_val if(header.crop_offset_y is None) else header.crop_offset_y, luint32))
        self._out_file.write(to_bytes(max_val if(header.crop_offset_x is None) else header.crop_offset_x, luint32))

        # Now we write the body part name chunk:
        self._out_file.write(DLCFSConstants.BP_NAME_CHUNK_MAGIC)
        for bodypart in header.bodypart_names:
            body_bytes = bodypart.encode("utf-8")
            self._out_file.write(to_bytes(len(body_bytes), luint16))
            self._out_file.write(body_bytes)

        # Finish by writing the begining of the frame data chunk:
        self._out_file.write(DLCFSConstants.FRAME_DATA_CHUNK_MAGIC)


    def make_flag_byte(self, is_sparse: bool, has_offsets: bool) -> int:
        return (is_sparse) | (has_offsets << 1)

    def write_data(self, data: TrackingData):
        """
        Write the following frames to the file.

        :param data: A TrackingData object, which contains frame data.
        """
        # Some checks to make sure tracking data parameters match those set in the header:
        self._current_frame += data.get_frame_count()
        if(self._current_frame > self._header.number_of_frames):
            raise ValueError(f"Data Overflow! '{self._header.number_of_frames}' frames expected, tried to write "
                             f"'{self._current_frame + 1}' frames.")

        if(data.get_bodypart_count() != len(self._header.bodypart_names)):
            raise ValueError(f"'{data.get_bodypart_count()}' body parts does not match the "
                             f"'{len(self._header.bodypart_names)}' body parts specified in the header.")

        for frm_idx in range(data.get_frame_count()):
            for bp in range(data.get_bodypart_count()):
                frame = data.get_prob_table(frm_idx, bp)
                offset_table = data.get_offset_map()

                if (offset_table is not None):
                    off_y = offset_table[frm_idx, :, :, bp, 1]
                    off_x = offset_table[frm_idx, :, :, bp, 0]
                else:
                    off_y = None
                    off_x = None

                if(self._threshold is not None):
                    # Sparsify the data by removing everything below the threshold...
                    sparse_y, sparse_x = np.nonzero(frame > self._threshold)
                    probs = frame[(sparse_y, sparse_x)]

                    # Check if we managed to strip out at least 2/3rds of the data, and if so write the frame using the
                    # sparse format. Otherwise it is actually more memory efficient to just store the entire frame...
                    if(len(frame.flat) >= (len(sparse_y) * DLCFSConstants.MIN_SPARSE_SAVING_FACTOR)):
                        # Sparse indicator flag and the offsets included flag...
                        self._out_file.write(to_bytes(True | ((offset_table is not None) << 1), luint8))
                        # COMPRESSED DATA:
                        buffer = BytesIO()
                        buffer.write(to_bytes(len(sparse_y), luint64))  # The length of the sparse data entries.
                        buffer.write(sparse_y.astype(luint32).tobytes('C'))  # Y coord data
                        buffer.write(sparse_x.astype(luint32).tobytes('C'))  # X coord data
                        buffer.write(probs.astype(lfloat).tobytes('C'))  # Probabilities
                        if(offset_table is not None): # If offset table exists, write y offsets and then x offsets.
                            buffer.write(off_y[(sparse_y, sparse_x)].astype(lfloat).tobytes('C'))
                            buffer.write(off_x[(sparse_y, sparse_x)].astype(lfloat).tobytes('C'))
                        # Compress the sparse data and write it's length, followed by itself....
                        comp_data = zlib.compress(buffer.getvalue(), self._compression_level)
                        self._out_file.write(to_bytes(len(comp_data), luint64))
                        self._out_file.write(comp_data)

                        continue
                # If sparse optimization mode is off or the sparse format wasted more space, just write the entire
                # frame...
                self._out_file.write(to_bytes(False | ((offset_table is not None) << 1), luint8))

                buffer = BytesIO()
                buffer.write(frame.astype(lfloat).tobytes('C')) # The probability frame...
                if(offset_table is not None): # Y, then X offset data if it exists...
                    buffer.write(off_y.astype(lfloat).tobytes('C'))
                    buffer.write(off_x.astype(lfloat).tobytes('C'))

                comp_data = zlib.compress(buffer.getvalue())
                self._out_file.write(to_bytes(len(comp_data), luint64))
                self._out_file.write(comp_data)

