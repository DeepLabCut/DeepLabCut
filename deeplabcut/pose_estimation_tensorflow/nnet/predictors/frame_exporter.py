"""
Package includes the frame exporter plugin. This plugin exports DeepLabCut probability maps to a binary format that can
be passed back into DeepLabCut again to perform frame predictions later. This allows for a video to be run through
the neural network (expensive) on a headless server or supercomputer, and then run through a predictor with gui
feedback on a laptop or somewhere else.
"""
from pathlib import Path
from typing import Union, List, Callable, Tuple, Any, Dict, BinaryIO
import tqdm
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData
from deeplabcut.pose_estimation_tensorflow.util.frame_store_fmt import DLCFSWriter, DLCFSHeader


class FrameExporter(Predictor):
    """
    Exports DeepLabCut probability maps to a binary format that can be passed back into DeepLabCut again to perform
    frame predictions later. This allows for a video to be run through the neural network (expensive) on a headless
    server or supercomputer, and then run through a predictor with gui feedback on a laptop or somewhere else.
    """

    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._num_frames = num_frames
        self._bodyparts = bodyparts
        self._video_metadata = video_metadata

        self._crop_off = self._video_metadata["cropping-offset"]
        self._crop_off = (None, None) if(self._crop_off is None) else self._crop_off

        self._num_outputs = num_outputs
        self._frame_writer = None
        # Making the output file...
        orig_h5_path = Path(video_metadata["h5-file-name"])
        vid_path = Path(video_metadata["orig-video-path"])
        self._out_file: BinaryIO = (orig_h5_path.parent / (vid_path.name + "~DATA.dlcf")).open("wb")
        # Load in the settings....
        self.SPARSIFY = settings["sparsify"]
        self.THRESHOLD = settings["threshold"]
        self.COMPRESSION_LEVEL = int(settings["compression_level"]) if(0 <= settings["compression_level"] <= 9) else 6
        # Initialize the frame counter...
        self._current_frame = 0


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # If we are just starting, write the header, body part names chunk, and magic for frame data chunk...
        if(self._current_frame == 0):
            header = DLCFSHeader(self._num_frames, scmap.get_frame_height(), scmap.get_frame_width(),
                                 self._video_metadata["fps"], scmap.get_down_scaling(),
                                 *self._video_metadata["size"], *self._crop_off, self._bodyparts)
            print(header)

            self._frame_writer = DLCFSWriter(self._out_file, header, self.THRESHOLD if(self.SPARSIFY) else None,
                                             self.COMPRESSION_LEVEL)

        # Writing all of the frames in this batch...
        self._frame_writer.write_data(scmap)
        self._current_frame += scmap.get_frame_count()

        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        self._out_file.flush()
        self._out_file.close()
        return None

    @classmethod
    def get_name(cls) -> str:
        return "frame_exporter"

    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("sparsify", "Boolean, specify whether optimize and store the data in a sparse format when it "
                         "saves storage", True),
            ("threshold", "A Float between 0 and 1. The threshold used if sparsify is true. Any values which land below "
                          "this threshold probability won't be included in the frame.", 1e-7),
            ("compression_level", "Integer, 0 through 9, determines the compression level. Higher compression level"
                                  "means it takes longer to compress the data, while 0 is no compression", 6)
        ]

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True