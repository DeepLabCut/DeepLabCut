
# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable

import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData

# Used specifically by plugin...
import math
import numpy as np
from matplotlib import pyplot
import matplotlib
from pathlib import Path
import cv2


class PlotterArgMax(Predictor):
    """
    Identical to singleargmax, but plots probability frames in form of video to the user using matplotlib...
    """
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._parts = bodyparts
        self._num_frames = num_frames
        self._num_outputs = num_outputs

        # Keeps track of how many frames
        self._current_frame = 0
        # Determines grid size of charts
        self._grid_width = int(math.ceil(math.sqrt(len(self._parts))))
        self._grid_height = int(math.ceil(len(self._parts) / self._grid_width))
        # Stores opencv video writer...
        self._vid_writer = None

        # Name of the video file to save to
        final_video_name = settings["video_name"].replace("$VIDEO", Path(video_metadata["orig-video-path"]).stem)
        self.VIDEO_NAME = str((Path(video_metadata["h5-file-name"]).parent) / final_video_name)
        # Output codec to save using
        self.OUTPUT_CODEC = cv2.VideoWriter_fourcc(*settings["codec"])
        # Frames per second to use for video
        self.OUTPUT_FPS = video_metadata["fps"]
        # Determines if we are using log scaling...
        self.LOG_SCALE = settings["use_log_scale"]
        # Determines if we are using 3d projection...
        self.PROJECT_3D = settings["3d_projection"]
        self.Z_SHRINK_F = settings["z_shrink_factor"]
        # The colormap to use while plotting...
        self.COLOR_MAP = settings["colormap"]
        self.DPI = settings["dpi"]
        self.AXIS_ON = settings["axis_on"]
        # Miscellaneous arguments to pass to the figure and axis...
        self.AXES_ARGS = settings["axes_args"] if (isinstance(settings["axes_args"], dict)) else {}
        self.FIGURE_ARGS = settings["figure_args"] if (isinstance(settings["figure_args"], dict)) else {}
        self.FIGURE_ARGS.update({"dpi": self.DPI})

        # Build the subplots...
        if(self.PROJECT_3D):
            self.AXES_ARGS.update({'projection': '3d'})
            from mpl_toolkits.mplot3d import Axes3D
            self._figure, self._axes = pyplot.subplots(self._grid_height, self._grid_width,
                                                       subplot_kw=self.AXES_ARGS, **self.FIGURE_ARGS)
        else:
            self._figure, self._axes = pyplot.subplots(self._grid_height, self._grid_width, subplot_kw=self.AXES_ARGS,
                                                       **self.FIGURE_ARGS)
        # Hide all axis.....
        if(not self.AXIS_ON):
            for ax in self._axes.flat:
                ax.axis("off")


    def _logify(self, arr: np.ndarray) -> np.ndarray:
        """
        Place the array in log scale, and then place the values between 0 and 1 using simple linear interpolation...
        """
        with np.errstate(divide='ignore'):
            arr_logged = np.log(arr)
            was_zero = np.isneginf(arr_logged)
            not_zero = ~was_zero
            low_val = np.min(arr_logged[not_zero])

            arr_logged[not_zero] = (np.abs(low_val) - np.abs(arr_logged[not_zero])) / np.abs(low_val)
            arr_logged[was_zero] = 0

            return arr_logged


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        for frame in range(scmap.get_frame_count()):
            # Plot all probability maps
            for bp, ax in zip(range(scmap.get_bodypart_count()), self._axes.flat):
                ax.clear()
                ax.set_aspect("equal")
                if(not self.AXIS_ON):
                    ax.axis("off")

                ax.set_title(f"Bodypart: {self._parts[bp]}, Frame: {self._current_frame}")

                if(self.PROJECT_3D):
                    x, y = np.arange(scmap.get_frame_width()), np.arange(scmap.get_frame_height())
                    x, y = np.meshgrid(x, y)
                    z = self._logify(scmap.get_prob_table(frame, bp)) if (self.LOG_SCALE) else \
                        scmap.get_prob_table(frame, bp)
                    ax.plot_surface(x, y, z, cmap=self.COLOR_MAP)
                    z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
                    ax.set_zlim(ax.get_zlim()[0], ax.get_zlim()[0] + (z_range * self.Z_SHRINK_F))
                else:
                    ax.pcolormesh(self._logify(scmap.get_prob_table(frame, bp)) if (self.LOG_SCALE) else
                                      scmap.get_prob_table(frame, bp), cmap=self.COLOR_MAP)
                # This reverses the y-axis data, so as probability maps match the video...
                ax.set_ylim(ax.get_ylim()[::-1])

            # Save chart to the buffer
            self._figure.tight_layout()
            self._figure.canvas.draw()

            img = np.reshape(np.frombuffer(self._figure.canvas.tostring_rgb(), dtype="uint8"),
                               self._figure.canvas.get_width_height()[::-1] + (3,))[:, :, ::-1]

            if (self._vid_writer is None):
                height, width, colors = img.shape
                self._vid_writer = cv2.VideoWriter(self.VIDEO_NAME, self.OUTPUT_CODEC, self.OUTPUT_FPS,
                                                   (width, height))

            self._vid_writer.write(img)
            self._current_frame += 1

        # Return argmax values for each frame...
        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        self._vid_writer.release()
        return None

    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("video_name", "Name of the video file that plotting data will be saved to. Can use $VIDEO to place the "
                           "name of original video somewhere in the text.", "$VIDEO-prob-dlc.mp4"),
            ("codec", "The codec to be used by the opencv library to save info to, typically a 4-byte string.", "mp4v"),
            ("use_log_scale", "Boolean, determines whether to apply log scaling to the frames in the video.", False),
            ("3d_projection", "Boolean, determines if probability frames should be plotted in 3d.", False),
            ("colormap", "String, determines the underlying colormap to be passed to matplotlib while plotting the "
                         "mesh.", "Blues"),
            ("z_shrink_factor", "Float, determines how much to shrink the z-axis if in 3D mode...", 5),
            ("dpi", "The dpi of the final video, the higher the dpi the more crisp...",
             matplotlib.rcParams['figure.dpi']),
            ("axis_on", "Boolean, determines if axis, or tick marks and grids of subplots are shown.", False),
            ("axes_args", "A dictionary, miscellaneous arguments to pass to matplotlib axes when constructing them.",
             None),
            ("figure_args", "A dictionary, miscellaneous arguments to pass to matplotlib figure when constructing it.",
             None)
        ]

    @classmethod
    def get_name(cls) -> str:
        return "plotterargmax"

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True