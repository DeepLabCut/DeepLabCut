from typing import Union, List, Callable, Tuple, Any, Dict

import cv2
import tqdm
import math
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import as_strided

from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData


class FastPlotterArgMax(Predictor):
    """
    Identical to plotterargmax, but avoids using matplotlib to generate probability maps, and instead
    directly uses cv2 to generate the plots. This means it runs much faster, but doesn't offer as much
    customization nor a 3D mode...
    """

    TEST_TEXT = "".join(chr(i) for i in range(32, 127))

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
        self._vid_writer: cv2.VideoWriter = None

        # Name of the video file to save to
        final_video_name = settings["video_name"].replace("$VIDEO", Path(video_metadata["orig-video-path"]).stem)
        self.VIDEO_NAME = str((Path(video_metadata["h5-file-name"]).parent) / final_video_name)
        # Output codec to save using
        self.OUTPUT_CODEC = cv2.VideoWriter_fourcc(*settings["codec"])
        # Frames per second to use for video
        self.OUTPUT_FPS = video_metadata["fps"]
        # Determines if we are using log scaling...
        self.LOG_SCALE = settings["use_log_scale"]
        # All font settings
        font_dict = _load_cv2_fonts()
        self.TITLE_FONT = font_dict[settings["title_font"]]
        self.TITLE_FONT_SIZE = settings["title_font_size"]
        self.SUBPLOT_FONT = font_dict[settings["subplot_font"]]
        self.SUBPLOT_FONT_SIZE = settings["subplot_font_size"]
        self.FONT_THICKNESS = settings["font_thickness"]
        # Compute the height of the titles and subtitles...
        (__, self._title_font_height), self._title_baseline = cv2.getTextSize(self.TEST_TEXT, self.TITLE_FONT,
                                                                              self.TITLE_FONT_SIZE, self.FONT_THICKNESS)
        (__, self._subplot_font_height), self._subplot_baseline = cv2.getTextSize(self.TEST_TEXT, self.SUBPLOT_FONT,
                                                                                  self.SUBPLOT_FONT_SIZE,
                                                                                  self.FONT_THICKNESS)
        # Get all of the colors and the colormap
        self.BACKGROUND_COLOR = settings["background_color"]
        self.TITLE_COLOR = settings["title_font_color"]
        self.SUBPLOT_COLOR = settings["subplot_font_color"]
        colormap_dict = _load_cv2_colormaps()
        self.COLORMAP = colormap_dict[settings["colormap"]]
        self.MULTIPLIER = settings["source_map_upscale"]
        # Store the padding...
        self.PADDING = settings["padding"]

        # Variable for math later when computing locations of things in the video...
        self._vid_height = None
        self._vid_width = None
        self._subplot_height = None
        self._subplot_width = None
        self._scmap_height = None
        self._scmap_width = None
        self._canvas = None # The numpy array we will use for drawing...
        # Will store colormap per run to avoid reallocating large arrays over and over....
        self._colormap_temp = None
        self._colormap_view = None


    def _compute_video_measurements(self, scmap_width: int, scmap_height: int):
        """
        Compute all required measurements needed to render text/source maps to the correct locations, and also
        initialize the video writer...
        """
        self._scmap_width = scmap_width * self.MULTIPLIER
        self._scmap_height = scmap_height * self.MULTIPLIER

        self._subplot_width = (self.PADDING * 2) + self._scmap_width
        total_subplot_text_height = self._subplot_font_height + self._subplot_baseline
        self._subplot_height = (self.PADDING * 2) + total_subplot_text_height + self._scmap_height

        self._vid_width = self._grid_width * self._subplot_width
        self._vid_height = (self._grid_height * self._subplot_height) + (self._title_font_height + self._title_baseline)

        self._canvas = np.zeros((self._vid_height, self._vid_width, 3), dtype=np.uint8)

        self._vid_writer = cv2.VideoWriter(self.VIDEO_NAME, self.OUTPUT_CODEC, self.OUTPUT_FPS,
                                           (self._vid_width, self._vid_height))
        # Array which stores color maps temporarily... Takes advantage of numpy's abilities to make custom strides
        # to access data... The colormap_view maps the
        self._colormap_temp = np.zeros((self._scmap_height, self._scmap_width, 3), dtype=np.uint8)
        shape, strides = self._colormap_temp.shape, self._colormap_temp.strides
        view_shape = (self.MULTIPLIER, self.MULTIPLIER, shape[0] // self.MULTIPLIER, shape[1] // self.MULTIPLIER,
                      shape[2])
        view_strides = (strides[0], strides[1], strides[0] * self.MULTIPLIER, strides[1] * self.MULTIPLIER, strides[2])
        self._colormap_view = as_strided(self._colormap_temp, shape=view_shape, strides=view_strides)
        self._unscaled_cmap_temp = np.zeros((scmap_height, scmap_width, 3), dtype=np.uint8)


    def _probs_to_grayscale(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert numpy probability array into a grayscale image of unsigned 8 bit integers.
        """
        return (arr * 255).astype(dtype=np.uint8)


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


    def _draw_title(self, text: str):
        """
        Draws the title text to the video frame....
        """
        (width, __), __ = cv2.getTextSize(text, self.TITLE_FONT, self.TITLE_FONT_SIZE, self.FONT_THICKNESS)
        x_in = max(int((self._vid_width - width) / 2), 0)
        y_in = (self._title_font_height + self._title_baseline) - 1
        cv2.putText(self._canvas, text, (x_in, y_in), self.TITLE_FONT, self.TITLE_FONT_SIZE, self.TITLE_COLOR,
                    self.FONT_THICKNESS)


    def _draw_subplot(self, bp_name: str, grid_x: int, grid_y: int, prob_map: np.ndarray):
        """
        Draws a single subplot for the provided frame...

        :param bp_name: The name of the body part...
        :param grid_x: The x grid location to draw this probability map at.
        :param grid_y: The y grid location to draw this probability map at.
        :param prob_map: The probability map, an array of 2D floats...
        """
        x_upper_corner = grid_x * self._subplot_width
        y_upper_corner = (grid_y * self._subplot_height) + (self._title_font_height + self._title_baseline)

        # Convert probabilities to a color image...
        grayscale_img = self._probs_to_grayscale(self._logify(prob_map) if(self.LOG_SCALE) else prob_map)
        self._colormap_view[:, :] = cv2.applyColorMap(grayscale_img, self.COLORMAP, self._unscaled_cmap_temp)
        # Insert the probability map...
        subplot_top_x, subplot_top_y = (x_upper_corner + self.PADDING) - 1, (y_upper_corner + self.PADDING) - 1
        subplot_bottom_x, subplot_bottom_y = subplot_top_x + self._scmap_width, subplot_top_y + self._scmap_height
        self._canvas[subplot_top_y:subplot_bottom_y, subplot_top_x:subplot_bottom_x] = self._colormap_temp
        # Now insert the text....
        (text_width, __), __ = cv2.getTextSize(bp_name, self.SUBPLOT_FONT, self.SUBPLOT_FONT_SIZE, self.FONT_THICKNESS)
        x_text_root = x_upper_corner + max(int((self._subplot_width - text_width) / 2), 0)
        y_text_root = y_upper_corner + (self._subplot_height - self.PADDING)
        cv2.putText(self._canvas, bp_name, (x_text_root, y_text_root), self.SUBPLOT_FONT, self.SUBPLOT_FONT_SIZE,
                    self.SUBPLOT_COLOR, self.FONT_THICKNESS)


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # If the video writer has not been created, create it now and compute all needed video dimensions...
        if(self._vid_writer is None):
            self._compute_video_measurements(scmap.get_frame_width(), scmap.get_frame_height())

        for frame in range(scmap.get_frame_count()):
            # Clear the canvas with the background color...
            self._canvas[:] = self.BACKGROUND_COLOR
            # Drawing the title...
            self._draw_title(f"Frame {self._current_frame}")

            for bp in range(len(self._parts)):
                # Compute the current subplot we are on...
                subplot_y = bp // self._grid_width
                subplot_x = bp % self._grid_width
                self._draw_subplot(self._parts[bp], subplot_x, subplot_y, scmap.get_prob_table(frame, bp))

            self._vid_writer.write(self._canvas)

            self._current_frame += 1

        # Return just like argmax...
        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))

    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        self._vid_writer.release()
        return None


    @staticmethod
    def get_name() -> str:
        return "fast_plotterargmax"

    @staticmethod
    def get_description() -> str:
        return ("Identical to plotterargmax, but avoids using matplotlib to generate probability maps, and instead"
                "directly uses cv2 to generate the plots. This means it runs much faster, but doesn't offer as much"
                "customization nor a 3D mode...")

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        font_options = "\n".join([f"\t - {key}" for key in _load_cv2_fonts()])
        colormap_options = "\n".join(f"\t - {key}" for key in _load_cv2_colormaps())

        return [
            ("video_name", "Name of the video file that plotting data will be saved to. Can use $VIDEO to place the "
                           "name of original video somewhere in the text.", "$VIDEO-fast-prob-dlc.mp4"),
            ("codec", "The codec to be used by the opencv library to save info to, typically a 4-byte string.", "mp4v"),
            ("use_log_scale", "Boolean, determines whether to apply log scaling to the frames in the video.", False),
            ("title_font_size", "Float, the font size of the main title", 2),
            ("title_font", f"String, the cv2 font to be used in the title, options for this are:\n{font_options}",
             "FONT_HERSHEY_SIMPLEX"),
            ("subplot_font_size", "Float, the font size of the titles of each subplot.", 1.5),
            ("subplot_font", "String, the cv2 font used in the subplot titles, look at options for 'title_font'.",
             "FONT_HERSHEY_SIMPLEX"),
            ("background_color", "Tuple of 3 integers, color of the background in BGR format", (255, 255, 255)),
            ("title_font_color", "Tuple of 3 integers, color of the title text in BGR format", (0, 0, 0)),
            ("subplot_font_color", "Tuple of 3 integers, color of the title text in BGR format", (0, 0, 0)),
            ("colormap", f"String, the cv2 colormap to use, options for this are:\n{colormap_options}",
             "COLORMAP_VIRIDIS"),
            ("font_thickness", "Integer, the thickness of the font being drawn.", 2),
            ("source_map_upscale", "Integer, The amount to upscale the probability maps.", 8),
            ("padding", "Integer, the padding to be applied around plots in pixels.", 20)
        ]

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True


def _load_cv2_fonts() -> Dict[str, int]:
    """
    Loads all cv2 fonts available by default, returning a dictionary of string to option.
    """
    return {item: getattr(cv2, item) for item in dir(cv2) if(item.startswith("FONT_HERSHEY"))}


def _load_cv2_colormaps() -> Dict[str, int]:
    """
    Loads all cv2 colormaps available by default, placing them in a dictionary...
    """
    return {item: getattr(cv2, item) for item in dir(cv2) if(item.startswith("COLORMAP"))}


