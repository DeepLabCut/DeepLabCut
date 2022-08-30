#!/usr/bin/env python3
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import skimage.color
from skimage import io
from skimage.util import img_as_ubyte
import cv2
import datetime
import numpy as np
import os
import subprocess
import warnings


# more videos are in principle covered, as OpenCV is used and allows many formats.
SUPPORTED_VIDEOS = 'avi', 'mp4', 'mov', 'mpeg', 'mpg', 'mpv', 'mkv', 'flv', 'qt', 'yuv'



class VideoReader:
    def __init__(self, video_path):
        if not os.path.isfile(video_path):
            raise ValueError(f'Video path "{video_path}" does not point to a file.')
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise IOError("Video could not be opened; it may be corrupted.")
        self.parse_metadata()
        self._bbox = 0, 1, 0, 1
        self._n_frames_robust = None

    def __repr__(self):
        string = "Video (duration={:0.2f}, fps={}, dimensions={}x{})"
        return string.format(self.calc_duration(), self.fps, *self.dimensions)

    def __len__(self):
        return self._n_frames

    def check_integrity(self):
        dest = os.path.join(self.directory, f"{self.name}.log")
        command = f"ffmpeg -v error -i {self.video_path} -f null - 2>{dest}"
        subprocess.call(command, shell=True)
        if os.path.getsize(dest) != 0:
            warnings.warn(f'Video contains errors. See "{dest}" for a detailed report.')

    def check_integrity_robust(self):
        numframes = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        fr = 0
        while fr < numframes:
            success, frame = self.video.read()
            if not success or frame is None:
                warnings.warn(
                    f"Opencv failed to load frame {fr}. Use ffmpeg to re-encode video file"
                )
            fr += 1

    @property
    def name(self):
        return os.path.splitext(os.path.split(self.video_path)[1])[0]

    @property
    def format(self):
        return os.path.splitext(self.video_path)[1]

    @property
    def directory(self):
        return os.path.dirname(self.video_path)

    @property
    def metadata(self):
        return dict(
            n_frames=len(self), fps=self.fps, width=self.width, height=self.height
        )

    def get_n_frames(self, robust=False):
        if not robust:
            return self._n_frames
        elif not self._n_frames_robust:
            command = (
                f'ffprobe -i "{self.video_path}" -v error -count_frames '
                f"-select_streams v:0 -show_entries stream=nb_read_frames "
                f"-of default=nokey=1:noprint_wrappers=1"
            )
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            self._n_frames_robust = int(output)
        return self._n_frames_robust

    def calc_duration(self, robust=False):
        if robust:
            command = (
                f'ffprobe -i "{self.video_path}" -show_entries '
                f'format=duration -v quiet -of csv="p=0"'
            )
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            return float(output)
        return len(self) / self.fps

    def set_to_frame(self, ind):
        if ind < 0:
            raise ValueError("Index must be a positive integer.")
        last_frame = len(self) - 1
        if ind > last_frame:
            warnings.warn(
                "Index exceeds the total number of frames. "
                "Setting to last frame instead."
            )
            ind = last_frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, ind)

    def reset(self):
        self.set_to_frame(0)

    def read_frame(self, shrink=1, crop=False):
        success, frame = self.video.read()
        if not success:
            return
        frame = frame[..., ::-1]  # return RGB rather than BGR!
        if crop:
            x1, x2, y1, y2 = self.get_bbox(relative=False)
            frame = frame[y1:y2, x1:x2]
        if shrink > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (w // shrink, h // shrink),
                fx=0,
                fy=0,
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def get_bbox(self, relative=False):
        x1, x2, y1, y2 = self._bbox
        if not relative:
            x1 = int(self._width * x1)
            x2 = int(self._width * x2)
            y1 = int(self._height * y1)
            y2 = int(self._height * y2)
        return x1, x2, y1, y2

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        if not fps > 0:
            raise ValueError("Frame rate should be positive.")
        self._fps = fps

    @property
    def width(self):
        x1, x2, _, _ = self.get_bbox(relative=False)
        return x2 - x1

    @property
    def height(self):
        _, _, y1, y2 = self.get_bbox(relative=False)
        return y2 - y1

    @property
    def dimensions(self):
        return self.width, self.height

    def parse_metadata(self):
        self._n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._n_frames >= 1e9:
            warnings.warn(
                "The video has more than 10^9 frames, we recommend chopping it up."
            )
        self._width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = round(self.video.get(cv2.CAP_PROP_FPS), 2)

    def close(self):
        self.video.release()


class VideoWriter(VideoReader):
    def __init__(self, video_path, codec="h264", dpi=100, fps=None):
        super(VideoWriter, self).__init__(video_path)
        self.codec = codec
        self.dpi = dpi
        if fps:
            self.fps = fps

    def set_bbox(self, x1, x2, y1, y2, relative=False):
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Coordinates look wrong... " f"Ensure {x1} < {x2} and {y1} < {y2}."
            )
        if not relative:
            x1 /= self._width
            x2 /= self._width
            y1 /= self._height
            y2 /= self._height
        bbox = x1, x2, y1, y2
        if any(coord > 1 for coord in bbox):
            warnings.warn(
                "Bounding box larger than the video... " "Clipping to video dimensions."
            )
            bbox = tuple(map(lambda x: min(x, 1), bbox))
        self._bbox = bbox

    def shorten(
        self, start, end, suffix="short", dest_folder=None, validate_inputs=True
    ):
        """
        Shorten the video from start to end.

        Parameter
        ----------
        start: str
            Time formatted in hours:minutes:seconds, where shortened video shall start.

        end: str
            Time formatted in hours:minutes:seconds, where shortened video shall end.

        suffix: str, optional
            String added to the name of the shortened video ('short' by default).

        dest_folder: str, optional
            Folder the video is saved into (by default, same as the original video)

        Returns
        -------
        str
            Full path to the shortened video
        """

        def validate_timestamp(stamp):
            if not isinstance(stamp, str):
                raise ValueError(
                    "Timestamp should be a string formatted "
                    "as hours:minutes:seconds."
                )
            time = datetime.datetime.strptime(stamp, "%H:%M:%S").time()
            # The above already raises a ValueError if formatting is wrong
            seconds = (time.hour * 60 + time.minute) * 60 + time.second
            if seconds > self.calc_duration():
                raise ValueError("Timestamps must not exceed the video duration.")

        if validate_inputs:
            for stamp in start, end:
                validate_timestamp(stamp)

        output_path = self.make_output_path(suffix, dest_folder)
        command = (
            f"ffmpeg -n -i {self.video_path} -ss {start} -to {end} "
            f"-c:a copy {output_path}"
        )
        subprocess.call(command, shell=True)
        return output_path

    def split(self, n_splits, suffix="split", dest_folder=None):
        """
        Split a video into several shorter ones of equal duration.

        Parameters
        ----------
        n_splits : int
            Number of shorter videos to produce

        suffix: str, optional
            String added to the name of the splits ('short' by default).

        dest_folder: str, optional
            Folder the video splits are saved into (by default, same as the original video)

        Returns
        -------
        list
            Paths of the video splits
        """
        if not n_splits > 1:
            raise ValueError("The video should at least be split in half.")
        chunk_dur = self.calc_duration() / n_splits
        splits = np.arange(n_splits + 1) * chunk_dur
        time_formatter = lambda val: str(datetime.timedelta(seconds=val))
        clips = []
        for n, (start, end) in enumerate(zip(splits, splits[1:]), start=1):
            clips.append(
                self.shorten(
                    time_formatter(start),
                    time_formatter(end),
                    f"{suffix}{n}",
                    dest_folder,
                    validate_inputs=False,
                )
            )
        return clips

    def crop(self, suffix="crop", dest_folder=None):
        x1, _, y1, _ = self.get_bbox()
        output_path = self.make_output_path(suffix, dest_folder)
        command = (
            f'ffmpeg -n -i "{self.video_path}" '
            f"-filter:v crop={self.width}:{self.height}:{x1}:{y1} "
            f'-c:a copy "{output_path}"'
        )
        subprocess.call(command, shell=True)
        return output_path

    def rescale(
        self,
        width,
        height=-1,
        rotatecw="No",
        angle=0.0,
        suffix="rescale",
        dest_folder=None,
    ):
        output_path = self.make_output_path(suffix, dest_folder)
        command = (
            f"ffmpeg -n -i {self.video_path} -filter:v "
            f'"scale={width}:{height}{{}}" -c:a copy {output_path}'
        )
        # Rotate, see: https://stackoverflow.com/questions/3937387/rotating-videos-with-ffmpeg
        # interesting option to just update metadata.
        if rotatecw == "Arbitrary":
            angle = np.deg2rad(angle)
            command = command.format(f", rotate={angle}")
        elif rotatecw == "Yes":
            command = command.format(f", transpose=1")
        else:
            command = command.format("")
        subprocess.call(command, shell=True)
        return output_path

    @staticmethod
    def write_frame(frame, where):
        cv2.imwrite(where, frame[..., ::-1])

    def make_output_path(self, suffix, dest_folder):
        if not dest_folder:
            dest_folder = self.directory
        return os.path.join(dest_folder, f"{self.name}{suffix}{self.format}")


def check_video_integrity(video_path):
    vid = VideoReader(video_path)
    vid.check_integrity()
    vid.check_integrity_robust()

def imread(image_path, mode="skimage"):
    ''' Read image either with skimage or cv2.
    Returns frame in uint with 3 color channels. '''
    if mode == "skimage":
        image = io.imread(image_path)
        if image.ndim == 2 or image.shape[-1] == 1:
            image = skimage.color.gray2rgb(image)
        elif image.shape[-1] == 4:
            image = skimage.color.rgba2rgb(image)

        return img_as_ubyte(image)

    elif mode=="cv2":
        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[..., ::-1]  # ~10% faster than using cv2.cvtColor


# https://docs.opencv.org/3.4.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
def imresize(img, size=1.0, interpolationmethod=cv2.INTER_AREA):
    if size != 1.0:
        return cv2.resize(
            img, None, fx=size, fy=size, interpolation=interpolationmethod
        )  # (int(height*size),int(width*size)))
    else:
        return img


def ShortenVideo(
    vname, start="00:00:01", stop="00:01:00", outsuffix="short", outpath=None
):
    """
    Auxiliary function to shorten video and output with outsuffix appended.
    to the same folder from start (hours:minutes:seconds) to stop (hours:minutes:seconds).

    Returns the full path to the shortened video!

    Parameter
    ----------
    videos : string
        A string containing the full paths of the video.

    start: hours:minutes:seconds
        Time formatted in hours:minutes:seconds, where shortened video shall start.

    stop: hours:minutes:seconds
        Time formatted in hours:minutes:seconds, where shortened video shall end.

    outsuffix: str
        Suffix for output videoname (see example).

    outpath: str
        Output path for saving video to (by default will be the same folder as the video)

    Examples
    ----------

    Linux/MacOs
    >>> deeplabcut.ShortenVideo('/data/videos/mouse1.avi')

    Extracts (sub)video from 1st second to 1st minutes (default values) and saves it in /data/videos as mouse1short.avi

    Windows:
    >>> deeplabcut.ShortenVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', start='00:17:00',stop='00:22:00',outsuffix='brief')

    Extracts (sub)video from minute 17 to 22 and and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1brief.avi
    """
    writer = VideoWriter(vname)
    return writer.shorten(start, stop, outsuffix, outpath)


def CropVideo(
    vname,
    width=256,
    height=256,
    origin_x=0,
    origin_y=0,
    outsuffix="cropped",
    outpath=None,
    useGUI=False,
):
    """
    Auxiliary function to crop a video and output it to the same folder with "outsuffix" appended in its name.
    Width and height will control the new dimensions.

    Returns the full path to the downsampled video!

    ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4

    Parameter
    ----------
    vname : string
        A string containing the full path of the video.

    width: int
        width of output video

    height: int
        height of output video.

    origin_x, origin_y: int
        x- and y- axis origin of bounding box for cropping.

    outsuffix: str
        Suffix for output videoname (see example).

    outpath: str
        Output path for saving video to (by default will be the same folder as the video)

    Examples
    ----------

    Linux/MacOs
    >>> deeplabcut.CropVideo('/data/videos/mouse1.avi')

    Crops the video using default values and saves it in /data/videos as mouse1cropped.avi

    Windows:
    >>> =deeplabcut.CropVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', width=220,height=320,outsuffix='cropped')

    Crops the video to a width of 220 and height of 320 starting at the origin (top left) and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1cropped.avi
    """
    writer = VideoWriter(vname)

    if useGUI:
        print(
            "Please, select your coordinates (draw from top left to bottom right ...)"
        )
        coords = draw_bbox(vname)

        if not coords:
            return
        origin_x, origin_y = coords[:2]
        width = int(coords[2]) - int(coords[0])
        height = int(coords[3]) - int(coords[1])

    writer.set_bbox(origin_x, origin_x + width, origin_y, origin_y + height)
    return writer.crop(outsuffix, outpath)


def DownSampleVideo(
    vname,
    width=-1,
    height=200,
    outsuffix="downsampled",
    outpath=None,
    rotatecw="No",
    angle=0.0,
):
    """
    Auxiliary function to downsample a video and output it to the same folder with "outsuffix" appended in its name.
    Width and height will control the new dimensions. You can also pass only height or width and set the other one to -1,
    this will keep the aspect ratio identical.

    Returns the full path to the downsampled video!

    Parameter
    ----------
    vname : string
        A string containing the full path of the video.

    width: int
        width of output video

    height: int
        height of output video.

    outsuffix: str
        Suffix for output videoname (see example).

    outpath: str
        Output path for saving video to (by default will be the same folder as the video)

    rotatecw: str
        Default "No", rotates clockwise if "Yes", "Arbitrary" for arbitrary rotation by specified angle.

    angle: float
        Angle to rotate by in degrees, default 0.0. Negative values rotate counter-clockwise

    Examples
    ----------

    Linux/MacOs
    >>> deeplabcut.DownSampleVideo('/data/videos/mouse1.avi')

    Downsamples the video using default values and saves it in /data/videos as mouse1cropped.avi

    Windows:
    >>> shortenedvideoname=deeplabcut.DownSampleVideo('C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi', width=220,height=320,outsuffix='cropped')

    Downsamples the video to a width of 220 and height of 320 and saves it in C:\\yourusername\\rig-95\\Videos as reachingvideo1cropped.avi
    """
    writer = VideoWriter(vname)
    return writer.rescale(width, height, rotatecw, angle, outsuffix, outpath)


def draw_bbox(video):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector, Button

    clip = VideoWriter(video)
    frame = None
    # Read the video until a frame is successfully read
    while frame is None:
        frame = clip.read_frame()

    bbox = [0, 0, frame.shape[1], frame.shape[0]]

    def line_select_callback(eclick, erelease):
        bbox[:2] = int(eclick.xdata), int(eclick.ydata)  # x1, y1
        bbox[2:] = int(erelease.xdata), int(erelease.ydata)  # x2, y2

    def validate_crop(*args):
        fig.canvas.stop_event_loop()

    def display_help(*args):
        print(
            "1. Use left click to select the region of interest. A red box will be drawn around the selected region. \n\n2. Use the corner points to expand the box and center to move the box around the image. \n\n3. Click "
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(frame[:, :, ::-1])
    ax_help = fig.add_axes([0.9, 0.2, 0.1, 0.1])
    ax_save = fig.add_axes([0.9, 0.1, 0.1, 0.1])
    crop_button = Button(ax_save, "Crop")
    crop_button.on_clicked(validate_crop)
    help_button = Button(ax_help, "Help")
    help_button.on_clicked(display_help)

    rs = RectangleSelector(
        ax,
        line_select_callback,
        drawtype="box",
        minspanx=5,
        minspany=5,
        interactive=True,
        spancoords="pixels",
        rectprops=dict(facecolor="red", edgecolor="black", alpha=0.3, fill=True),
    )
    plt.show()

    # import platform
    # if platform.system() == "Darwin":  # for OSX use WXAgg
    #    fig.canvas.start_event_loop(timeout=-1)
    # else:
    fig.canvas.start_event_loop(timeout=-1)  # just tested on Ubuntu I also need this.
    #    #fig.canvas.stop_event_loop()

    plt.close(fig)
    return bbox
