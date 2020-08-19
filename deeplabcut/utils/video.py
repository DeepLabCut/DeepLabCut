import cv2
import datetime
import numpy as np
import os
import subprocess
import warnings


class VideoReader:
    def __init__(self, video_path):
        if not os.path.isfile(video_path):
            raise ValueError(f'Video path "{video_path}" does not point to a file.')
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise IOError('Video could not be opened; it may be corrupted.')
        self.parse_metadata()
        self._bbox = 0, 1, 0, 1
        self._n_frames_robust = None

    def __repr__(self):
        string = 'Video (duration={:0.2f}, fps={}, dimensions={}x{})'
        return string.format(self.calc_duration(), self.fps, *self.dimensions)

    def __len__(self):
        return self._n_frames

    def check_integrity(self):
        dest = os.path.join(self.directory, f'{self.name}.log')
        command = f'ffmpeg -v debug -i {self.video_path} -f null - 2>{dest}'
        subprocess.call(command, shell=True)
        if os.path.getsize(dest) != 0:
            warnings.warn(f'Video contains errors. See "{dest}" for a detailed report.')

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
            n_frames=len(self),
            fps=self.fps,
            width=self.width,
            height=self.height
        )

    def get_n_frames(self, robust=False):
        if not robust:
            return self._n_frames
        elif not self._n_frames_robust:
            command = f'ffprobe -i "{self.video_path}" -v error -count_frames ' \
                      f'-select_streams v:0 -show_entries stream=nb_read_frames ' \
                      f'-of default=nokey=1:noprint_wrappers=1'
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            self._n_frames_robust = int(output)
        return self._n_frames_robust

    def calc_duration(self, robust=True):
        if robust:
            command = f'ffprobe -i "{self.video_path}" -show_entries ' \
                      f'format=duration -v quiet -of csv="p=0"'
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            return float(output)
        return len(self) / self.fps

    def set_to_frame(self, ind):
        if ind < 0:
            raise ValueError('Index must be a positive integer.')
        last_frame = len(self) - 1
        if ind > last_frame:
            warnings.warn('Index exceeds the total number of frames. '
                          'Setting to last frame instead.')
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
            frame = cv2.resize(frame, (w // shrink, h // shrink), fx=0, fy=0,
                               interpolation=cv2.INTER_AREA)
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
            raise ValueError('Frame rate should be positive.')
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
            warnings.warn('The video has more than 10^9 frames, we recommend chopping it up.')
        self._width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = round(self.video.get(cv2.CAP_PROP_FPS), 2)

    def close(self):
        self.video.release()


class VideoWriter(VideoReader):
    def __init__(self, video_path, codec='h264', dpi=100, fps=None):
        super(VideoWriter, self).__init__(video_path)
        self.codec = codec
        self.dpi = dpi
        if fps:
            self.fps = fps

    def set_bbox(self, x1, x2, y1, y2, relative=False):
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'Coordinates look wrong... '
                             f'Ensure {x1} < {x2} and {y1} < {y2}.')
        if not relative:
            x1 /= self._width
            x2 /= self._width
            y1 /= self._height
            y2 /= self._height
        bbox = x1, x2, y1, y2
        if any(coord > 1 for coord in bbox):
            warnings.warn('Bounding box larger than the video... '
                          'Clipping to video dimensions.')
            bbox = tuple(map(lambda x: min(x, 1), bbox))
        self._bbox = bbox

    def shorten(self, start, end, suffix='short', dest_folder=None,
                validate_inputs=True):
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
                raise ValueError('Timestamp should be a string formatted '
                                 'as hours:minutes:seconds.')
            time = datetime.datetime.strptime(stamp, '%H:%M:%S').time()
            # The above already raises a ValueError if formatting is wrong
            seconds = (time.hour * 60 + time.minute) * 60 + time.second
            if seconds > self.calc_duration():
                raise ValueError('Timestamps must not exceed the video duration.')

        if validate_inputs:
            for stamp in start, end:
                validate_timestamp(stamp)

        output_path = self.make_output_path(suffix, dest_folder)
        command = f'ffmpeg -n -i {self.video_path} -ss {start} -to {end} ' \
                  f'-vcodec {self.codec} -c:a copy {output_path}'
        subprocess.call(command, shell=True)
        return output_path

    def split(self, n_splits, suffix='split', dest_folder=None):
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
            raise ValueError('The video should at least be split in half.')
        chunk_dur = self.calc_duration() / n_splits
        splits = np.arange(n_splits + 1) * chunk_dur
        time_formatter = lambda val: str(datetime.timedelta(seconds=val))
        clips = []
        for n, (start, end) in enumerate(zip(splits, splits[1:]), start=1):
            clips.append(self.shorten(time_formatter(start),
                                      time_formatter(end),
                                      f'{suffix}{n}',
                                      dest_folder,
                                      validate_inputs=False))
        return clips

    def crop(self, suffix='crop', dest_folder=None):
        x1, _, y1, _ = self.get_bbox()
        output_path = self.make_output_path(suffix, dest_folder)
        command = f'ffmpeg -n -i {self.video_path} ' \
                  f'-filter:v crop={self.width}:{self.height}:{x1}:{y1} ' \
                  f'-vcodec {self.codec} -c:a copy {output_path}'
        subprocess.call(command, shell=True)
        return output_path

    def rescale(self, width, height=-1, rotateccw=False,
                suffix='rescale', dest_folder=None):
        output_path = self.make_output_path(suffix, dest_folder)
        command = f'ffmpeg -n -i {self.video_path} -filter:v ' \
                  f'scale={width}:{height} {{}}-vcodec {self.codec} -c:a copy {output_path}'
        # Rotate, see: https://stackoverflow.com/questions/3937387/rotating-videos-with-ffmpeg
        # interesting option to just update metadata.
        command = command.format("-vf 'transpose=1' ") if rotateccw else command.format('')
        subprocess.call(command, shell=True)
        return output_path

    @staticmethod
    def write_frame(frame, where):
        cv2.imwrite(where, frame[..., ::-1])

    def make_output_path(self, suffix, dest_folder):
        if not dest_folder:
            dest_folder = self.directory
        return os.path.join(dest_folder, f'{self.name}{suffix}{self.format}')
