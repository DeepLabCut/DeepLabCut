#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os
import pytest
from conftest import TEST_DATA_DIR
from deeplabcut.utils.auxfun_videos import VideoWriter


POS_FRAMES = 1  # Equivalent to cv2.CAP_PROP_POS_FRAMES


@pytest.fixture()
def video_clip():
    return VideoWriter(os.path.join(TEST_DATA_DIR, "vid.avi"))


def test_reader_wrong_inputs(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(str(tmp_path))
    fake_vid = tmp_path / "fake.avi"
    fake_vid.write_bytes(b"42")
    with pytest.raises(IOError):
        VideoWriter(str(fake_vid))


def test_reader_check_integrity(video_clip):
    video_clip.check_integrity()
    log_file = os.path.join(video_clip.directory, f"{video_clip.name}.log")
    assert os.path.getsize(log_file) == 0


def test_reader_video_path(video_clip):
    assert video_clip.name == "vid"
    assert video_clip.format == ".avi"
    assert video_clip.directory == TEST_DATA_DIR


def test_reader_metadata(video_clip):
    metadata = video_clip.metadata
    assert metadata["n_frames"] == video_clip.get_n_frames(True) == 256
    assert metadata["fps"] == 30
    assert metadata["width"] == 416
    assert metadata["height"] == 374


def test_reader_wrong_fps(video_clip):
    with pytest.raises(ValueError):
        video_clip.fps = 0


def test_reader_duration(video_clip):
    assert video_clip.calc_duration() == pytest.approx(
        video_clip.calc_duration(robust=False), abs=0.01
    )


def test_reader_set_frame(video_clip):
    with pytest.raises(ValueError):
        video_clip.set_to_frame(-1)
    video_clip.set_to_frame(2)
    assert int(video_clip.video.get(POS_FRAMES)) == 2
    video_clip.set_to_frame(len(video_clip) + 10)
    assert int(video_clip.video.get(POS_FRAMES)) == len(video_clip) - 1
    video_clip.reset()
    assert int(video_clip.video.get(POS_FRAMES)) == 0


@pytest.mark.parametrize("shrink, crop", [(1, False), (1, True), (2, False), (2, True)])
def test_reader_read_frame(video_clip, shrink, crop):
    if crop:
        video_clip.set_bbox(0, 0.5, 0, 0.5, relative=True)
    frame = video_clip.read_frame(shrink, crop)
    height, width, _ = frame.shape
    assert height == video_clip.height // shrink
    assert width == video_clip.width // shrink


def test_writer_bbox(video_clip):
    bbox = 0, 100, 0, 100
    video_clip.set_bbox(*bbox)
    assert video_clip.get_bbox() == bbox
    with pytest.raises(ValueError):
        video_clip.set_bbox(200, 100, 0, 100, relative=False)
    video_clip.set_bbox(0, 1, 0, 1.01, relative=True)
    assert video_clip.get_bbox(relative=True) == (0, 1, 0, 1)


@pytest.mark.parametrize(
    "start, end", [(0, 10), ("0:0", "0:10"), ("00:00:00", "00:00:10")]
)
def test_writer_shorten_invalid_timestamps(video_clip, start, end):
    with pytest.raises(ValueError):
        video_clip.shorten(start, end)


def test_writer_shorten(tmp_path, video_clip):
    file = video_clip.shorten("00:00:00", "00:00:02", dest_folder=str(tmp_path))
    vid = VideoWriter(file)
    assert pytest.approx(vid.calc_duration(), abs=0.1) == 2


def test_writer_split(tmp_path, video_clip):
    with pytest.raises(ValueError):
        video_clip.split(1)
    n_splits = 3
    clips = video_clip.split(n_splits, dest_folder=str(tmp_path))
    assert len(clips) == n_splits
    vid = VideoWriter(clips[0])
    assert pytest.approx(len(vid), abs=1) == len(video_clip) // n_splits


def test_writer_crop(tmp_path, video_clip):
    x1, x2, y1, y2 = 0, 50, 0, 100
    video_clip.set_bbox(x1, x2, y1, y2)
    file = video_clip.crop(dest_folder=str(tmp_path))
    vid = VideoWriter(file)
    assert vid.dimensions == (x2 - x1, y2 - y1)


@pytest.mark.parametrize("target_height", [200, 177])
def test_writer_rescale(tmp_path, video_clip, target_height):
    file = video_clip.rescale(width=-1, height=target_height, dest_folder=str(tmp_path))
    vid = VideoWriter(file)
    assert vid.height == target_height
    # Verify the aspect ratio is preserved
    ar = video_clip.height / target_height
    assert vid.width == pytest.approx(video_clip.width // ar, abs=1)
