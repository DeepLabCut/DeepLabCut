import os
os.environ['DLClight'] = 'True'
import pytest
from deeplabcut.utils import video


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
videos = [os.path.join(TEST_DATA_DIR, 'vid1.mov'),
          os.path.join(TEST_DATA_DIR, 'vid2.mov')]
POS_FRAMES = 1  # Equivalent to cv2.CAP_PROP_POS_FRAMES


@pytest.fixture()
def video_clip():
    return video.VideoPlayer(videos[0])


def test_player_invalid_file(tmpdir):
    fake_vid = tmpdir.join('fake.avi')
    with pytest.raises(IOError):
        video.VideoPlayer(str(fake_vid))


def test_player_set_frame(video_clip):
    video_clip.set_to_frame(2)
    assert int(video_clip.video.get(POS_FRAMES)) == 2
    video_clip.set_to_frame(video_clip.nframes + 10)
    assert int(video_clip.video.get(POS_FRAMES)) == video_clip.last_frame


def test_player_bbox(video_clip):
    bbox = 0, 100, 0, 100
    video_clip.set_bbox(*bbox, relative=False)
    assert video_clip.get_bbox(relative=False) == bbox
    with pytest.raises(ValueError):
        video_clip.set_bbox(200, 100, 0, 100, relative=False)
    video_clip.set_bbox(0, 1, 0, 1.01)
    assert video_clip.get_bbox() == (0, 1, 0, 1)


@pytest.mark.parametrize('shrink, crop',
                         [(1, False),
                          (1, True),
                          (2, False),
                          (2, True)])
def test_player_read_frame(video_clip, shrink, crop):
    video_clip.set_bbox(0, 0.5, 0, 0.5)
    frame = video_clip.read_frame(shrink, crop)
    height, width, _ = frame.shape
    if crop:
        x1, x2, y1, y2 = video_clip.get_bbox(False)
        assert height == (y2 - y1) // shrink
        assert width == (x2 - x1) // shrink
    else:
        assert height == video_clip.height // shrink
        assert width == video_clip.width // shrink