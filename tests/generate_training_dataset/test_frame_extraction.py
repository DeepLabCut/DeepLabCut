import os
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from deeplabcut.generate_training_dataset.frame_extraction import _filter_config_videos, extract_frames
from deeplabcut.utils import auxfun_videos, auxiliaryfunctions, frameselectiontools


def test_extract_frames_accepts_path_videos_list(tmp_path, monkeypatch):
    video = tmp_path / "videos" / "video.mp4"
    video.parent.mkdir()
    video.touch()

    cfg = {
        "video_sets": {
            str(video): {"crop": "0, 100, 0, 100"},
        },
        "numframes2pick": 1,
        "start": 0.0,
        "stop": 1.0,
    }

    monkeypatch.setattr(
        auxiliaryfunctions,
        "read_config",
        lambda _: cfg,
    )

    processed = []

    class FakeVideoWriter:
        def __init__(self, path):
            processed.append(path)

        def __len__(self):
            return 10

        def set_to_frame(self, index):
            pass

        def read_frame(self, crop=True):
            return np.zeros((100, 100, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(
        auxfun_videos,
        "VideoWriter",
        FakeVideoWriter,
    )

    # Isolate path filtering from frame-selection behavior.
    monkeypatch.setattr(
        frameselectiontools,
        "UniformFramescv2",
        lambda *args, **kwargs: [0],
    )

    # Avoid writing an actual PNG.
    monkeypatch.setattr(io, "imsave", lambda *args, **kwargs: None)

    result = extract_frames(
        tmp_path / "config.yaml",
        mode="automatic",
        algo="uniform",
        videos_list=[video],
        userfeedback=False,
    )

    assert processed == [str(video)]
    assert result == [False]


class TestFilterConfigVideos:
    def test_filter_config_videos_matches_path_to_string(self):
        configured = [r"C:\project\videos\video.mp4"]
        selected = [Path(r"C:\project\videos\video.mp4")]

        result = _filter_config_videos(configured, selected)

        assert result == configured
        assert isinstance(result[0], str)

    def test_filter_config_videos_matches_string_to_path(self):
        configured = [Path(r"C:\project\videos\video.mp4")]
        selected = [r"C:\project\videos\video.mp4"]

        result = _filter_config_videos(configured, selected)

        assert result == configured
        assert isinstance(result[0], Path)

    def test_filter_config_videos_preserves_original_config_key(self):
        configured = [r"C:\project\videos\video.mp4"]
        selected = [Path(r"C:\project\videos\video.mp4")]

        result = _filter_config_videos(configured, selected)

        assert result[0] is configured[0]

    def test_filter_config_videos_returns_all_when_selection_is_none(self):
        configured = ["video-a.mp4", "video-b.mp4"]

        assert _filter_config_videos(configured, None) == configured

    def test_filter_config_videos_returns_empty_for_nonmatching_selection(self):
        configured = ["video-a.mp4"]
        selected = [Path("video-b.mp4")]

        assert _filter_config_videos(configured, selected) == []

    @pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
    def test_filter_config_videos_is_case_insensitive_on_windows(self):
        configured = [r"C:\Project\Videos\VIDEO.MP4"]
        selected = [Path(r"c:\project\videos\video.mp4")]

        assert _filter_config_videos(configured, selected) == configured
