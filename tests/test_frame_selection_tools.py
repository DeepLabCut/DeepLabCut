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
"""Tests for frame selection tools."""

import math
from unittest.mock import Mock

import pytest

import deeplabcut.utils.frameselectiontools as fst


@pytest.mark.parametrize(
    "fps, duration, n_to_pick, start, end, index",
    [
        (32, 10, 10, 0, 1, None),
        (16, 100, 50, 0, 1, list(range(100, 500, 5))),
        (16, 100, 5, 0.25, 0.3, list(range(100, 500, 5))),
    ],
)
def test_uniform_frames(fps, duration, n_to_pick, start, end, index):
    start_idx = int(math.floor(start * duration * fps))
    end_idx = int(math.ceil(end * duration * fps))
    if index is None:
        valid_indices = list(range(start_idx, end_idx))
    else:
        valid_indices = [idx for idx in index if start_idx <= idx <= end_idx]

    clip = Mock()
    clip.fps = fps
    clip.duration = duration
    frames = fst.UniformFrames(clip, n_to_pick, start, end, index)
    print(f"FPS: {fps}")
    print(f"Duration: {duration}")
    print(f"Selected Frames: {frames}")
    print(f"Valid Indices: {valid_indices}")

    # Check that we get the number of frames we asked for
    assert len(frames) == n_to_pick, f"Wrong nb. of frames: {n_to_pick}!={len(frames)}"
    # Check that all indices are valid
    for index in frames:
        assert index in valid_indices, f"Invalid index: {index} not in {valid_indices}"
    # Check that all frames are unique
    assert len(set(frames)) == len(frames), "Duplicate indices found"


@pytest.mark.parametrize(
    "fps, nframes, n_to_pick, start, end, index",
    [
        (32, 320, 10, 0, 1, None),
        (16, 1600, 50, 0, 1, list(range(100, 500, 5))),
        (16, 1600, 5, 0.25, 0.3, list(range(100, 500, 5))),
    ],
)
def test_uniform_frames_cv2(fps, nframes, n_to_pick, start, end, index):
    start_idx = int(math.floor(start * nframes))
    end_idx = int(math.ceil(end * nframes))
    if index is None:
        valid_indices = list(range(start_idx, end_idx))
    else:
        valid_indices = [idx for idx in index if start_idx <= idx <= end_idx]

    cap = Mock()
    cap.fps = fps
    cap.__len__ = Mock(return_value=nframes)
    frames = fst.UniformFramescv2(cap, n_to_pick, start, end, index)
    print(f"FPS: {fps}")
    print(f"Nframes: {nframes}")
    print(f"Selected Frames: {frames}")
    print(f"Valid Indices: {valid_indices}")

    # Check that we get the number of frames we asked for
    assert len(frames) == n_to_pick, f"Wrong nb. of frames: {n_to_pick}!={len(frames)}"
    # Check that all indices are valid
    for index in frames:
        assert index in valid_indices, f"Invalid index: {index} not in {valid_indices}"
    # Check that all frames are unique
    assert len(set(frames)) == len(frames), "Duplicate indices found"


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("start, stop", [(0, 1), (0.5, 1), (0.25, 0.75)])
def test_kmeans_cv2_reads_frames_in_requested_range(fake_video_reader, start, stop, color):
    nframes = 100
    cap = fake_video_reader(nframes)
    frames = fst.KmeansbasedFrameselectioncv2(cap, 5, start, stop, resizewidth=20, color=color)

    start_idx = int(math.floor(start * nframes))
    stop_idx = int(math.ceil(stop * nframes))
    # The frames that are clustered must be the ones in the requested range
    assert cap.read_indices == list(range(start_idx, stop_idx))
    assert all(start_idx <= frame < stop_idx for frame in frames)


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(list(range(31, 60)), id="consecutive"),
        pytest.param(list(range(10, 90, 4)), id="non-consecutive"),
    ],
)
def test_kmeans_cv2_reads_preselected_frames(fake_video_reader, index, color):
    # extract_outlier_frames passes the candidate frames as Index
    cap = fake_video_reader(100)
    frames = fst.KmeansbasedFrameselectioncv2(cap, 5, 0, 1, Index=index, resizewidth=20, color=color)

    # The frames that are clustered must be the preselected ones
    assert cap.read_indices == index
    assert set(frames) <= set(index)
