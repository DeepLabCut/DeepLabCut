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
""" Tests for frame selection tools """
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
