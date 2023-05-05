""" Tests for frame selection tools """
from unittest.mock import Mock

import deeplabcut.utils.frameselectiontools as fst


def test_uniform_frames():
    # TODO: What happens if there are less frames in the video that the number we asked to select?
    #   -> return all frames from the video?

    # Index bounds: (0, 1) => valid_frame_index_1 == frame_index_1
    frame_index_1 = list(range(100, 500, 5))
    # Index bounds: 0.25 * 1600 = 400, 0.3 * 1600 = 480
    frame_index_2 = list(range(100, 500, 5))
    valid_frame_index_2 = [idx for idx in frame_index_2 if 400 <= idx <= 480]
    for fps, duration, n_to_pick, start, end, index, valid_frame_indices in [
        (32, 10, 10, 0, 1, None, list(range(0, 320))),  # 320 frames at 32 fps -> 10 seconds
        (16, 100, 50, 0, 1, frame_index_1, frame_index_1),  # 1600 frames at 16 fps -> 100 seconds
        (16, 100, 5, 0.25, 0.3, frame_index_2, valid_frame_index_2),  # 1600 frames at 16 fps -> 100 seconds
    ]:
        clip = Mock()
        clip.fps = fps
        clip.duration = duration
        frames = fst.UniformFrames(clip, n_to_pick, start, end, index)
        print(f"FPS: {fps}")
        print(f"Duration: {duration}")
        print(f"Selected Frames: {frames}")
        print(f"Valid Indices: {valid_frame_indices}")

        # Check that we get the number of frames we asked for
        assert len(frames) == n_to_pick, f"Not the number of frames desired: {n_to_pick} != {len(frames)}"
        # Check that all indices are valid
        for index in frames:
            assert index in valid_frame_indices, f"Invalid index: {index} not in {valid_frame_indices}"
        # Check that all frames are unique
        assert len(set(frames)) == len(frames), "Duplicate indices found"



def test_uniform_frames_cv2():
    # TODO: What happens if there are less frames in the video that the number we asked to select?
    #   -> return all frames from the video?

    # Index bounds: (0, 1) => valid_frame_index_1 == frame_index_1
    frame_index_1 = list(range(100, 500, 5))
    # Index bounds: 0.25 * 1600 = 400, 0.3 * 1600 = 480
    frame_index_2 = list(range(100, 500, 5))
    valid_frame_index_2 = [idx for idx in frame_index_2 if 400 <= idx <= 480]
    for fps, nframes, n_to_pick, start, end, index, valid_frame_indices in [
        (32, 320, 10, 0, 1, None, list(range(0, 320))),  # 320 frames at 32 fps -> 10 seconds
        (16, 1600, 50, 0, 1, frame_index_1, frame_index_1),  # 1600 frames at 16 fps -> 100 seconds
        (16, 1600, 5, 0.25, 0.3, frame_index_2, valid_frame_index_2),  # 1600 frames at 16 fps -> 100 seconds
    ]:
        cap = Mock()
        cap.fps = fps
        cap.__len__ = Mock(return_value=nframes)
        frames = fst.UniformFramescv2(cap, n_to_pick, start, end, index)
        print(f"FPS: {fps}")
        print(f"Nframes: {nframes}")
        print(f"Selected Frames: {frames}")
        print(f"Valid Indices: {valid_frame_indices}")

        # Check that we get the number of frames we asked for
        assert len(frames) == n_to_pick, f"Not the number of frames desired: {n_to_pick} != {len(frames)}"
        # Check that all indices are valid
        for index in frames:
            assert index in valid_frame_indices, f"Invalid index: {index} not in {valid_frame_indices}"
        # Check that all frames are unique
        assert len(set(frames)) == len(frames), "Duplicate indices found"
