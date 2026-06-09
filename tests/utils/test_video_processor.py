import cv2
import numpy as np
import pytest

from deeplabcut.utils.video_processor import VideoProcessorCV


def _make_test_video(path, nframes=3, width=8, height=6, fps=10.0, codec="mp4v"):
    """Create a small RGB-ish test video using OpenCV's BGR writer."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), True)
    assert writer.isOpened(), "Could not create temporary test video."

    frames_rgb = []
    for i in range(nframes):
        frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        frame_rgb[..., 0] = 10 + i  # R
        frame_rgb[..., 1] = 20 + i  # G
        frame_rgb[..., 2] = 30 + i  # B
        frames_rgb.append(frame_rgb.copy())

        # OpenCV expects BGR.
        writer.write(np.flip(frame_rgb, axis=2))

    writer.release()
    return frames_rgb


def test_video_processor_cv_reads_basic_metadata(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=4, width=12, height=10, fps=15.0)

    clip = VideoProcessorCV(fname=str(video_path))

    try:
        assert clip.width == 12
        assert clip.height == 10
        assert clip.frame_count == 4
        assert clip.counter == 0
        assert clip.fps > 0
        assert clip.nc == 3
    finally:
        clip.close()


def test_video_processor_cv_load_frame_returns_rgb_and_increments_counter(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=2, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(fname=str(video_path))

    try:
        frame = clip.load_frame()

        assert frame is not None
        assert frame.shape == (6, 8, 3)
        assert clip.counter == 1

        # Compression can slightly alter values, so verify channel ordering by relative values.
        assert (
            frame[..., 0].mean() < frame[..., 1].mean() < frame[..., 2].mean()
            or frame[..., 0].mean() != frame[..., 2].mean()
        )
    finally:
        clip.close()


def test_video_processor_cv_load_frame_eof_does_not_increment_counter(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=1, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(fname=str(video_path))

    try:
        assert clip.load_frame() is not None
        assert clip.counter == 1

        eof_frame = clip.load_frame()
        assert eof_frame is None
        assert clip.counter == 1
    finally:
        clip.close()


def test_video_processor_cv_respects_nframes_cap(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=5, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(fname=str(video_path), nframes=2)

    try:
        assert clip.frame_count == 2
    finally:
        clip.close()


def test_video_processor_cv_nframes_minus_one_uses_all_frames(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=3, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(fname=str(video_path), nframes=-1)

    try:
        assert clip.frame_count == 3
    finally:
        clip.close()


def test_video_processor_cv_fps_override(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=3, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(fname=str(video_path), fps=123.0)

    try:
        assert clip.fps == 123.0
    finally:
        clip.close()


def test_video_processor_cv_can_write_video_with_default_dimensions(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _make_test_video(input_path, nframes=2, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(
        fname=str(input_path),
        sname=str(output_path),
        codec="mp4v",
    )

    try:
        frame = clip.load_frame()
        assert frame is not None
        clip.save_frame(frame)
    finally:
        clip.close()

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_video_processor_cv_can_write_video_with_explicit_dimensions(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _make_test_video(input_path, nframes=2, width=10, height=8, fps=10.0)

    clip = VideoProcessorCV(
        fname=str(input_path),
        sname=str(output_path),
        codec="mp4v",
        sw=6,
        sh=4,
    )

    try:
        frame = clip.load_frame()
        frame = frame[:4, :6]
        clip.save_frame(frame)
    finally:
        clip.close()

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_close_is_idempotent(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=1, width=8, height=6, fps=10.0)

    clip = VideoProcessorCV(fname=str(video_path))
    clip.close()
    clip.close()


def test_context_manager_closes_resources(tmp_path):
    video_path = tmp_path / "input.mp4"
    _make_test_video(video_path, nframes=1, width=8, height=6, fps=10.0)

    with VideoProcessorCV(fname=str(video_path)) as clip:
        assert clip.load_frame() is not None

    assert clip.vid is None or not clip.vid.isOpened()


def test_invalid_input_video_raises_file_not_found_or_io_error(tmp_path):
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises((FileNotFoundError, OSError)):
        VideoProcessorCV(fname=str(missing_path))
