from unittest.mock import Mock

import pytest

import deeplabcut.pose_estimation_pytorch.apis.videos as videos

POSE_CFG = {
    "all_joints": [[0], [1], [2]],
    "all_joints_names": ["snout", "leftear", "rightear"],
    "nmsradius": 5,
    "minconfidence": 0.1,
    "sigma": 1,
}


def test_generate_output_data_handles_empty_predictions():
    output = videos._generate_output_data(POSE_CFG, [])

    assert output["metadata"].pop("PAFinds").tolist() == []
    assert output == {
        "metadata": {
            "nms radius": 5,
            "minimal confidence": 0.1,
            "sigma": 1,
            "PAFgraph": None,
            "all_joints": [[0], [1], [2]],
            "all_joints_names": ["snout", "leftear", "rightear"],
            "nframes": 0,
            "key_str_width": 1,
        }
    }


def test_create_df_from_prediction_rejects_empty_predictions(tmp_path):
    with pytest.raises(ValueError, match="empty predictions list"):
        videos.create_df_from_prediction(
            predictions=[],
            dlc_scorer="DLC_test",
            multi_animal=False,
            model_cfg={
                "metadata": {
                    "bodyparts": ["snout", "leftear", "rightear"],
                    "unique_bodyparts": [],
                    "individuals": ["animal_0"],
                }
            },
            output_path=tmp_path,
            output_prefix="video",
            save_as_csv=False,
        )

    assert not (tmp_path / "video.h5").exists()


@pytest.fixture
def patch_video_iterator(monkeypatch):
    """Patches ``VideoIterator`` with a stub reporting ``n_frames`` empty frames."""

    def _patch(n_frames: int):
        class FakeVideoIterator:
            def __init__(self, video_path, cropping=None):
                self.video_path = video_path
                self.fps = 25
                self.dimensions = (640, 480)

            def get_n_frames(self, robust=False):
                return n_frames

            def set_context(self, context):
                pass

            def __iter__(self):
                return iter(())

        monkeypatch.setattr(videos, "VideoIterator", FakeVideoIterator)
        return FakeVideoIterator

    return _patch


def _pose_runner(predictions: list) -> Mock:
    runner = Mock()
    runner.batch_size = 2
    runner.inference.return_value = predictions
    return runner


def test_video_inference_zero_predictions_warns_about_unreadable_video(patch_video_iterator, caplog):
    """Without a detector, every readable frame yields a prediction, so an empty
    result means the video could not be read - not that no animals were found."""
    patch_video_iterator(3)

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([]))

    assert predictions == []
    assert "No predictions were produced for video.mp4" in caplog.text
    assert "the video could not be read" in caplog.text
    assert "no animals were detected" not in caplog.text
    # the re-encoding tip is the actionable advice in this case
    assert "re-encoding your video" in caplog.text


def test_video_inference_zero_predictions_with_detector_warns_about_detections(patch_video_iterator, caplog):
    patch_video_iterator(3)

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([]), detector_runner=_pose_runner([]))

    assert predictions == []
    assert "no animals were detected in any frame" in caplog.text
    assert "the video could not be read" in caplog.text
    assert "re-encoding your video" in caplog.text


def test_video_inference_zero_predictions_warns_when_frame_count_is_zero(patch_video_iterator, caplog):
    """A video whose metadata reports 0 frames must still warn."""
    patch_video_iterator(0)

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([]))

    assert predictions == []
    assert "No predictions were produced for video.mp4" in caplog.text


def test_video_inference_warns_when_some_frames_are_missing(patch_video_iterator, caplog):
    patch_video_iterator(3)

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([{}, {}]))

    assert len(predictions) == 2
    assert "there are 3 frames in the video, but only 2" in caplog.text
    assert "re-encoding your video" in caplog.text
    assert "No predictions were produced" not in caplog.text


def test_video_inference_does_not_warn_when_all_frames_are_predicted(patch_video_iterator, caplog):
    patch_video_iterator(3)

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([{}, {}, {}]))

    assert len(predictions) == 3
    assert "No predictions were produced" not in caplog.text
    assert "were able to be processed" not in caplog.text


def test_video_inference_does_not_warn_when_writing_to_a_shelf(patch_video_iterator, caplog):
    """The returned list is empty by design when a shelf writer is given."""
    patch_video_iterator(3)

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([]), shelf_writer=Mock())

    assert predictions == []
    assert "No predictions were produced" not in caplog.text
    assert "were able to be processed" not in caplog.text
