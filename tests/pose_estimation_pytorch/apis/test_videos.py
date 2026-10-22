from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.apis.videos as videos
from deeplabcut.pose_estimation_pytorch.task import Task

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


@pytest.mark.parametrize("with_detector", [False, True])
def test_video_inference_zero_predictions_warns_about_unreadable_video(patch_video_iterator, caplog, with_detector):
    """Every readable frame yields a prediction (padded when no animal is detected),
    so an empty result means the video could not be read - not that no animals were
    found."""
    patch_video_iterator(3)
    detector_runner = _pose_runner([]) if with_detector else None

    with caplog.at_level("WARNING"):
        predictions = videos.video_inference("video.mp4", _pose_runner([]), detector_runner=detector_runner)

    assert predictions == []
    assert "No predictions were produced for video.mp4" in caplog.text
    assert "the video could not be read" in caplog.text
    assert "no animals were detected" not in caplog.text
    # the re-encoding tip is the actionable advice in this case
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


def test_analyze_videos_skips_unreadable_video_without_writing_outputs(
    tmp_path, monkeypatch, patch_video_iterator, caplog
):
    """An unreadable video must not abort the remaining videos, nor leave a
    ``_full.pickle`` that makes reruns with ``overwrite=False`` skip it."""
    patch_video_iterator(3)

    loader = Mock()
    loader.project_cfg = {"TrainingFraction": [0.95], "multianimalproject": False}
    loader.model_cfg = {
        "metadata": {
            "bodyparts": ["snout", "leftear", "rightear"],
            "unique_bodyparts": [],
            "individuals": ["animal_0"],
        }
    }
    loader.model_folder = tmp_path / "train"
    loader.pose_task = Task.BOTTOM_UP
    loader.scorer.return_value = "DLC_test"
    monkeypatch.setattr(videos, "DLCLoader", Mock(return_value=loader))
    monkeypatch.setattr(videos.auxiliaryfunctions, "read_plainconfig", Mock(return_value=POSE_CFG))
    monkeypatch.setattr(videos.utils, "parse_snapshot_index_for_analysis", Mock(return_value=(0, None)))
    monkeypatch.setattr(videos.utils, "get_model_snapshots", Mock(return_value=[Mock(path="snapshot.pt")]))
    monkeypatch.setattr(videos.utils, "get_pose_inference_runner", Mock())
    monkeypatch.setattr(videos, "_generate_metadata", Mock(return_value={}))

    unreadable, readable = tmp_path / "unreadable.mp4", tmp_path / "readable.mp4"
    monkeypatch.setattr(videos, "collect_video_paths", Mock(return_value=[unreadable, readable]))
    predictions = {unreadable: [], readable: [dict(bodyparts=np.zeros((1, 3, 3)))]}
    monkeypatch.setattr(videos, "video_inference", lambda video, **kwargs: predictions[Path(video.video_path)])
    create_df = Mock()
    monkeypatch.setattr(videos, "create_df_from_prediction", create_df)

    with caplog.at_level("WARNING"):
        videos.analyze_videos(tmp_path / "config.yaml", [unreadable, readable])

    assert not list(tmp_path.glob("unreadable*"))
    assert (tmp_path / "readableDLC_test_full.pickle").exists()
    assert f"Skipping {unreadable}" in caplog.text
    create_df.assert_called_once()
    assert create_df.call_args.kwargs["output_prefix"] == "readableDLC_test"
