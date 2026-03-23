#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import pathlib
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

fmpose3d = pytest.importorskip("fmpose3d", reason="fmpose3d not installed")
pytestmark = pytest.mark.fmpose3d

# DLC fmpose_3d modules import fmpose3d; load only after importorskip above.
from deeplabcut.pose_estimation_pytorch.modelzoo.fmpose_3d import inference as fmp_inf  # noqa: E402
from deeplabcut.pose_estimation_pytorch.modelzoo.fmpose_3d.fmpose3d import (  # noqa: E402
    get_fmpose3d_inference_api,
)


def _has_network(host="huggingface.co", port=443, timeout=3) -> bool:
    """Return True if we can reach *host* (used to download model weights)."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _has_network(),
    reason="No network connection (needed to download model weights)",
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EXAMPLE_IMAGE = (
    _REPO_ROOT / "examples" / "Reaching-Mackenzie-2018-08-30" / "labeled-data" / "reachingvideo1" / "img005.png"
)


# ---------------------------------------------------------------------------
# Lightweight: verifies the API object is constructed correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_type", ["fmpose3d_humans", "fmpose3d_animals"])
@pytest.mark.unittest
def test_api_init(model_type):
    api = get_fmpose3d_inference_api(model_type, device="cpu")
    assert api is not None
    assert hasattr(api, "prepare_2d")
    assert hasattr(api, "pose_3d")
    assert hasattr(api, "predict")


# ---------------------------------------------------------------------------
# Integration: downloads weights and runs inference (needs network)
# ---------------------------------------------------------------------------
@requires_network
@pytest.mark.functional
def test_prepare_2d_and_pose_3d():
    """2D detection followed by 3D lifting on a real image."""
    api = get_fmpose3d_inference_api("fmpose3d_animals", device="cpu")

    result_2d = api.prepare_2d(source=str(_EXAMPLE_IMAGE))
    assert isinstance(result_2d.keypoints, np.ndarray)
    assert result_2d.keypoints.shape[-1] == 2

    keypoints_3d = api.pose_3d(
        keypoints_2d=result_2d.keypoints,
        image_size=result_2d.image_size,
    )
    assert isinstance(keypoints_3d.poses_3d, np.ndarray)
    assert keypoints_3d.poses_3d.shape[-1] == 3


@requires_network
@pytest.mark.functional
def test_predict_end_to_end():
    """Full pipeline (2D -> 3D) in a single call."""
    api = get_fmpose3d_inference_api("fmpose3d_animals", device="cpu")
    predictions_3d = api.predict(source=str(_EXAMPLE_IMAGE))

    assert isinstance(predictions_3d.poses_3d, np.ndarray)
    assert predictions_3d.poses_3d.shape[-1] == 3


@pytest.mark.unittest
def test_pose2d_to_dlc_predictions_shapes():
    pose_2d = SimpleNamespace(
        keypoints=np.random.rand(2, 3, 4, 2).astype(np.float32),
        scores=np.random.rand(2, 3, 4).astype(np.float32),
    )
    preds = fmp_inf._pose2d_to_dlc_predictions(
        pose_2d=pose_2d,
        max_individuals=1,
        num_bodyparts=4,
    )

    assert len(preds) == 3
    assert preds[0]["bodyparts"].shape == (1, 4, 3)
    np.testing.assert_allclose(preds[0]["bodyparts"][0, :, :2], pose_2d.keypoints[0, 0])


@pytest.mark.unittest
def test_poses3d_to_dataframe_layout():
    scorer = "DLC_test"
    bodyparts = ["bp1", "bp2", "bp3"]
    columns_2d = pd.MultiIndex.from_product(
        [[scorer], ["individual1"], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df_2d = pd.DataFrame(np.zeros((2, len(columns_2d))), columns=columns_2d)

    poses_3d = [
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]),
    ]
    df_3d = fmp_inf._poses3d_to_dataframe(poses_3d, df_2d, f"{scorer}_3d")

    assert df_3d.columns.names == ["scorer", "bodyparts", "coords"]
    assert set(df_3d.columns.get_level_values("coords")) == {"x", "y", "z"}
    assert df_3d.loc[0, (f"{scorer}_3d", "bp1", "x")] == 1.0
    assert df_3d.loc[1, (f"{scorer}_3d", "bp3", "z")] == 18.0


@pytest.mark.functional
def test_video_inference_fmpose3d_include_3d_return(tmp_path, monkeypatch):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    class FakeVideoIterator:
        def __init__(self, _path, cropping=None):
            self.dimensions = (8, 8)
            self.fps = 30
            self._frames = frames

        def __iter__(self):
            return iter(self._frames)

    class FakeAPI:
        def prepare_2d(self, source):
            n_frames = source.shape[0]
            return SimpleNamespace(
                keypoints=np.zeros((1, n_frames, 26, 2), dtype=np.float32),
                scores=np.ones((1, n_frames, 26), dtype=np.float32),
                image_size=(8, 8),
            )

        def pose_3d(self, keypoints_2d, image_size):
            n_frames = keypoints_2d.shape[1]
            return SimpleNamespace(
                poses_3d=np.zeros((n_frames, 26, 3), dtype=np.float32),
            )

    def _fake_create_df_from_prediction(predictions, dlc_scorer, multi_animal, model_cfg, output_path, output_prefix):
        bodyparts = model_cfg["metadata"]["bodyparts"]
        individuals = model_cfg["metadata"]["individuals"]
        columns = pd.MultiIndex.from_product(
            [[dlc_scorer], individuals, bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        return pd.DataFrame(np.zeros((len(predictions), len(columns))), columns=columns)

    monkeypatch.setattr(fmp_inf, "VideoIterator", FakeVideoIterator)
    monkeypatch.setattr(
        fmp_inf,
        "get_fmpose3d_inference_api",
        lambda model_type, device: FakeAPI(),
    )
    monkeypatch.setattr(fmp_inf, "create_df_from_prediction", _fake_create_df_from_prediction)
    monkeypatch.setattr(
        fmp_inf,
        "get_superanimal_colormaps",
        lambda: {
            "superanimal_quadruped": "viridis",
            "superanimal_humanbody": "viridis",
        },
    )

    result = fmp_inf._video_inference_fmpose3d(
        video_paths=[str(tmp_path / "dummy.mp4")],
        model_name="fmpose3d_animals",
        dest_folder=tmp_path,
        create_labeled_video=False,
        include_3d_in_return=True,
    )

    payload = result[str(tmp_path / "dummy.mp4")]
    assert "df_2d" in payload
    assert "df_3d" in payload
    assert isinstance(payload["df_3d"], pd.DataFrame)
    assert (tmp_path / "dummy_DLC_fmpose3d_animals_3d.h5").exists()
