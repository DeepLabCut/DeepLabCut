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

import numpy as np
import pytest

fmpose3d = pytest.importorskip("fmpose3d", reason="fmpose3d not installed")

from deeplabcut.modelzoo.fmpose_3d.fmpose3d import get_fmpose3d_inference_api


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
    _REPO_ROOT
    / "examples"
    / "Reaching-Mackenzie-2018-08-30"
    / "labeled-data"
    / "reachingvideo1"
    / "img005.png"
)


# ---------------------------------------------------------------------------
# Lightweight: verifies the API object is constructed correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_type", ["fmpose3d_humans", "fmpose3d_animals"]
)
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
    assert isinstance(keypoints_3d, np.ndarray)
    assert keypoints_3d.shape[-1] == 3


@requires_network
def test_predict_end_to_end():
    """Full pipeline (2D -> 3D) in a single call."""
    api = get_fmpose3d_inference_api("fmpose3d_animals", device="cpu")
    predictions_3d = api.predict(source=str(_EXAMPLE_IMAGE))

    assert isinstance(predictions_3d, np.ndarray)
    assert predictions_3d.shape[-1] == 3
