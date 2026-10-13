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
"""Regression tests for empty video analysis output (issue #3485)."""

from __future__ import annotations

import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.apis.videos import (
    _generate_output_data,
    create_df_from_prediction,
)

POSE_CFG = {
    "all_joints": [[0], [1], [2]],
    "all_joints_names": ["snout", "leftear", "rightear"],
    "nmsradius": 5,
    "minconfidence": 0.1,
    "sigma": 1,
}

MODEL_CFG = {
    "metadata": {
        "bodyparts": ["snout", "leftear", "rightear"],
        "unique_bodyparts": [],
        "individuals": ["animal0"],
    }
}


def _make_prediction(num_assemblies: int = 1, num_bpts: int = 3) -> dict:
    rng = np.random.default_rng(42)
    return {"bodyparts": rng.random((num_assemblies, num_bpts, 3)).astype(np.float32)}


def test_generate_output_data_empty_does_not_crash():
    """Empty predictions (no detections on any frame) must yield valid output.

    Regression for #3485: ``int(np.ceil(np.log10(0)))`` raised
    ``OverflowError: cannot convert float infinity to integer`` and aborted
    the whole ``analyze_videos`` run for clips where the animal is absent.
    """
    output = _generate_output_data(POSE_CFG, [])

    assert output["metadata"]["nframes"] == 0
    assert output["metadata"]["key_str_width"] == 1
    assert [k for k in output if k != "metadata"] == []


def test_generate_output_data_single_frame_width_is_one():
    """A single frame still needs width 1 (log10(1) is 0, zfill(0) is wrong)."""
    output = _generate_output_data(POSE_CFG, [_make_prediction()])

    assert output["metadata"]["nframes"] == 1
    assert output["metadata"]["key_str_width"] == 1
    assert "frame0" in output


@pytest.mark.parametrize(
    "num_frames, width",
    [(9, 1), (10, 1), (100, 2), (1000, 3)],
)
def test_generate_output_data_key_width(num_frames, width):
    predictions = [_make_prediction() for _ in range(num_frames)]
    output = _generate_output_data(POSE_CFG, predictions)

    assert output["metadata"]["key_str_width"] == width
    expected_key = "frame" + str(num_frames - 1).zfill(width)
    assert expected_key in output


def test_create_df_from_prediction_empty_writes_valid_h5(tmp_path):
    """Empty predictions must produce an empty (but valid) h5, not crash."""
    df = create_df_from_prediction(
        predictions=[],
        dlc_scorer="testscorer",
        multi_animal=False,
        model_cfg=MODEL_CFG,
        output_path=tmp_path,
        output_prefix="empty_test",
    )

    assert len(df) == 0
    assert (tmp_path / "empty_test.h5").exists()
