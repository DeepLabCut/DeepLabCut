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
"""Tests for ShelfWriter / ShelfReader"""
from __future__ import annotations

import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.runners.shelving import (
    ShelfReader,
    ShelfWriter,
)

POSE_CFG = {
    "all_joints": [[0], [1], [2]],
    "all_joints_names": ["snout", "leftear", "rightear"],
    "nmsradius": 5,
    "minconfidence": 0.1,
    "sigma": 1,
}


def _make_bodyparts(num_assemblies: int = 2, num_bpts: int = 3) -> np.ndarray:
    """(num_assemblies, num_bpts, 3)  — x, y, score"""
    rng = np.random.default_rng(0)
    return rng.random((num_assemblies, num_bpts, 3)).astype(np.float32)


# -- lifecycle ----------------------------------------------------------------


def test_write_before_open_raises(tmp_path):
    writer = ShelfWriter(POSE_CFG, tmp_path / "shelf")
    with pytest.raises(ValueError, match="open"):
        writer.add_prediction(_make_bodyparts())


def test_open_close_roundtrip(tmp_path):
    path = tmp_path / "shelf"
    writer = ShelfWriter(POSE_CFG, path)
    writer.open()
    writer.add_prediction(_make_bodyparts())
    writer.close()

    reader = ShelfReader(path)
    reader.open()
    assert "metadata" in reader.keys()
    assert "frame00000" in reader.keys()
    reader.close()


# -- key formatting -----------------------------------------------------------


@pytest.mark.parametrize("num_frames,width", [(9, 1), (100, 2), (1000, 3)])
def test_key_str_width(tmp_path, num_frames, width):
    writer = ShelfWriter(POSE_CFG, tmp_path / "shelf", num_frames=num_frames)
    writer.open()
    writer.add_prediction(_make_bodyparts())
    writer.close()

    reader = ShelfReader(tmp_path / "shelf")
    reader.open()
    expected_key = "frame" + "0".zfill(width)
    assert expected_key in reader.keys()
    reader.close()


# -- data shape ---------------------------------------------------------------


def test_add_prediction_stores_correct_shapes(tmp_path):
    num_assemblies, num_bpts = 2, 3
    bp = _make_bodyparts(num_assemblies, num_bpts)

    writer = ShelfWriter(POSE_CFG, tmp_path / "shelf", num_frames=10)
    writer.open()
    writer.add_prediction(bp)
    writer.close()

    reader = ShelfReader(tmp_path / "shelf")
    reader.open()
    data = reader["frame0"]

    coords = data["coordinates"][0]
    assert len(coords) == num_bpts
    assert coords[0].shape == (num_assemblies, 2)

    scores = data["confidence"]
    assert len(scores) == num_bpts
    assert scores[0].shape == (num_assemblies, 1)
    reader.close()


# -- metadata on close --------------------------------------------------------


def test_metadata_nframes_updated_on_close(tmp_path):
    writer = ShelfWriter(POSE_CFG, tmp_path / "shelf", num_frames=100)
    writer.open()
    for _ in range(3):
        writer.add_prediction(_make_bodyparts())
    writer.close()

    reader = ShelfReader(tmp_path / "shelf")
    reader.open()
    assert reader["metadata"]["nframes"] == 3
    reader.close()


# -- unique bodyparts ---------------------------------------------------------


def test_unique_bodyparts_appended(tmp_path):
    num_assemblies, num_bpts, num_unique = 2, 3, 1
    bp = _make_bodyparts(num_assemblies, num_bpts)
    ubp = np.random.default_rng(1).random((num_assemblies, num_unique, 3)).astype(
        np.float32
    )

    writer = ShelfWriter(POSE_CFG, tmp_path / "shelf", num_frames=5)
    writer.open()
    writer.add_prediction(bp, unique_bodyparts=ubp)
    writer.close()

    reader = ShelfReader(tmp_path / "shelf")
    reader.open()
    data = reader["frame0"]
    assert len(data["coordinates"][0]) == num_bpts + num_unique
    assert len(data["confidence"]) == num_bpts + num_unique
    reader.close()


# -- identity scores ----------------------------------------------------------


def test_identity_scores_stored(tmp_path):
    num_assemblies, num_bpts, num_individuals = 2, 3, 2
    bp = _make_bodyparts(num_assemblies, num_bpts)
    ids = np.random.default_rng(2).random(
        (num_assemblies, num_bpts, num_individuals)
    ).astype(np.float32)

    writer = ShelfWriter(POSE_CFG, tmp_path / "shelf", num_frames=5)
    writer.open()
    writer.add_prediction(bp, identity_scores=ids)
    writer.close()

    reader = ShelfReader(tmp_path / "shelf")
    reader.open()
    data = reader["frame0"]
    assert "identity" in data
    assert len(data["identity"]) == num_bpts
    assert data["identity"][0].shape == (num_assemblies, num_individuals)
    reader.close()
