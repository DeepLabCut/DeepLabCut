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
"""Tests that deeplabcut.create_pretrained_project routes correctly.

Imports the public top-level API (not deeplabcut.api.*) so lazy exports and
decorator stacking are exercised together.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import deeplabcut
from deeplabcut.core.deprecation import DLCDeprecationWarning
from deeplabcut.core.engine import Engine

_PT = "deeplabcut.create_project.modelzoo.create_pretrained_project"
_GET_TF = "deeplabcut.api._tf_routing._get_tensorflow_impl"


def test_create_pretrained_project_routes_to_pytorch():
    mock_pt = MagicMock(return_value="/path/to/config.yaml")

    with patch(_PT, mock_pt):
        result = deeplabcut.create_pretrained_project(
            "proj",
            "exp",
            ["/videos/v.mp4"],
            model="superanimal_quadruped",
            working_directory="/tmp",
            copy_videos=True,
            analyzevideo=False,
            engine=Engine.PYTORCH,
            net_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
        )

    assert result == "/path/to/config.yaml"
    mock_pt.assert_called_once_with(
        project="proj",
        experimenter="exp",
        videos=["/videos/v.mp4"],
        model="superanimal_quadruped",
        working_directory="/tmp",
        copy_videos=True,
        video_extensions=None,
        analyzevideo=False,
        filtered=True,
        createlabeledvideo=True,
        trainFraction=None,
        multi_animal=False,
        individuals=None,
        net_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
    )


def test_create_pretrained_project_routes_to_tensorflow_compat():
    mock_tf = MagicMock(return_value="/path/to/tf_config.yaml")

    with (
        patch(_GET_TF, return_value=mock_tf) as get_impl,
        pytest.warns(DLCDeprecationWarning),
    ):
        result = deeplabcut.create_pretrained_project(
            "proj",
            "exp",
            ["/videos/v.mp4"],
            model="full_cat",
            engine=Engine.TF,
        )

    assert result == "/path/to/tf_config.yaml"
    get_impl.assert_called_once_with(
        "_tf_create_pretrained_project",
        module="deeplabcut.tensorflow_compat.create_project",
    )
    mock_tf.assert_called_once()
    # engine must not be forwarded to the TF implementation
    assert "engine" not in mock_tf.call_args.kwargs
    assert mock_tf.call_args.args[0] == "proj"
    assert mock_tf.call_args.args[1] == "exp"
    assert mock_tf.call_args.args[2] == ["/videos/v.mp4"]


def test_create_pretrained_project_default_engine_is_pytorch():
    mock_pt = MagicMock(return_value="cfg")

    with patch(_PT, mock_pt):
        deeplabcut.create_pretrained_project("proj", "exp", ["/videos/v.mp4"])

    mock_pt.assert_called_once()
