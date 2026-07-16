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
"""Tests that deeplabcut.video_inference_superanimal routes correctly.

Imports the public top-level API (not deeplabcut.api.*) so lazy exports and
decorator stacking are exercised together.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import deeplabcut
from deeplabcut.core.deprecation import DLCDeprecationWarning

_PT = "deeplabcut.modelzoo.video_inference.video_inference_superanimal"
_GET_TF = "deeplabcut.api._tf_routing._get_tensorflow_impl"


def test_video_inference_superanimal_routes_to_pytorch_for_non_dlcrnet():
    mock_pt = MagicMock(return_value={"video.mp4": "df"})

    with patch(_PT, mock_pt):
        result = deeplabcut.video_inference_superanimal(
            videos=["/videos/v.mp4"],
            superanimal_name="superanimal_topviewmouse",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            batch_size=2,
            max_individuals=3,
        )

    assert result == {"video.mp4": "df"}
    mock_pt.assert_called_once()
    kwargs = mock_pt.call_args.kwargs
    assert kwargs["videos"] == ["/videos/v.mp4"]
    assert kwargs["superanimal_name"] == "superanimal_topviewmouse"
    assert kwargs["model_name"] == "hrnet_w32"
    assert kwargs["detector_name"] == "fasterrcnn_resnet50_fpn_v2"
    assert kwargs["batch_size"] == 2
    assert kwargs["max_individuals"] == 3


def test_video_inference_superanimal_routes_to_tensorflow_for_dlcrnet():
    mock_tf = MagicMock(return_value="tf_result")

    with (
        patch(_GET_TF, return_value=mock_tf) as get_impl,
        pytest.warns(DLCDeprecationWarning),
    ):
        result = deeplabcut.video_inference_superanimal(
            videos=["/videos/v.mp4"],
            superanimal_name="superanimal_quadruped",
            model_name="dlcrnet",
            scale_list=[200, 300],
        )

    assert result == "tf_result"
    get_impl.assert_called_once_with(
        "video_inference_superanimal_tf",
        module="deeplabcut.tensorflow_compat.superanimal_inference",
    )
    mock_tf.assert_called_once()
    assert mock_tf.call_args.kwargs["model_name"] == "dlcrnet"
    assert mock_tf.call_args.kwargs["superanimal_name"] == "superanimal_quadruped"
