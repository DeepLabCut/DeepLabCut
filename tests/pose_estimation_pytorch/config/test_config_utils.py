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
"""Test util functions for config creation"""
import pytest

import deeplabcut.pose_estimation_pytorch.config.utils as utils


@pytest.mark.parametrize(
    "data",
    [
        dict(
            config={},
            num_bodyparts=None,
            num_individuals=None,
            backbone_output_channels=None,
            output_config={},
        ),
        dict(
            config={
                "a": "num_bodyparts",
                "b": ["num_bodyparts // 2", "num_bodyparts // 3"],
                "c": "num_bodyparts x 2",
                "d": "num_bodyparts + 2",
            },
            num_bodyparts=10,
            num_individuals=None,
            backbone_output_channels=None,
            output_config={
                "a": 10,
                "b": [5, 3],
                "c": 20,
                "d": 12,
            },
        ),
        dict(
            config={
                "a": [{"b": "num_individuals x 3"}],
                "b": [[{"b": "num_bodyparts x 3"}]],
            },
            num_bodyparts=10,
            num_individuals=1,
            backbone_output_channels=None,
            output_config={
                "a": [{"b": 3}],
                "b": [[{"b": 30}]],
            },
        )
    ],
)
def test_replace_default_values_no_extras(data: dict):
    output_config = utils.replace_default_values(
        config=data["config"],
        num_bodyparts=data["num_bodyparts"],
        num_individuals=data["num_individuals"],
        backbone_output_channels=data["backbone_output_channels"],
    )
    assert output_config == data["output_config"]
