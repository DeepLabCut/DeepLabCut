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
"""Tests the heatmap target generators (plateau and gaussian)"""
import numpy as np
import torch
import pytest

from deeplabcut.pose_estimation_pytorch.models.target_generators.heatmap_targets import (
    HeatmapGenerator,
    HeatmapPlateauGenerator,
)


@pytest.mark.parametrize(
    "data",
    [
        {
            "dist_thresh": 1,
            "num_heatmaps": 1,
            "in_shape": (3, 3),
            "out_shape": (3, 3),
            "centers": [(1, 1)],
            "expected_output": [
                [0., 1., 0.],
                [1., 1., 1.],
                [0., 1., 0.],
            ],
        },
        {
            "dist_thresh": 2,
            "num_heatmaps": 1,
            "in_shape": (5, 5),
            "out_shape": (5, 5),
            "centers": [[1, 1], [2, 2]],
            "expected_output": [
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.],
                [0., 1., 1., 1., 0.],
                [0., 0., 1., 0., 0.],
            ],
        },
        {
            "dist_thresh": 2,
            "num_heatmaps": 1,
            "in_shape": (4, 4),
            "out_shape": (4, 4),
            "centers": [[1, 1]],
            "expected_output": [
                [1., 1., 1., 0.],
                [1., 1., 1., 1.],
                [1., 1., 1., 0.],
                [0., 1., 0., 0.],
            ],
        },
    ],
)
def test_plateau_heatmap_generation_single_keypoint(data):
    dist_thresh = data["dist_thresh"]
    generator = HeatmapPlateauGenerator(
        num_heatmaps=data["num_heatmaps"],
        pos_dist_thresh=dist_thresh,
        heatmap_mode=HeatmapGenerator.Mode.KEYPOINT,
        generate_locref=False,
    )
    stride = data["in_shape"][0] / data["out_shape"][0]
    outputs = torch.zeros((1, data["num_heatmaps"], *data["out_shape"]))
    ann_shape = (1, len(data["centers"]), data["num_heatmaps"], 2)
    annotations = {
        "keypoints": torch.tensor(data["centers"]).reshape(ann_shape)  # x, y
    }
    targets = generator(stride, {"heatmap": outputs}, annotations)

    print("Targets")
    print(targets["heatmap"]["target"])
    print()
    np.testing.assert_almost_equal(
        targets["heatmap"]["target"].cpu().numpy().reshape(data["out_shape"]),
        np.array(data["expected_output"]),
        decimal=3,
    )
