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
    HeatmapGaussianGenerator,
)

@pytest.mark.parametrize(
    "data",
    [
        {
            "dist_thresh": 3,
            "num_heatmaps": 1,
            "in_shape": (3, 3),
            "out_shape": (3, 3),
            "centers": [(1, 1)],
            "expected_output": [
                [0.7788, 0.8825, 0.7788],
                [0.8825, 1.0000, 0.8825],
                [0.7788, 0.8825, 0.7788],
            ],
        },
        {
            "dist_thresh": 3,
            "num_heatmaps": 1,
            "in_shape": (5, 5),
            "out_shape": (5, 5),
            "centers": [[1, 1], [2, 2]],
            "expected_output": [
                [0.7788, 0.8825, 0.7788, 0.5353, 0.3679],
                [0.8825, 1.0000, 0.8825, 0.7788, 0.5353],
                [0.7788, 0.8825, 1.0000, 0.8825, 0.6065],
                [0.5353, 0.7788, 0.8825, 0.7788, 0.5353],
                [0.3679, 0.5353, 0.6065, 0.5353, 0.3679],
            ],
        },
        {
            "dist_thresh": 1,
            "num_heatmaps": 1,
            "in_shape": (4, 4),
            "out_shape": (4, 4),
            "centers": [[1, 1]],
            "expected_output": [
                [0.1054, 0.3247, 0.1054, 0.0036],
                [0.3247, 1.0, 0.3247, 0.0111],
                [0.1054, 0.3247, 0.1054, 0.0036],
                [0.0036, 0.0111, 0.0036, 0.0001]
            ],
        },
    ],
)
def test_gaussian_heatmap_generation_single_keypoint(data):
    dist_thresh = data["dist_thresh"]
    generator = HeatmapGaussianGenerator(
        num_heatmaps=data["num_heatmaps"],
        pos_dist_thresh=dist_thresh,
        heatmap_mode=HeatmapGaussianGenerator.Mode.KEYPOINT,
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


@pytest.mark.parametrize(
    "batch_size, num_keypoints, image_size",
    [(2, 2, (64, 64)), (1, 5, (48, 64)), (15, 50, (64, 48))],
)
def test_random_gaussian_target_generation(
    batch_size: int, num_keypoints: int, image_size: tuple, num_animals=1
):
    # generate annotations
    annotations = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }  # batch size, num animals, num keypoints, 2 for x,y

    # model stride 1
    stride = 1

    # generate predictions
    predicted_heatmaps = {
        "heatmap": torch.zeros((batch_size, num_keypoints, *image_size))
    }

    # generate heatmap
    generator = HeatmapGaussianGenerator(
        num_heatmaps=num_keypoints,
        pos_dist_thresh=17,
        heatmap_mode=HeatmapGaussianGenerator.Mode.KEYPOINT,
        generate_locref=False,
    )
    targets = generator(stride, predicted_heatmaps, annotations)
    target_heatmap = targets["heatmap"]["target"].reshape(
        batch_size, num_keypoints, image_size[0] * image_size[1]
    )

    # get coords of max value of the heatmap
    gaus_max = torch.argmax(target_heatmap, dim=2)

    # get unraveled coords
    x = gaus_max % image_size[1]
    y = gaus_max // image_size[1]

    # get heatmap center tensor
    predict_kp = torch.stack((x, y), dim=-1)
    # Remove num_animals dimension - only one animal is supported
    annotations["keypoints"] = torch.squeeze(annotations["keypoints"], dim=1)

    # compare heatmap center to annotation
    assert torch.eq(annotations["keypoints"], predict_kp).all().item()
