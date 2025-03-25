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
import pytest
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators import HeatmapGaussianGenerator


@pytest.mark.parametrize(
    "batch_size, num_keypoints, image_size",
    [(2, 2, (64, 64)), (1, 5, (48, 64)), (15, 50, (64, 48))],
)
def test_gaussian_target_generation(
    batch_size: int, num_keypoints: int, image_size: tuple, num_animals=1
):
    # generate annotations
    labels = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }  # batch size, num animals, num keypoints, 2 for x,y
    # generate predictions
    stride = 1
    prediction = {
        "heatmap": torch.rand((batch_size, num_keypoints, *image_size[:2])),
        "locref": torch.rand((batch_size, 2 * num_keypoints, *image_size[:2])),
    }

    # generate heatmap
    output = HeatmapGaussianGenerator(
        num_heatmaps=num_keypoints,
        pos_dist_thresh=17,
        locref_std=5.0,
    )
    output = output(stride, prediction, labels)["heatmap"]["target"].reshape(
        batch_size, num_keypoints, image_size[0] * image_size[1]
    )

    # get coords of max value of the heatmap
    gaus_max = torch.argmax(output, dim=2)

    # get unraveled coords
    x = gaus_max % image_size[1]
    y = gaus_max // image_size[1]

    # get heatmap center tensor
    predict_kp = torch.stack((x, y), dim=-1)
    # Remove num_animals dimension - only one animal is supported
    labels["keypoints"] = torch.squeeze(labels["keypoints"], dim=1)

    # compare heatmap center to annotation
    assert torch.eq(labels["keypoints"], predict_kp).all().item()
