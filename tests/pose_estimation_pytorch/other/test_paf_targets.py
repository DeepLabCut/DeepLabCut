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

from deeplabcut.pose_estimation_pytorch.models.target_generators import pafs_targets


@pytest.mark.parametrize(
    "batch_size, num_keypoints, image_size",
    [(2, 2, (64, 64)), (1, 5, (48, 64)), (8, 50, (64, 48))],
)
def test_paf_target_generation(
    batch_size: int, num_keypoints: int, image_size: tuple, num_animals=2
):
    labels = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }  # 2 for x,y coords
    graph = [(i, j) for i in range(num_keypoints) for j in range(i + 1, num_keypoints)]
    prediction = {
        "heatmap": torch.rand((batch_size, num_keypoints, image_size[0], image_size[1])),
        "paf": torch.rand((batch_size, len(graph) * 2, image_size[0], image_size[1])),
    }
    generator = pafs_targets.PartAffinityFieldGenerator(graph=graph, width=20)
    targets_output = generator(1, prediction, labels)
    assert targets_output["paf"]["target"].shape == (
        batch_size,
        len(graph) * 2,
        image_size[0],
        image_size[1],
    )
