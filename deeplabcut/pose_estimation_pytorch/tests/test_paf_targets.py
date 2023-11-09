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
    annotations = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }  # 2 for x,y coords
    prediction = [torch.rand((batch_size, num_keypoints, image_size[0], image_size[1]))]
    graph = [(i, j) for i in range(num_keypoints) for j in range(i + 1, num_keypoints)]
    generator = pafs_targets.PartAffinityFieldGenerator(graph=graph, width=20)
    targets_output = generator(annotations, prediction, image_size)
    assert targets_output["paf"]["target"].shape == (
        batch_size,
        len(graph) * 2,
        image_size[0],
        image_size[1],
    )
