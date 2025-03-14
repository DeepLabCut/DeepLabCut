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
from itertools import combinations

import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    TARGET_GENERATORS,
)


def test_sequential_generator():
    batch_size = 4
    image_size = 256, 256
    num_keypoints = 12
    num_animals = 2
    graph = [list(edge) for edge in combinations(range(num_keypoints), 2)]
    num_limbs = len(graph)
    cfg = {
        "type": "SequentialGenerator",
        "generators": [
            {
                "type": "HeatmapPlateauGenerator",
                "num_heatmaps": num_keypoints,
                "pos_dist_thresh": 17,
                "generate_locref": True,
                "locref_std": 7.2801,
            },
            {"type": "PartAffinityFieldGenerator", "graph": graph, "width": 20},
        ],
    }
    gen = TARGET_GENERATORS.build(cfg)

    annotations = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }
    head_outputs = {
        "heatmap": torch.rand(batch_size, num_keypoints, 32, 32),
        "locref": torch.rand(batch_size, num_keypoints * 2, 32, 32),
        "paf": torch.rand(batch_size, num_limbs * 2, 32, 32),
    }
    out = gen(stride=1, outputs=head_outputs, labels=annotations)
    assert all(s in out for s in list(head_outputs))
    for k, v in head_outputs.items():
        assert out[k]["target"].shape == v.shape
