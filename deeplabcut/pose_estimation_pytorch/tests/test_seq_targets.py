import torch
from itertools import combinations
from deeplabcut.pose_estimation_pytorch.models.target_generators import TARGET_GENERATORS


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
                "type": "PlateauGenerator",
                "locref_stdev": 7.2801,
                "num_joints": num_keypoints,
                "pos_dist_thresh": 17,
            },
            {
                "type": "PartAffinityFieldGenerator",
                "graph": graph,
                "width": 20,
            },
        ]
    }
    gen = TARGET_GENERATORS.build(cfg)

    annotations = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }
    prediction = [torch.rand((batch_size, num_keypoints, image_size[0], image_size[1]))]
    inputs = torch.rand(batch_size, 3, *image_size)
    head_outputs = {
        'heatmap': torch.rand(batch_size, num_keypoints, 32, 32),
        'locref': torch.rand(batch_size, num_keypoints * 2, 32, 32),
        'paf': torch.rand(batch_size, num_limbs * 2, 32, 32),
    }
    out = gen(inputs=inputs, outputs=head_outputs, labels=annotations)
    assert all(s in out for s in list(head_outputs))
    for k, v in head_outputs.items():
        assert out[k]['target'].shape == v.shape
