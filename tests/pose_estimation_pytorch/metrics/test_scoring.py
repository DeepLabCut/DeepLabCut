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
"""Tests for the scoring methods"""
import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.metrics.scoring as scoring


@pytest.mark.parametrize(
    "data",
    [
        {
            "individuals": ["i1", "i2"],
            "bodyparts": ["arm"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[2.0, 2.0, 0.8]], [[1.0, 1.0, 0.7]],  # x, y, score
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.8, 0.5]], [[0.51, 0.49]],
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[1.0, 1.0, 2]], [[0, 0, 0]]  # x, y, visibility
                ]
            },
            "accuracy": {
                "arm_accuracy": 1.0,
            },
        },
        {
            "individuals": ["i1", "i2"],
            "bodyparts": ["arm"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[1.0, 1.0, 0.7]], [[2.0, 2.0, 0.7]],  # x, y, score
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.4, 0.6]], [[0.6, 0.4]]
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[2.0, 2.0, 2]], [[1.0, 1.0, 2]],  # x, y, visibility
                ]
            },
            "accuracy": {
                "arm_accuracy": 1.0,
            },
        },
        {
            "individuals": ["i1", "i2"],
            "bodyparts": ["arm"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[1.0, 1.0, 0.7]], [[2.0, 2.0, 0.7]],  # x, y, score
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.6, 0.4]], [[0.6, 0.4]]  # both assemblies assigned to idv 1
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[2.0, 2.0, 2]], [[1.0, 1.0, 2]],  # x, y, visibility
                ]
            },
            "accuracy": {
                "arm_accuracy": 0.5,
            },
        },
        {
            "individuals": ["i1", "i2"],
            "bodyparts": ["arm"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[1.0, 1.0, 0.7]], [[2.0, 2.0, 0.7]],  # x, y, score
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.6, 0.4]], [[0.4, 0.6]]  # both assigned to wrong ID
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[2.0, 2.0, 2]],  # x, y, visibility
                    [[1.0, 1.0, 2]],
                ]
            },
            "accuracy": {
                "arm_accuracy": 0.0,
            },
        },
        {
            "individuals": ["i1", "i2"],
            "bodyparts": ["arm", "leg"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[1.0, 1.0, 0.7], [10.0, 10.0, 0.9]],
                    [[100.0, 100.0, 0.9], [90.0, 90.9, 0.8]],
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.7, 0.3], [0.6, 0.2]],
                    [[0.6, 0.3], [0.6, 0.2]],  # should not matter, not assigned to GT
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[2.0, 2.0, 2], [8.0, 8.0, 2]],  # x, y, visibility
                    [[-1, -1, 0.0], [-1, -1, 0.0]],  # not visible
                ]
            },
            "accuracy": {
                "arm_accuracy": 1.0,
                "leg_accuracy": 1.0,
            },
        },
        {
            "individuals": ["i1", "i2", "i3"],
            "bodyparts": ["arm", "leg"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[1.0, 1.0, 0.7], [10.0, 10.0, 0.9]],
                    [[100.0, 100.0, 0.9], [90.0, 90.9, 0.8]],
                    [[110.0, 110.0, 0.9], [98.0, 91.9, 0.8]],
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.7, 0.3], [0.6, 0.2]],  # assigned to correct ID
                    [[0.6, 0.3], [0.6, 0.2]],  # should not matter, not assigned to GT
                    [[0.6, 0.3], [0.6, 0.2]],  # should not matter, not assigned to GT
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[2.0, 2.0, 2], [8.0, 8.0, 2]],  # x, y, visibility
                    [[-1, -1, 0.0], [-1, -1, 0.0]],  # not visible
                    [[-1, -1, 0.0], [-1, -1, 0.0]],  # not visible
                ]
            },
            "accuracy": {
                "arm_accuracy": 1.0,
                "leg_accuracy": 1.0,
            },
        },
        {
            "individuals": ["i1", "i2", "i3"],
            "bodyparts": ["arm", "leg"],
            "predictions": {
                "img0.png": [  # (num_assemblies, num_bodyparts, 3)
                    [[1.0, 1.0, 0.7], [10.0, 10.0, 0.9]],
                    [[100.0, 100.0, 0.9], [90.0, 90.9, 0.8]],
                    [[110.0, 110.0, 0.9], [98.0, 91.9, 0.8]],
                ],
            },
            "identity_scores": {
                "img0.png": [  # (num_assemblies, num_bodyparts, num_individuals)
                    [[0.7, 0.3, 0.1], [0.6, 0.2, 0.1]],  # assigned to correct ID
                    [[0.1, 0.2, 0.7], [0.4, 0.3, 0.2]],  # 1st correct, 2nd wrong
                    [[0.6, 0.3, 0.5], [0.6, 0.2, 0.4]],  # should not matter, not assigned to GT
                ],
            },
            "ground_truth": {
                "img0.png": [  # (num_individuals, num_bodyparts, 3)
                    [[2.0, 2.0, 2], [8.0, 8.0, 2]],  # x, y, visibility
                    [[-1, -1, 0.0], [-1, -1, 0.0]],  # not visible
                    [[90.0, 90, 2], [80, 80, 2.0]],  # x, y, visibility
                ]
            },
            "accuracy": {
                "arm_accuracy": 1.0,
                "leg_accuracy": 0.5,
            },
        },
    ],
)
def test_id_accuracy(data) -> None:
    scores = scoring.compute_identity_scores(
        individuals=data["individuals"],
        bodyparts=data["bodyparts"],
        predictions={k: np.array(v) for k, v in data["predictions"].items()},
        identity_scores={k: np.array(v) for k, v in data["identity_scores"].items()},
        ground_truth={k: np.array(v) for k, v in data["ground_truth"].items()},
    )
    assert scores == data["accuracy"]
