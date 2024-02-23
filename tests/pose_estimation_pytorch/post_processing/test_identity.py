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
""" Tests identity matching """
import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.post_processing.identity import assign_identity


@pytest.mark.parametrize(
    "prediction, identity_scores, output_order",
    [
        (
            [
                [[0, 0, 1.0], [0, 0, 1.0]],  # assembly 1
                [[5, 5, 1.0], [5, 5, 1.0]],  # assembly 2
                [[9, 9, 1.0], [9, 9, 1.0]],  # assembly 3
            ],
            [  # a0 -> idv1, a1 -> idv2, a2 -> idv0
                [[0.1, 0.8, 0.3], [0.1, 0.7, 0.3]],  # assembly 1 ID scores
                [[0.2, 0.1, 0.6], [0.3, 0.1, 0.5]],  # assembly 2 ID scores
                [[0.7, 0.1, 0.1], [0.6, 0.2, 0.2]],  # assembly 3 ID scores
            ],
            [2, 0, 1],
        ),
        (
            [
                [[0, 0, 1.0], [0, 0, 1.0]],  # assembly 1
                [[1, 1, 1.0], [5, 5, 1.0]],  # assembly 2
                [[0, 0, 1.0], [9, 9, 1.0]],  # assembly 3
            ],
            [  # a0 -> idv0, a1 -> idv1, a2 -> idv2
                [[0.4, 0.4, 0.3], [0.5, 0.3, 0.3]],  # assembly 1 ID scores
                [[0.4, 0.4, 0.3], [0.3, 0.5, 0.4]],  # assembly 2 ID scores
                [[0.2, 0.2, 0.4], [0.2, 0.2, 0.3]],  # assembly 3 ID scores
            ],
            [0, 1, 2],
        ),
    ],
)
def test_single_identity_assignment(prediction, identity_scores, output_order):
    predictions = np.array(prediction)
    identity_scores = np.array(identity_scores)
    predictions_with_id = assign_identity([predictions], [identity_scores])

    print()
    print(predictions.shape)
    print(identity_scores.shape)

    np.testing.assert_equal(
        predictions[output_order],
        predictions_with_id[0],
    )
