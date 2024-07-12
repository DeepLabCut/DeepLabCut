#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""General tests for the metrics API"""
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import deeplabcut.core.metrics as metrics


def test_computing_metrics_with_no_predictions():
    gt = np.arange(5 * 6 * 3).astype(float).reshape((5, 6, 3))
    gt[..., 2] = 2
    metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": np.zeros((0, 12, 3))},
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )


@pytest.mark.parametrize("error", [0.5, 1, 2])
def test_computing_metrics_with_constant_error(error):
    # only works for small errors: otherwise another matching can be found
    gt = np.arange(5 * 6 * 3).astype(float).reshape((5, 6, 3))
    gt[..., 2] = 2
    predictions = gt.copy()
    predictions[..., 2] = 0.9
    predictions[..., :2] += error
    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert_almost_equal(results["rmse"], np.sqrt(2) * error)
    assert_almost_equal(results["rmse_pcutoff"], np.sqrt(2) * error)
