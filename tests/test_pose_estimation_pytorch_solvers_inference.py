import pytest
import numpy as np
import pandas as pd
import torch
from torch import nn
import deeplabcut.pose_estimation_pytorch.solvers.inference as dlc_pose_estimation_pytorch_solvers_inference


# Sample test data
cfg = {
    'location_refinement': True,
    'locref_stdev': 0.1,
    'pcutoff': 0.5
}
output = (torch.randn(2, 10, 64, 64), torch.randn(2, 10, 64, 64))
stride = (8, 8)
prediction = pd.DataFrame(
    {
        ("scorer1", "likelihood", "bodypart1"): [0.8, 0.9],
        ("scorer1", "x", "bodypart1"): [1.0, 2.0],
        ("scorer1", "y", "bodypart1"): [3.0, 4.0],
    }
)
target = pd.DataFrame(
    {
        ("scorer2", "likelihood", "bodypart1"): [0.8, 0.9],
        ("scorer2", "x", "bodypart1"): [1.5, 2.5],
        ("scorer2", "y", "bodypart1"): [3.5, 4.5],
    }
)
bodyparts = [("bodypart1",)]

def test_multi_pose_predict():
    scmap = np.random.rand(64, 64, 10)
    locref = np.random.rand(64, 64, 10, 2)
    stride = (8, 8)
    num_outputs = 5
    pose = dlc_pose_estimation_pytorch_solvers_inference.multi_pose_predict(scmap, locref, stride, num_outputs)
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (10, 15)

def test_get_prediction_invalid_output():
    # Test get_prediction function with invalid output
    with pytest.raises(Exception):
        invalid_output = (torch.randn(2, 10, 64,

         64),)  # Missing locref
        dlc_pose_estimation_pytorch_solvers_inferenceget_prediction(cfg, invalid_output, stride)

def test_get_prediction():
    predictions = dlc_pose_estimation_pytorch_solvers_inference.get_prediction(cfg, output, stride)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (output[0].shape[0], output[0].shape[1], 3)


@pytest.mark.parametrize(
            "test_n_top", 
            [
                (10),
                (4),
                (15),
                (20),
                (1)
            ])


def test_get_top_values(test_n_top):
    """
        Tests if n_tops are actually selected
    """
    test_scmap = np.random.rand(5, 64, 64, 6) 
    batchsize, ny, nx, num_joints = test_scmap.shape
    top_vals = dlc_pose_estimation_pytorch_solvers_inference.get_top_values(test_scmap,test_n_top)
    assert len(top_vals[0]) == test_n_top





