import torch
import pytest
import deeplabcut.pose_estimation_pytorch.models.predictors.dekr_predictor as dlc_pep_models_predictors_dekr_predictor

def test_DEKRPredictor():
    predictor = dlc_pep_models_predictors_dekr_predictor.DEKRPredictor(num_animals=2)
    outputs = (
        torch.randn(1, 18, 64, 64),  # example heatmap 
        torch.randn(1, 34, 64, 64),  # example offsets
    )
    scale_factors = (1.0, 0.5)

    try:
        poses_with_scores = predictor.forward(outputs, scale_factors)
    except Exception as e:
        pytest.fail(f"DEKRPredictor forward pass raised an exception: {e}")

    assert poses_with_scores.shape == (1, 2, 17, 3)

    assert torch.all(poses_with_scores[:, :, :, 2] >= 0)
    assert torch.all(poses_with_scores[:, :, :, 2] <= 1)


 