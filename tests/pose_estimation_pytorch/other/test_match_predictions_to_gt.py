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

import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.post_processing.match_predictions_to_gt as deeplabcut_torch_match_predictions_gt


@pytest.fixture
def animals_and_keypoints_invalid():
    """Summary:
    Fixture with invalid pred_kpts and gt_kpts shapes that will raise ValueErrors.

    Returns:
        tuple containing:
             predicted keypoints(pred_kpts), of shape num_animals, num_keypoints, (x,y,score)
             ground truth keypoints (gt_kpts), of shape num_animals, num_keypoints, (x,y)
             individual names (indv_names)
    """
    gt_kpts = 2 * np.ones((6, 6, 3))  # num animals, num keypoints, (x,y,vis)
    gt_kpts[:, :, :2] = np.random.rand(6, 6, 2)
    pred_kpts = np.random.rand(6, 8, 3)  # num animals, num keypoints, (x,y,score)
    indv_names = ["indv1", "indv2"]
    return pred_kpts, gt_kpts, indv_names


@pytest.fixture
def animals_and_keypoints():
    """Summary:
    Fixture with pred_kpts, gt_kpts shapes and indv_names.

    Returns:
        tuple containing:
             predicted keypoints(pred_kpts), of shape num_animals, num_keypoints, (x,y,score)
             ground truth keypoints (gt_kpts), of shape num_animals, num_keypoints, (x,y)
             individual names (indv_names)
    """
    gt_kpts = 2 * np.ones((6, 6, 3))  # num animals, num keypoints, (x,y,vis)
    gt_kpts[:, :, :2] = np.random.rand(6, 6, 2)

    # adding score value because the shape of pred_kpts should be (6,6,3)
    score = np.full((gt_kpts.shape[0], gt_kpts.shape[1], 1), 0.5)
    pred_kpts = np.concatenate((gt_kpts, score), axis=2)
    np.random.shuffle(pred_kpts)  # shuffle predicted keypoints

    indv_names = ["indv1", "indv2"]
    return pred_kpts, gt_kpts, indv_names


def test_invalid_rmse(animals_and_keypoints_invalid: tuple) -> None:
    """Summary:
    Tets if an invalid output really returns a ValueError in the rmse function.

    Args:
        animals_and_keypoints_invalid (tuple): containing predicted keypoints (pred_kpts),
        ground truth keypoints (gt_kpts) and individual names (indv_names).
    """
    pred_kpts, gt_kpts, indv_names = animals_and_keypoints_invalid

    with pytest.raises(ValueError):
        deeplabcut_torch_match_predictions_gt.rmse_match_prediction_to_gt(
            pred_kpts, gt_kpts
        )


def test_invalid_oks(animals_and_keypoints_invalid: tuple) -> None:
    """Summary:
    Test if an invalid output really returns a ValueError in the oks function.

    Args:
        animals_and_keypoints_invalid   (tuple): containing predicted keypoints (pred_kpts), ground truth keypoints (gt_kpts)
                and individual names (indv_names)
    """
    pred_kpts, gt_kpts, indv_names = animals_and_keypoints_invalid

    with pytest.raises(ValueError):
        deeplabcut_torch_match_predictions_gt.oks_match_prediction_to_gt(
            pred_kpts, gt_kpts, indv_names
        )


def test_rmse_match_predictions_to_gt(
    animals_and_keypoints: tuple, num_animals: int = 6
) -> None:
    """Summary:
    Test if rmse_match_prediction_to_gt function returns the expected shape output.

    Args:
        animals_and_keypoints (tuple): containing predicted keypoints (pred_kpts), ground truth keypoints (gt_kpts)
                and individual names (indv_names)
    """
    pred_kpts, gt_kpts, indv_names = animals_and_keypoints

    col_ind = deeplabcut_torch_match_predictions_gt.rmse_match_prediction_to_gt(
        pred_kpts, gt_kpts
    )
    assert isinstance(col_ind, np.ndarray)
    assert col_ind.shape == (num_animals,)


def test_oks_match_predictions_to_gt(
    animals_and_keypoints: tuple, num_animals: int = 6
) -> None:
    """Summary:
    Test if oks_match_predictions_to_gt function returns the expected shape output.

    Args:
        animals_and_keypoints (tuple): containing predicted keypoints (pred_kpts), ground truth keypoints (gt_kpts)
                and individual names (indv_names)
    """
    pred_kpts, gt_kpts, indv_names = animals_and_keypoints

    col_ind = deeplabcut_torch_match_predictions_gt.rmse_match_prediction_to_gt(
        pred_kpts, gt_kpts
    )
    assert isinstance(col_ind, np.ndarray)
    assert col_ind.shape == (num_animals,)


def test_extend_col_ind(animals_and_keypoints: tuple, num_animals: int = 6) -> None:
    """Summary:
    Test if the column indices have the expected shape.

    Args:
        animals_and_keypoints (tuple): containing predicted keypoints (pred_kpts), ground truth keypoints (gt_kpts)
                and individual names (indv_names)
    """
    pred_kpts, gt_kpts, indv_names = animals_and_keypoints

    col_ind = deeplabcut_torch_match_predictions_gt.rmse_match_prediction_to_gt(
        pred_kpts, gt_kpts
    )
    extended_array = deeplabcut_torch_match_predictions_gt.extend_col_ind(
        col_ind, num_animals
    )
    assert extended_array.shape == (num_animals,)
