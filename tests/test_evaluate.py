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
import numpy as np
import pandas as pd

import deeplabcut.pose_estimation_tensorflow as pet


KEYPOINT_ERROR_NAMES = [
    "Train error (px)",
    "Test error (px)",
    "Train error (px) with p-cutoff",
    "Test error (px) with p-cutoff",
]


def make_single_animal_rmse_df(
    bodyparts,
    train_indices,
    test_indices,
) -> pd.DataFrame:
    error_data = np.ones(
        (len(train_indices) + len(test_indices), len(bodyparts))
    )
    return pd.DataFrame(error_data, columns=bodyparts)


def make_multi_animal_rmse_df(
    scorer,
    individuals,
    bodyparts,
    train_indices,
    test_indices,
) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts],
        names=["scorer", "individuals", "bodyparts"],
    )
    error_data = np.ones(
        (len(train_indices) + len(test_indices), len(individuals) * len(bodyparts))
    )
    return pd.DataFrame(error_data, columns=columns)


# TODO: Parametrize and test with NaNs
def test_evaluate_keypoint_error():
    bodyparts = ["leg", "arm", "head"]
    train_indices = [0, 1, 3]
    test_indices = [2]
    df_error = make_single_animal_rmse_df(bodyparts, train_indices, test_indices)
    keypoint_error = pet.keypoint_error(
        df_error,
        df_error,
        train_indices,
        test_indices,
    )
    for bodypart in bodyparts:
        for error_name in KEYPOINT_ERROR_NAMES:
            assert keypoint_error.loc[error_name, bodypart] == 1.0


# TODO: Parametrize and test with NaNs
def test_evaluate_keypoint_error_multianimal():
    scorer = "john"
    individuals = ["individual_1", "individual_2"]
    bodyparts = ["leg", "arm", "head"]
    train_indices = [0, 1, 3]
    test_indices = [2]
    df_error = make_multi_animal_rmse_df(
        scorer, individuals, bodyparts, train_indices, test_indices
    )
    keypoint_error = pet.keypoint_error(
        df_error,
        df_error,
        train_indices,
        test_indices,
    )
    for bodypart in bodyparts:
        for error_name in KEYPOINT_ERROR_NAMES:
            assert keypoint_error.loc[error_name, bodypart] == 1.0
