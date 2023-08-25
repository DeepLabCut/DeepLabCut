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
import pytest

import deeplabcut.pose_estimation_tensorflow as pet


def make_single_animal_rmse_df(
    bodyparts,
    train_indices,
    test_indices,
    error_data=None,
) -> pd.DataFrame:
    if error_data is None:
        error_data = np.ones((len(train_indices) + len(test_indices), len(bodyparts)))
    return pd.DataFrame(error_data, columns=bodyparts)


def make_multi_animal_rmse_df(
    scorer,
    individuals,
    bodyparts,
    train_indices,
    test_indices,
    error_data=None,
) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts],
        names=["scorer", "individuals", "bodyparts"],
    )
    if error_data is None:
        error_data = np.ones(
            (len(train_indices) + len(test_indices), len(individuals) * len(bodyparts))
        )
    return pd.DataFrame(error_data, columns=columns)


KEYPOINT_ERROR_NAMES = [
    "Train error (px)",
    "Test error (px)",
    "Train error (px) with p-cutoff",
    "Test error (px) with p-cutoff",
]

KEYPOINT_ERROR_TEST_DATA = [
    (
        {
            "df_error": make_single_animal_rmse_df(
                bodyparts=["leg", "arm", "head"],
                train_indices=[0, 1, 3],
                test_indices=[2, 4],
            ),
            "train_indices": [0, 1, 3],
            "test_indices": [2, 4],
        },
        {
            "leg": [1.0, 1.0],  # train, test
            "arm": [1.0, 1.0],  # train, test
            "head": [1.0, 1.0],  # train, test
        },
    ),
    (
        {
            "df_error": make_single_animal_rmse_df(
                bodyparts=["leftHand", "rightHand"],
                train_indices=[0, 2],
                test_indices=[1, 3],
                error_data=[
                    [1.0, np.nan],
                    [1.0, 0.0],
                    [0.0, 10.0],
                    [5.0, 5.0],
                ],
            ),
            "train_indices": [0, 2],
            "test_indices": [1, 3],
        },
        {
            "leftHand": [0.5, 3.0],  # train, test
            "rightHand": [10.0, 2.5],  # train, test
        },
    ),
    (
        {
            "df_error": make_single_animal_rmse_df(
                bodyparts=["leg", "arm", "head"],
                train_indices=[0, 1, 3],
                test_indices=[2, 4],
            ),
            "train_indices": [0, 1, 3],
            "test_indices": [2, 4],
        },
        {
            "leg": [1.0, 1.0],  # train, test
            "arm": [1.0, 1.0],  # train, test
            "head": [1.0, 1.0],  # train, test
        },
    ),
    (
        {
            "df_error": make_multi_animal_rmse_df(
                scorer="john",
                individuals=["individual_1", "individual_2"],
                bodyparts=["leftArm", "rightArm"],
                train_indices=[0, 1, 3],
                test_indices=[2],
                error_data=[
                    # individual_1, individual2
                    # leftArm, rightArm, leftArm, rightArm
                    [1.0, np.nan, 1.0, 2.0],
                    [2.0, 0.0, 1.0, np.nan],
                    [3.0, 10.0, 1.0, np.nan],
                    [10.0, 4.0, np.nan, np.nan],
                ],
            ),
            "train_indices": [0, 1, 3],
            "test_indices": [2],
        },
        {
            "leftArm": [3.0, 2.0],  # train, test
            "rightArm": [2.0, 10.0],  # train, test
        },
    ),
]


@pytest.mark.parametrize("inputs, expected_values", KEYPOINT_ERROR_TEST_DATA)
def test_evaluate_keypoint_error(inputs, expected_values):
    keypoint_error = pet.keypoint_error(
        inputs["df_error"],
        inputs["df_error"],
        inputs["train_indices"],
        inputs["test_indices"],
    )
    print(inputs["df_error"])
    print(keypoint_error)
    for bodypart, mean_errors in expected_values.items():
        for error_name in KEYPOINT_ERROR_NAMES:
            if "train" in error_name.lower():
                mean_error = mean_errors[0]
            else:
                mean_error = mean_errors[1]

            assert keypoint_error.loc[error_name, bodypart] == mean_error
