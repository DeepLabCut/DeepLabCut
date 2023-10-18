from __future__ import annotations

import os

import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDataset
from deeplabcut.pose_estimation_pytorch.data.dlcproject import DLCProject
from deeplabcut.pose_estimation_pytorch.data.helper import merge_list_of_dicts


@pytest.mark.parametrize("repo_path", ["/home/anastasiia/DLCdev"])
def test_propertymeta_project(repo_path):
    project_root = os.path.join(
        repo_path,
        "examples",
        "openfield-Pranav-2018-10-30",
    )
    dlc_project = DLCProject(project_root, shuffle=1)

    for prop in dlc_project.properties:
        print(prop, getattr(dlc_project, prop))


@pytest.mark.parametrize(
    "repo_path, mode",
    [("/home/anastasiia/DLCdev", "train"), ("/home/anastasiia/DLCdev", "test")],
)
def test_propertymeta_dataset(repo_path, mode):
    repo_path = "/home/anastasiia/DLCdev"
    mode = "train"
    mode = "train"
    project_root = os.path.join(
        repo_path,
        "examples",
        "openfield-Pranav-2018-10-30",
    )
    dlc_project = DLCProject(project_root, shuffle=1)
    dataset = PoseDataset(dlc_project, mode)

    for prop in dataset.properties:
        print(prop, getattr(dataset, prop))


@pytest.mark.parametrize(
    "list_dicts, keys_to_include",
    [
        ([{"a": 1, "b": 2}, {"a": 3, "b": 4}], ["a"]),
        (
            [
                *[
                    {
                        "keypoints": np.random.randn(27, 3),
                        "images": np.random.randn(
                            256,
                            192,
                        ),
                    }
                ]
                * 10,
            ],
            [*["keypoints", "images"] * 10],
        ),
    ],
)
def test_merge_list_of_dicts(list_dicts, keys_to_include):
    result_dict = merge_list_of_dicts(list_dicts, keys_to_include)
    expected_result_dict = {}
    for dictionary in list_dicts:
        for key in dictionary:
            if key not in keys_to_include:
                continue
            else:
                if key not in expected_result_dict:
                    expected_result_dict[key] = []
                expected_result_dict[key].append(dictionary[key])
    assert result_dict == expected_result_dict
