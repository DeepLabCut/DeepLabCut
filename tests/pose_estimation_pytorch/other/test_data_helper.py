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
from __future__ import annotations

import os
from unittest.mock import patch, Mock
from zipfile import Path

import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
from deeplabcut.pose_estimation_pytorch.data.utils import merge_list_of_dicts
from deeplabcut.generate_training_dataset import create_training_dataset


def mock_aux() -> Mock:
    aux_functions = Mock()
    aux_functions.read_plainconfig = Mock()
    aux_functions.read_plainconfig.return_value = {}
    return aux_functions


@patch("deeplabcut.pose_estimation_pytorch.data.base.auxiliaryfunctions", mock_aux())
def _get_loader(project_root):
    if not (Path(project_root) / "training-datasets").exists():
        create_training_dataset(config=str(Path(project_root) / "config.yaml"))
    return DLCLoader(Path(project_root) / "config.yaml", shuffle=1)


@pytest.mark.skip
@pytest.mark.parametrize("repo_path", ["/home/anastasiia/DLCdev"])
def test_propertymeta_project(repo_path):
    project_root = os.path.join(repo_path, "examples", "openfield-Pranav-2018-10-30")
    dlc_loader = _get_loader(project_root)

    for prop in dlc_loader.properties:
        print(prop, getattr(dlc_loader, prop))


@pytest.mark.skip
@pytest.mark.parametrize(
    "repo_path, mode",
    [("/home/anastasiia/DLCdev", "train"), ("/home/anastasiia/DLCdev", "test")],
)
def test_propertymeta_dataset(repo_path, mode):
    repo_path = "/home/anastasiia/DLCdev"
    mode = "train"
    project_root = os.path.join(repo_path, "examples", "openfield-Pranav-2018-10-30")
    dlc_loader = _get_loader(project_root)
    dataset = dlc_loader.create_dataset(transform=None, mode=mode)

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
                        "images": np.random.randn(256, 192),
                    }
                ]
                * 10
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
