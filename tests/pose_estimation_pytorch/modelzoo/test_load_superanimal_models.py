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
import dlclibrary
import pytest
import torch

from deeplabcut.pose_estimation_pytorch.modelzoo import get_super_animal_snapshot_path


@pytest.mark.skip(reason="require-models")
def test_load_superanimal_models_weights_only():
    super_animal_names = dlclibrary.get_available_datasets()
    for super_animal in super_animal_names:
        print(f"\nTesting {super_animal}")
        for detector in dlclibrary.get_available_detectors(super_animal):
            print(super_animal, detector)
            path = get_super_animal_snapshot_path(super_animal, detector)
            snapshot = torch.load(path, map_location="cpu", weights_only=True)

        for pose_model in dlclibrary.get_available_models(super_animal):
            print(super_animal, pose_model)
            path = get_super_animal_snapshot_path(super_animal, pose_model)
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
