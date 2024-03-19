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
from unittest.mock import Mock, patch

import pytest

from deeplabcut.pose_estimation_tensorflow.export import load_model



@pytest.mark.parametrize("multi_animal", [True, False])
@pytest.mark.parametrize("tf_gpu_inference", [True, False])
def test_load_model_calls_correct_setup_pose_method(
    tmpdir_factory, multi_animal, tf_gpu_inference
):
    export_path = "deeplabcut.pose_estimation_tensorflow.export"
    with (
        patch(f"{export_path}.predict") as mock_predict,
        patch(f"{export_path}.load_config") as mock_load_config,
        patch(f"{export_path}.auxiliaryfunctions.get_model_folder") as mock_get_folder,
        patch(f"{export_path}.tf") as _,
    ):
        # Project parameters
        train_fraction = 0.8
        model_folder_name = "model_folder"
        dataset_type = "multi-animal-imgaug" if multi_animal else "imgaug"

        # Create project and train folder
        project_dir = tmpdir_factory.mktemp("project")
        model_folder = project_dir / model_folder_name
        model_folder.mkdir()
        train_dir = model_folder / "train"
        train_dir.mkdir()

        # Create fake snapshots
        for idx in [10, 1000, 10000]:
            with open(train_dir / f"snapshot-{idx}.index", "w") as f:
                f.write("")

        # Create a fake configuration file
        cfg = {
            "TrainingFraction": [train_fraction],
            "project_path": str(project_dir),
            "snapshotindex": -1,
        }
        train_cfg = {
            "dataset_type": dataset_type,
            "location_refinement": False,
        }

        # Mock parts of code that should not be used
        mock_setup_pose_prediction = Mock()
        mock_setup_gpu_pose_prediction = Mock()
        mock_setup_pose_prediction.return_value = Mock(), Mock(), Mock()
        mock_setup_gpu_pose_prediction.return_value = Mock(), Mock(), Mock()

        mock_predict.setup_pose_prediction = mock_setup_pose_prediction
        mock_predict.setup_GPUpose_prediction = mock_setup_gpu_pose_prediction
        mock_get_folder.return_value = model_folder_name
        mock_load_config.return_value = train_cfg

        # Check that loading the model uses the TFGPUinference variable
        load_model(cfg, TFGPUinference=tf_gpu_inference)
        print(tf_gpu_inference)
        print(mock_predict.mock_calls)
        print(mock_setup_pose_prediction.mock_calls)
        print(mock_setup_gpu_pose_prediction.mock_calls)
    
        if tf_gpu_inference:
            mock_setup_pose_prediction.assert_not_called()
            mock_setup_gpu_pose_prediction.assert_called_once_with(train_cfg)
        else:
            mock_setup_pose_prediction.assert_called_once_with(train_cfg)
            mock_setup_gpu_pose_prediction.assert_not_called()
