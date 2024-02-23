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
import os

import cv2
import numpy as np
import pytest

from deeplabcut.modelzoo.webapp.inference import SuperanimalPyTorchInference
from deeplabcut.utils import auxiliaryfunctions


@pytest.mark.parametrize("max_individuals", [1, 3])
@pytest.mark.parametrize(
    "project_name", ["superanimal_quadruped", "superanimal_topviewmouse"]
)
@pytest.mark.parametrize("pose_model_type", ["hrnetw32"])
def test_class_init(project_name, pose_model_type, max_individuals):
    inference_pipeline = SuperanimalPyTorchInference(
        project_name, pose_model_type, max_individuals=max_individuals
    )

    assert isinstance(inference_pipeline.config, dict)
    assert inference_pipeline.config["bodyparts"]
    assert len(inference_pipeline.config["bodyparts"]) > 0


@pytest.mark.skip(reason="require-models")
@pytest.mark.parametrize(
    "project_name", ["superanimal_quadruped", "superanimal_topviewmouse"]
)
@pytest.mark.parametrize("pose_model_type", ["hrnetw32"])
def test_runner_init(project_name, pose_model_type):
    inference_pipeline = SuperanimalPyTorchInference(
        project_name, pose_model_type, max_individuals=1
    )
    weight_folder = f"{auxiliaryfunctions.get_deeplabcut_path()}/modelzoo/checkpoints"
    snapshot_path = f"{weight_folder}/{project_name}_{pose_model_type}.pth"
    detector_path = f"{weight_folder}/{project_name}_fasterrcnn.pt"

    inference_pipeline.initialize_models(snapshot_path, detector_path)

    assert inference_pipeline.models.pose_runner
    assert inference_pipeline.models.detector_runner


@pytest.mark.skip(reason="require-models")
@pytest.mark.parametrize("max_individuals", [10, 4, 1])
@pytest.mark.parametrize(
    "project_name", ["superanimal_quadruped", "superanimal_topviewmouse"]
)
@pytest.mark.parametrize("pose_model_type", ["hrnetw32"])
def test_predict(project_name, pose_model_type, max_individuals):
    inference_pipeline = SuperanimalPyTorchInference(
        project_name, pose_model_type, max_individuals=max_individuals
    )
    image_path = "img0001.png"
    weight_folder = f"{auxiliaryfunctions.get_deeplabcut_path()}/modelzoo/checkpoints"
    snapshot_path = f"{weight_folder}/{project_name}_{pose_model_type}.pth"
    detector_path = f"{weight_folder}/{project_name}_fasterrcnn.pt"

    inference_pipeline.initialize_models(snapshot_path, detector_path)
    frame = {image_path: np.random.rand(100, 100, 3)}
    response = inference_pipeline.predict(frame)

    assert isinstance(response, dict)
    assert response["joint_names"] == inference_pipeline.config["bodyparts"]
    assert response["predictions"][0]["markers"].shape == (
        max_individuals,
        len(inference_pipeline.config["bodyparts"]),
        3,
    )
    assert response["predictions"][0]["image_path"] == image_path
