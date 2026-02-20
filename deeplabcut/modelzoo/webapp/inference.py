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
from typing import Dict

import numpy as np

import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import update_config


class SingletonTopDownRunners:
    """Singleton class for topdown runners

    This class is a singleton class for topdown runners. It is used to
    ensure that only one instance of the topdown runners is created.

    Attrs:
        config: Configuration dictionary
        pose_model_path: Path to the pose model
        detector_model_path: Path to the detector model
        num_bodyparts: Number of bodyparts
        max_individuals: Maximum number of individuals
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config,
        pose_model_path: str,
        detector_model_path: str,
        num_bodyparts: int,
        max_individuals: int,
    ):

        pose_runner, detector_runner = get_inference_runners(
            config,
            snapshot_path=pose_model_path,
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=0,
            detector_path=detector_model_path,
        )
        self.pose_runner = pose_runner
        self.detector_runner = detector_runner


class SuperanimalPyTorchInference:
    """Superanimal inference class

    This class is used to perform inference on a superanimal model from the
    DeepLabCut model zoo website.
    """

    def __init__(
        self,
        project_name: str,
        pose_model_type: str = "hrnet_w32",
        detector_model_type: str = "fasterrcnn_resnet50_fpn_v2",
        max_individuals: int = 30,
        device: str = "cpu",
    ):
        self.max_individuals = max_individuals
        config = modelzoo.load_super_animal_config(
            super_animal=project_name,
            model_name=pose_model_type,
            detector_name=detector_model_type,
        )
        # TODO @deruyter92: super animal configs are currently not 
        # validated against the pydantic schema. We should add this functionality.
        config = update_config(config, max_individuals, device)
        self._config = config

    def initialize_models(self, pose_model_path: str, detector_model_path: str):
        self.models = SingletonTopDownRunners(
            self.config,
            pose_model_path,
            detector_model_path,
            len(self.config["bodyparts"]),
            self.max_individuals,
        )

    @property
    def config(self):
        return self._config

    def predict(self, frames: Dict[str, np.array]):

        input_images = np.array(list(frames.values()), dtype=float)

        bbox_predictions = self.models.detector_runner.inference(images=input_images)
        input_images = list(zip(input_images, bbox_predictions))
        predictions = self.models.pose_runner.inference(images=input_images)
        predictions = [
            {("markers" if k == "bodyparts" else k): v for k, v in d.items()}
            for d in predictions
        ]
        predictions = [
            {**item[1], "image_path": item[0]}
            for item in zip(frames.keys(), predictions)
        ]
        responses = {
            "joint_names": self.config["bodyparts"],
            "predictions": predictions,
        }

        return responses
