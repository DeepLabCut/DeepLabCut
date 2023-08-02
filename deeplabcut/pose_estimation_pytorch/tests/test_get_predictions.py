from itertools import product

import deeplabcut
import pytest
import torch
from deeplabcut.generate_training_dataset.make_pytorch_config import *
from deeplabcut.pose_estimation_pytorch.apis import inference_utils, utils
from deeplabcut.pose_estimation_pytorch.default_config import *
from deeplabcut.pose_estimation_pytorch.models.detectors import DETECTORS, BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.models.predictors import (
    PREDICTORS,
    BasePredictor,
)
from deeplabcut.pose_estimation_pytorch.tests.test_utils import (
    write_config
)
from deeplabcut.utils import auxiliaryfunctions

# Check implemented net types
single_nets = [
    "resnet_50",
]
multi_nets = [
    "dekr_w18",
]
multi_nets_td = [
    "token_pose_w32",
]

single = [ele for ele in product(single_nets, [False])]
multi = [ele for ele in product(multi_nets, [True])]
multi_td = [ele for ele in product(multi_nets_td, [True])]

params_bu = single + multi

@pytest.mark.parametrize("net_type, multianimal", params_bu)
def test_get_predictions_bottom_up(
    net_type: str,
    multianimal: bool,
    batch_size: int = 8,
    image_shape: tuple = (128, 256),
):
    # Create a batch image tensors
    images = torch.rand((batch_size, 3, image_shape[1], image_shape[0])) * 100
    # Create config and pytorch_config dicts
    cfg = write_config(multianimal)
    # read N animals and N keypoints from config
    num_animals = len(cfg["individuals"]) if "individuals" in cfg.keys() else 1
    num_keypoints = (
        len(cfg["multianimalbodyparts"])
        if cfg["multianimalproject"]
        else len(cfg["bodyparts"])
    )
    pytorch_config = make_pytorch_config(
        cfg, net_type, config_template=pytorch_cfg_template.copy()
    )
    # Pretrained set to False to initialize model without using a snapshot
    pytorch_config["model"]["backbone"]["pretrained"] = False
    # build model
    model = utils.build_pose_model(pytorch_config["model"], pytorch_config)

    # build predictor
    predictor: BasePredictor = PREDICTORS.build(dict(pytorch_config["predictor"]))

    # get predictions
    with torch.no_grad():
        output = inference_utils.get_predictions_bottom_up(model, predictor, images)

    # Generate test tensor with expected output shape
    test = torch.randint(1, 12, (batch_size, num_animals, num_keypoints, 3))

    assert test.shape == output.shape


# Doesn't work yet since top down doesn't fully work
@pytest.mark.parametrize("net_type, multianimal", multi_td)
def test_get_predicitons_top_down(
    net_type: str,
    multianimal: bool,
    batch_size: int = 8,
    num_animals: int = 10,
    num_keypoints: int = 10,
    image_shape: tuple = (128, 256),
):
    # Create a batch image tensors
    images = torch.rand((batch_size, 3, image_shape[1], image_shape[0])) * 100
    # Create config and pytorch_config dicts
    cfg = write_config(multianimal)
    pytorch_config = make_pytorch_config(
        cfg, net_type, config_template=pytorch_cfg_template.copy()
    )
    # Pretrained set to False to initialize model without using a snapshot
    pytorch_config["model"]["backbone"]["pretrained"] = False
    # read N animals and N keypoints from config
    num_animals = len(cfg["individuals"]) if "individuals" in cfg.keys() else 1
    num_keypoints = (
        len(cfg["multianimalbodyparts"])
        if cfg["multianimalproject"]
        else len(cfg["bodyparts"])
    )

    # build detector
    detector: BaseDetector = DETECTORS.build(
        dict(pytorch_config["detector"]["detector_model"])
    )
    # build model
    model = utils.build_pose_model(pytorch_config["model"], pytorch_config)

    # build predictors
    top_down_predictor: BasePredictor = PREDICTORS.build(
        {"type": "TopDownPredictor", "format_bbox": "xyxy"}
    )
    pose_predictor: BasePredictor = PREDICTORS.build(dict(pytorch_config["predictor"]))

    # get predictions
    with torch.no_grad():
        output = inference_utils.get_predictions_top_down(
            detector,
            top_down_predictor,
            model,
            pose_predictor,
            images,
            num_animals,
            num_keypoints,
            device="cpu",
        )

    # Generate test tensor with expected output shape
    test = torch.randint(1, 12, (batch_size, num_animals, num_keypoints, 3))

    assert test.shape == output.shape
