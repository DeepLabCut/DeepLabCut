from itertools import product

import pytest
import torch
from torchvision.transforms import Resize as TorchResize

from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.apis import inference, utils
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector, DETECTORS
from deeplabcut.pose_estimation_pytorch.models.predictors import (
    BasePredictor,
    PREDICTORS,
)
from deeplabcut.pose_estimation_pytorch.tests.test_utils import write_config

# Check implemented net types
single_nets = ["resnet_50"]
multi_nets = ["dekr_w18"]
multi_nets_td = ["token_pose_w32"]

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
    pytorch_config = make_pytorch_pose_config(
        project_config=cfg, pose_config_path="my/pytorch_config.yaml", net_type=net_type
    )
    # Pretrained set to False to initialize model without using a snapshot
    pytorch_config["model"]["backbone"]["pretrained"] = False
    # build model
    model = utils.build_pose_model(pytorch_config)

    # build predictor
    predictor: BasePredictor = PREDICTORS.build(dict(pytorch_config["predictor"]))

    # get predictions
    with torch.no_grad():
        output, _ = inference.get_predictions_bottom_up(model, predictor, images)

    # Generate test tensor with expected output shape
    test = torch.randint(1, 12, (batch_size, num_animals, num_keypoints, 3))

    assert test.shape == output.shape


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
    pytorch_config = make_pytorch_pose_config(
        project_config=cfg, pose_config_path="my/pytorch_config.yaml", net_type=net_type
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
    model = utils.build_pose_model(pytorch_config)

    # build predictors
    top_down_predictor: BasePredictor = PREDICTORS.build(
        {"type": "TopDownPredictor", "format_bbox": "xyxy"}
    )
    pose_predictor: BasePredictor = PREDICTORS.build(dict(pytorch_config["predictor"]))
    detector.eval()
    model.eval()
    pose_predictor.eval()
    top_down_predictor.eval()

    # get predictions
    with torch.no_grad():
        output, _ = inference.get_predictions_top_down(
            detector,
            model,
            pose_predictor,
            top_down_predictor,
            images,
            num_animals,
            num_keypoints,
            TorchResize((256, 256)),
        )

    # Generate test tensor with expected output shape
    test = torch.randint(1, 12, (batch_size, num_animals, num_keypoints, 3))

    assert test.shape == output.shape
