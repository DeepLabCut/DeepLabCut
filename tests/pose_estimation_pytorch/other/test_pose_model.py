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
import copy

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.models as dlc_models
from deeplabcut.pose_estimation_pytorch.models import CRITERIONS, PREDICTORS, TARGET_GENERATORS
from deeplabcut.pose_estimation_pytorch.models.criterions import LOSS_AGGREGATORS
from deeplabcut.pose_estimation_pytorch.models.modules import AdaptBlock, BasicBlock

backbones_dicts = [
    {
        "type": "HRNet",
        "model_name": "hrnet_w32",
        "output_channels": 480,
        "stride": 4,
        "interpolate_branches": True,
    },
    {
        "type": "HRNet",
        "model_name": "hrnet_w18",
        "output_channels": 270,
        "stride": 4,
        "interpolate_branches": True,
    },
    {
        "type": "HRNet",
        "model_name": "hrnet_w48",
        "output_channels": 720,
        "stride": 4,
        "interpolate_branches": True,
    },
    {
        "type": "HRNet",
        "model_name": "hrnet_w32",
        "output_channels": 32,
        "interpolate_branches": False,
        "increased_channel_count": False,
        "stride": 4,
    },
    {
        "type": "HRNet",
        "model_name": "hrnet_w18",
        "output_channels": 18,
        "interpolate_branches": False,
        "increased_channel_count": False,
        "stride": 4,
    },
    {
        "type": "HRNet",
        "model_name": "hrnet_w48",
        "output_channels": 48,
        "interpolate_branches": False,
        "increased_channel_count": False,
        "stride": 4,
    },
    {"type": "ResNet", "model_name": "resnet50_gn", "output_channels": 2048, "stride": 32},
]

heads_dicts = [
    {
        "type": "HeatmapHead",
        "predictor": {
            "type": "HeatmapPredictor",
            "location_refinement": True,
            "locref_std": 7.2801,
        },
        "target_generator": {
            "type": "HeatmapPlateauGenerator",
            "num_heatmaps": "num_bodyparts",
            "pos_dist_thresh": 17,
            "heatmap_mode": "KEYPOINT",
            "generate_locref": True,
            "locref_std": 7.2801,
        },
        "criterion": {
            "heatmap": {
                "type": "WeightedBCECriterion",
                "weight": 1.0,
            },
            "locref": {
                "type": "WeightedHuberCriterion",
                "weight": 0.05,
            },
        },
        "heatmap_config": {
            "channels": [2048, 1024, -1],
            "kernel_size": [2, 2],
            "strides": [2, 2],
        },
        "locref_config": {
            "channels": [2048, 1024, -1],
            "kernel_size": [2, 2],
            "strides": [2, 2],
        },
        "output_channels": -1,
        "input_channels": 2048,
        "total_stride": 4,
    },
    {
        "type": "TransformerHead",
        "predictor": {
            "type": "HeatmapPredictor",
            "location_refinement": False,
        },
        "target_generator": {
            "type": "HeatmapPlateauGenerator",
            "num_heatmaps": "num_bodyparts",
            "pos_dist_thresh": 17,
            "heatmap_mode": "KEYPOINT",
            "generate_locref": False,
        },
        "criterion": {"type": "WeightedBCECriterion"},
        "dim": 192,
        "hidden_heatmap_dim": 384,
        "heatmap_dim": -1,
        "apply_multi": True,
        "heatmap_size": [-1, -1],
        "apply_init": True,
        "total_stride": 1,
        "input_channels": -1,
        "output_channels": -1,
        "head_stride": 1,
    },
    {
        "type": "DEKRHead",
        "predictor": {
            "type": "DEKRPredictor",
            "num_animals": 1,
            "keypoint_score_type": "heatmap",
            "max_absorb_distance": 75,
        },
        "target_generator": {
            "type": "DEKRGenerator",
            "num_joints": "num_bodyparts",
            "pos_dist_thresh": 17,
            "bg_weight": 0.1,
        },
        "criterion": {
            "heatmap": {
                "type": "WeightedBCECriterion",
                "weight": 1.0,
            },
            "offset": {
                "type": "WeightedHuberCriterion",
                "weight": 0.03,
            },
        },
        "heatmap_config": {
            "channels": [480, 64, -1],
            "num_blocks": 1,
            "dilation_rate": 1,
            "final_conv_kernel": 1,
            "block": BasicBlock,
        },
        "offset_config": {
            "channels": [480, -1, -1],
            "num_offset_per_kpt": 15,
            "num_blocks": 1,
            "dilation_rate": 1,
            "final_conv_kernel": 1,
            "block": AdaptBlock,
        },
        "total_stride": 1,
        "input_channels": 480,
        "output_channels": -1,
    },
]


def _backbone_id(backbone: dict) -> str:
    """A stable test id, so a backbone can be selected by name from the CLI or an IDE."""
    if backbone["type"] == "HRNet":
        branches = "interpolated" if backbone.get("interpolate_branches") else "branches"
        return f"{backbone['model_name']}-{branches}"

    return backbone["model_name"]


BACKBONE_CASES = [pytest.param(backbone, id=_backbone_id(backbone)) for backbone in backbones_dicts]
HEAD_CASES = [pytest.param(head, id=head["type"]) for head in heads_dicts]


@pytest.mark.parametrize("backbone_dict", BACKBONE_CASES)
def test_backbone(backbone_dict):
    # Sizes divisible by 64, to be able to predict the output size consistently
    # (and to be able to do the forward pass of HRNet)
    rng = np.random.default_rng(0)
    x_size, y_size = (int(size) - int(size) % 64 for size in rng.integers(100, 1001, size=2))

    input_size = (x_size, y_size)
    backbone_dict = copy.deepcopy(backbone_dict)
    input_tensor = torch.Tensor(1, 3, input_size[1], input_size[0])

    stride = backbone_dict.pop("stride")
    output_channels = backbone_dict.pop("output_channels")
    backbone = dlc_models.BACKBONES.build(backbone_dict)

    features = backbone(input_tensor)
    _, c, h, w = features.shape
    assert c == output_channels
    assert h == input_size[1] // stride
    assert w == input_size[0] // stride


@pytest.mark.parametrize("head_dict", HEAD_CASES)
def test_head(head_dict):
    # Sizes divisible by 4, to be able to predict the output size consistently
    rng = np.random.default_rng(0)
    w, h = (int(size) - int(size) % 4 for size in rng.integers(8, 501, size=2))
    num_keypoints = int(rng.integers(2, 51))

    head_dict = copy.deepcopy(head_dict)

    head_type = head_dict["type"]
    input_channels = head_dict.pop("input_channels")
    output_channels = head_dict.pop("output_channels")
    total_stride = head_dict.pop("total_stride")
    if head_type == "HeatmapHead":
        output_channels = num_keypoints
        head_dict["heatmap_config"]["channels"][2] = output_channels
        head_dict["locref_config"]["channels"][2] = 2 * output_channels
        head_dict["target_generator"]["num_heatmaps"] = output_channels
        input_tensor = torch.zeros((1, input_channels, h, w))

    elif head_type == "TransformerHead":
        output_channels = num_keypoints
        input_channels = num_keypoints
        head_dict["heatmap_dim"] = h * w
        head_dict["heatmap_size"] = [h, w]
        head_dict["target_generator"]["num_heatmaps"] = output_channels
        input_tensor = torch.zeros((1, input_channels, head_dict["dim"] * 3))

    elif head_type == "DEKRHead":
        output_channels = num_keypoints + 1
        head_dict["target_generator"]["num_joints"] = num_keypoints
        head_dict["heatmap_config"]["channels"][2] = num_keypoints + 1
        head_dict["offset_config"]["channels"][1] = num_keypoints * head_dict["offset_config"]["num_offset_per_kpt"]
        head_dict["offset_config"]["channels"][2] = num_keypoints
        input_tensor = torch.zeros((1, input_channels, h, w))

    if "type" in head_dict["criterion"]:
        head_dict["criterion"] = CRITERIONS.build(head_dict["criterion"])
    else:
        weights = {}
        criterions = {}
        for loss_name, criterion_cfg in head_dict["criterion"].items():
            weights[loss_name] = criterion_cfg.get("weight", 1.0)
            criterion_cfg = {k: v for k, v in criterion_cfg.items() if k != "weight"}
            criterions[loss_name] = CRITERIONS.build(criterion_cfg)

        aggregator_cfg = {"type": "WeightedLossAggregator", "weights": weights}
        head_dict["aggregator"] = LOSS_AGGREGATORS.build(aggregator_cfg)
        head_dict["criterion"] = criterions

    head_dict["target_generator"] = TARGET_GENERATORS.build(head_dict["target_generator"])
    head_dict["predictor"] = PREDICTORS.build(head_dict["predictor"])
    head = dlc_models.HEADS.build(head_dict)

    output = head(input_tensor)["heatmap"]
    _, c_out, h_out, w_out = output.shape
    assert (h_out == h * total_stride) and (w_out == w * total_stride)
    assert c_out == output_channels


def test_msa_hrnet():
    # TODO: build microsoft asia hrnet and check dimension of output
    # TODO: check if hyperparameters are loaded correctly (from the config file)
    pass


def test_msa_tokenpose():
    # TODO: build microsoft asia hrnet and check dimension of output
    # TODO: check if hyperparameters are loaded correctly (from the config file)
    # cf https://github.com/amathislab/BUCTDdev/blob/main/lib/models/transpose_h.py#L1
    pass


def test_msa_hrnetCOAM():
    # TODO: build BUCTD COAM hrnet and check dimension of output
    # TODO: check if hyperparameters are loaded correctly (from the config file)
    pass


# TODO: add other model variants our pipeline can build ;)
