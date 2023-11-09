import random

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.models as dlc_models
from deeplabcut.pose_estimation_pytorch.models.modules import AdaptBlock, BasicBlock

backbones_dicts = [
    {"type": "HRNet", "model_name": "hrnet_w32", "output_channels": 480, "stride": 4},
    {"type": "HRNet", "model_name": "hrnet_w18", "output_channels": 270, "stride": 4},
    {"type": "HRNet", "model_name": "hrnet_w48", "output_channels": 720, "stride": 4},
    {
        "type": "HRNetTopDown",
        "model_name": "hrnet_w32",
        "output_channels": 32,
        "stride": 4,
    },
    {
        "type": "HRNetTopDown",
        "model_name": "hrnet_w18",
        "output_channels": 18,
        "stride": 4,
    },
    {
        "type": "HRNetTopDown",
        "model_name": "hrnet_w48",
        "output_channels": 48,
        "stride": 4,
    },
    {"type": "ResNet", "model_name": "resnet50", "output_channels": 2048, "stride": 32},
]

heads_dicts = [
    {
        "type": "SimpleHead",
        "channels": [2048, 1024, -1],
        "kernel_size": [2, 2],
        "strides": [2, 2],
        "output_channels": -1,
        "input_channels": 2048,
        "total_stride": 4,
    },
    {
        "type": "TransformerHead",
        "dim": 192,
        "hidden_heatmap_dim": 384,
        "heatmap_dim": -1,
        "apply_multi": True,
        "heatmap_size": [-1, -1],
        "apply_init": True,
        "total_stride": 1,
        "input_channels": -1,
        "output_channels": -1,
    },
    {
        "type": "HeatmapDEKRHead",
        "channels": [480, 64, -1],
        "num_blocks": 1,
        "dilation_rate": 1,
        "final_conv_kernel": 1,
        "block": BasicBlock,
        "total_stride": 1,
        "input_channels": 480,
        "output_channels": -1,
    },
    {
        "type": "OffsetDEKRHead",
        "channels": [480, -1, -1],
        "num_offset_per_kpt": 15,
        "num_blocks": 1,
        "dilation_rate": 1,
        "final_conv_kernel": 1,
        "total_stride": 1,
        "input_channels": 480,
        "output_channels": -1,
    },
]


def _generate_random_backbone_inputs(i):
    # Returns sizes that are divisible by 64to be able to predict consistently output szie
    # (and be able to do the forward pass of HRNet)
    x_size_tmp, y_size_tmp = random.randint(100, 1000), random.randint(100, 1000)
    return (
        backbones_dicts[i],
        (x_size_tmp - x_size_tmp % 64, y_size_tmp - y_size_tmp % 64),
    )


@pytest.mark.parametrize(
    "backbone_dict, input_size",
    [_generate_random_backbone_inputs(i) for i in range(len(backbones_dicts))],
)
def test_backbone(backbone_dict, input_size):
    input_tensor = torch.Tensor(1, 3, input_size[1], input_size[0])

    stride = backbone_dict.pop("stride")
    output_channels = backbone_dict.pop("output_channels")
    backbone = dlc_models.BACKBONES.build(backbone_dict)

    features = backbone(input_tensor)
    _, c, h, w = features.shape
    assert c == output_channels
    assert h == input_size[1] // stride
    assert w == input_size[0] // stride


def _generate_random_head_inputs(i):
    # Returns sizes that are divisible by 64to be able to predict consistently output szie
    # (and be able to do the forward pass of HRNet)
    x_size_tmp, y_size_tmp = random.randint(8, 500), random.randint(8, 500)
    num_kpts = random.randint(2, 50)
    return (
        heads_dicts[i],
        (x_size_tmp - x_size_tmp % 4, y_size_tmp - y_size_tmp % 4),
        num_kpts,
    )


@pytest.mark.parametrize(
    "head_dict, input_shape, num_keypoints",
    [_generate_random_head_inputs(i) for i in range(len(heads_dicts))],
)
def test_head(head_dict, input_shape, num_keypoints):
    w, h = input_shape

    head_type = head_dict["type"]
    input_channels = head_dict.pop("input_channels")
    output_channels = head_dict.pop("output_channels")
    total_stride = head_dict.pop("total_stride")
    if head_type == "SimpleHead":
        output_channels = num_keypoints
        head_dict["channels"][2] = output_channels
        input_tensor = torch.zeros((1, input_channels, h, w))

    elif head_type == "TransformerHead":
        output_channels = num_keypoints
        input_channels = num_keypoints
        head_dict["heatmap_dim"] = h * w
        head_dict["heatmap_size"] = [h, w]
        input_tensor = torch.zeros((1, input_channels, head_dict["dim"] * 3))

    elif head_type == "HeatmapDEKRHead":
        output_channels = num_keypoints + 1
        head_dict["channels"][2] = output_channels
        input_tensor = torch.zeros((1, input_channels, h, w))

    elif head_type == "OffsetDEKRHead":
        output_channels = num_keypoints * 2
        head_dict["channels"][1] = num_keypoints * head_dict["num_offset_per_kpt"]
        head_dict["channels"][2] = num_keypoints
        input_tensor = torch.zeros((1, input_channels, h, w))

    head = dlc_models.HEADS.build(head_dict)

    output = head(input_tensor)
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
