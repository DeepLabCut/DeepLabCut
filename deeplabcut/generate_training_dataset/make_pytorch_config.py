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
from __future__ import annotations

import torch
from copy import deepcopy
from itertools import combinations

from deeplabcut.utils import auxiliaryfunctions


BACKBONE_OUT_CHANNELS = {
    "resnet-50": 2048,
    "mobilenet_v2_1.0": 1280,
    "mobilenet_v2_0.75": 1280,
    "mobilenet_v2_0.5": 1280,
    "mobilenet_v2_0.35": 1280,
    "efficientnet-b0": 1280,
    "efficientnet-b1": 1280,
    "efficientnet-b2": 1408,
    "efficientnet-b3": 1536,
    "efficientnet-b4": 1792,
    "efficientnet-b5": 2048,
    "efficientnet-b6": 2304,
    "efficientnet-b7": 2560,
    "efficientnet-b8": 2816,
    "hrnet_w18": 270,
    "hrnet_w32": 480,
    "hrnet_w48": 720,
}
SUPPORTED_MODELS = (
    "resnet_50",
    "mobilenet_v2_1.0",
    "mobilenet_v2_0.75",
    "mobilenet_v2_0.5",
    "mobilenet_v2_0.35",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "efficientnet-b8",
    "hrnet_w18",
    "hrnet_w32",
    "hrnet_w48",
    "dekr_w18",
    "dekr_w32",
    "dekr_w48",
    "token_pose_w18",
    "token_pose_w32",
    "token_pose_w48",
)


def make_pytorch_config(
    project_config: dict,
    net_type: str,
    augmenter_type: str = "default",
    config_template: dict = None,
):
    """
    Currently supported net types :
        Single Animal :
            - resnet-50
            - mobilenet_v2_1.0
            - mobilenet_v2_0.75
            - mobilenet_v2_0.5
            - mobilenet_v2_0.35
            - efficientnet-b0
            - efficientnet-b1
            - efficientnet-b2
            - efficientnet-b3
            - efficientnet-b4
            - efficientnet-b5
            - efficientnet-b6
            - efficientnet-b7
            - efficientnet-b8
            - hrnet_w18
            - hrnet_w32
            - hrnet_w48

        Multi Animal:
            - dekr_w18
            - dekr_w32
            - dekr_w48

        Multi Animal top-down models:
            - token_pose_w18
            - token_pose_w32
            - token_pose_w48

    """
    if net_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported network {net_type}.")

    bodyparts = auxiliaryfunctions.get_bodyparts(project_config)
    num_joints = len(bodyparts)
    unique_bpts = auxiliaryfunctions.get_unique_bodyparts(project_config)
    num_unique_bpts = len(unique_bpts)
    compute_unique_bpts = num_unique_bpts > 0
    identification_head = project_config.get("identity")
    pytorch_config = deepcopy(config_template)
    pytorch_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    pytorch_config["method"] = "bu"
    if project_config.get("multianimalproject", False):
        num_individuals = len(project_config.get("individuals", [0]))
        if "dekr" in net_type:
            version = net_type.split("_")[-1]
            backbone_type = "hrnet_" + version
            num_offset_per_kpt = 15
            pytorch_config["data"]["auto_padding"] = {
                "min_height": None,
                "min_width": None,
                "pad_width_divisor": 32,
                "pad_height_divisor": 32,
            }
            pytorch_config["model"]["backbone"] = {
                "type": "HRNet",
                "model_name": "hrnet_" + version,
            }
            pytorch_config["model"]["heads"] = {
                "bodypart": make_dekr_head_cfg(
                    num_individuals, num_joints, backbone_type, num_offset_per_kpt
                ),
            }
            pytorch_config["with_center_keypoints"] = True

            if compute_unique_bpts:
                pytorch_config["model"]["heads"]["unique_bodypart"] = make_unique_bodyparts_head(
                    num_unique_bpts, backbone_type
                )

            if identification_head:
                pytorch_config["model"]["heads"]["identity"] = make_identity_head(
                    num_individuals,
                    backbone_out_channels=BACKBONE_OUT_CHANNELS[backbone_type],
                )

        elif "resnet" in net_type:
            num_stages = pytorch_config.get("num_stages", 0)
            dim = BACKBONE_OUT_CHANNELS["resnet-50"] if num_stages == 0 else 2304
            graph = [
                list(edge) for edge in combinations(range(num_joints), 2)
            ]  # TODO Parse from config
            num_limbs = len(graph)
            pytorch_config["model"]["backbone"] = {"type": "ResNet"}
            pytorch_config["model"]["heads"] = {
                "bodypart": make_dlcrnet_head(
                    num_joints,
                    num_unique_bpts,
                    num_individuals,
                    graph,
                    edges_to_keep=list(
                        pytorch_config.get("paf_best", list(range(num_limbs)))
                    ),
                    # TODO Below is hardcoded for output stride 32;
                    heatmap_channels=[dim, dim // 2, num_joints],
                    locref_channels=[dim, dim // 2, 2 * num_joints],
                    paf_channels=[dim, dim // 2, 2 * num_limbs],
                    num_stages=num_stages,
                    # TODO Set remaining params from config
                )
            }
            if identification_head:
                pytorch_config["model"]["heads"]["identity"] = make_identity_head(
                    num_individuals,
                    backbone_out_channels=dim,
                )
            # pytorch_config["data"]["crop_sampling"] = {
            #     "height": 400,
            #     "width": 400,
            #     "max_shift": 0.4,
            #     "method": "hybrid",
            # }
        elif "token_pose" in net_type:
            if compute_unique_bpts:
                raise NotImplementedError(
                    "Unique body parts are currently not handled by top down models"
                )
            if identification_head:
                raise NotImplementedError(
                    "Identification heads are currently not handled by top down models"
                )

            pytorch_config["method"] = "td"
            version = net_type.split("_")[-1]
            backbone_type = "hrnet_" + version
            pytorch_config["data_detector"] = make_detector_data_aug()
            pytorch_config["detector"] = make_detector_cfg(num_individuals)
            pytorch_config["model"] = make_token_pose_model_cfg(
                num_joints, backbone_type
            )
            pytorch_config["criterion"] = {"type": "HeatmapOnlyCriterion"}
            pytorch_config["runner"] = {"type": "PoseRunner"}
            pytorch_config["with_center_keypoints"] = False
        else:
            raise NotImplementedError(
                "Currently no other model than dlcrnet, dekr, and token_pose are implemented"
            )
    else:
        pytorch_config["model"]["heads"] = {
            "bodypart": make_single_head_cfg(num_joints, net_type)
        }

        if "efficientnet" in net_type:
            raise NotImplementedError("efficientnet config not yet implemented")
        elif "mobilenetv2" in net_type:
            raise NotImplementedError("mobilenet config not yet implemented")
        elif "hrnet" in net_type:
            raise NotImplementedError("hrnet config not yet implemented")

    if augmenter_type == None:
        pytorch_config["data"] = {}
    elif augmenter_type != "default" and augmenter_type != None:
        raise NotImplementedError(
            "Other augmentations than default are not implemented"
        )

    return pytorch_config


def make_dlcrnet_head(
    num_joints: int,
    num_unique_joints: int,
    num_animals: int,
    graph: list[tuple[int, int]],
    edges_to_keep: list[int],
    heatmap_channels: list[int],
    locref_channels: list[int],
    paf_channels: list[int],
    locref_weight: float = 0.05,
    paf_weight: float = 0.1,
    paf_width: int = 20,
    nms_radius: int = 5,
    sigma: float = 1.0,
    min_affinity: float = 0.05,
    num_stages: int = 5,
) -> dict:
    dict_ = make_heatmap_head(num_joints, heatmap_channels, locref_channels)
    dict_["type"] = "DLCRNetHead"
    dict_["criterion"]["locref"]["weight"] = locref_weight
    dict_["criterion"]["paf"] = {"type": "WeightedHuberCriterion", "weight": paf_weight}
    n_deconv_layers = len(paf_channels) - 1
    dict_["paf_config"] = {
        "channels": paf_channels,
        "kernel_size": [3] * n_deconv_layers,
        "strides": [2] * n_deconv_layers,
    }
    dict_["num_stages"] = num_stages
    dict_["target_generator"] = {
        "type": "SequentialGenerator",
        "generators": [
            dict_["target_generator"],
            {"type": "PartAffinityFieldGenerator", "graph": graph, "width": paf_width},
        ],
    }
    dict_["predictor"] = {
        "type": "PartAffinityFieldPredictor",
        "num_animals": num_animals,
        "num_multibodyparts": num_joints,
        "num_uniquebodyparts": num_unique_joints,
        "nms_radius": nms_radius,
        "sigma": sigma,
        "locref_stdev": 7.2801,
        "min_affinity": min_affinity,
        "graph": graph,
        "edges_to_keep": edges_to_keep,
    }
    return dict_


def make_heatmap_head(
    num_heatmaps: int, heatmap_channels: list[int], locref_channels: list[int] | None
) -> dict:
    n_deconv_heatmap = len(heatmap_channels) - 1
    with_locref = (locref_channels is not None and len(locref_channels) > 0)
    head_config = {
        "type": "HeatmapHead",
        "predictor": {
            "type": "SinglePredictor",
            "location_refinement": with_locref,
            "locref_stdev": 7.2801,
            "num_animals": 1,
        },
        "target_generator": {
            "type": "HeatmapPlateauGenerator",
            "num_heatmaps": num_heatmaps,
            "pos_dist_thresh": 17,
            "heatmap_mode": "KEYPOINT",
            "generate_locref": with_locref,
            "locref_std": 7.2801,
        },
        "criterion": {
            "heatmap": {"type": "WeightedBCECriterion", "weight": 1.0},
        },
        "heatmap_config": {
            "channels": heatmap_channels,
            "kernel_size": [3] * n_deconv_heatmap,
            "strides": [2] * n_deconv_heatmap,
        },
    }

    if locref_channels:
        n_deconv_locref = len(locref_channels) - 1
        head_config["locref_config"] = {
            "channels": locref_channels,
            "kernel_size": [3] * n_deconv_locref,
            "strides": [2] * n_deconv_locref,
        }
        head_config["criterion"]["locref"] = {
            "type": "WeightedHuberCriterion",  # or WeightedMSECriterion
            "weight": 0.05,
        }

    return head_config


def make_identity_head(
    num_individuals: int, backbone_out_channels: list[int]
) -> dict:
    heatmap_head = make_heatmap_head(
        num_individuals,
        heatmap_channels=[backbone_out_channels, num_individuals],
        locref_channels=None,
    )
    heatmap_head["predictor"] = {
        "type": "IdentityPredictor",
        "apply_sigmoid": True,
    }
    heatmap_head["target_generator"]["heatmap_mode"] = "INDIVIDUAL"
    return heatmap_head


def make_single_head_cfg(num_joints: int, net_type: str) -> dict:
    """
    Args:
        num_joints: the number of keypoints to predict
        net_type: the type of neural net to make the head for

    Raises:
        NotImplementedError if unique bodyparts are not implemented for backbone_type

    Returns:
        the head configuration
    """
    if "resnet" in net_type:
        return make_heatmap_head(
            num_joints,
            heatmap_channels=[2048, 1024, num_joints],
            locref_channels=[2048, 1024, 2 * num_joints],
        )

    raise NotImplementedError(
        f"Heads for single animals are not yet implemented with a {net_type} "
        f"backbone"
    )


def make_unique_bodyparts_head(num_unique_bodyparts: int, backbone_type: str) -> dict:
    """Creates a deconvolutional head to predict unique bodyparts

    Args:
        num_unique_bodyparts: number of unique bodyparts
        backbone_type: type of the backbone

    Raises:
        NotImplementedError if unique bodyparts are not implemented for backbone_type

    Returns:
        The configs for the unique bodyparts heatmap and locref heads
    """
    if "hrnet" in backbone_type:
        # Only one deconvolutional layer since hrnet stride is 1/4
        heatmap_in_channels = BACKBONE_OUT_CHANNELS[backbone_type]
        head = make_heatmap_head(
            num_unique_bodyparts,
            heatmap_channels=[heatmap_in_channels, num_unique_bodyparts],
            locref_channels=[heatmap_in_channels, 2 * num_unique_bodyparts],
        )
        head["target_generator"]["label_keypoint_key"] = "keypoints_unique"
        return head

    raise NotImplementedError(
        f"Unique bodyparts prediction is not implemented yet for backbone {backbone_type}"
    )


def make_dekr_head_cfg(
    num_individuals: int, num_joints: int, backbone_type: str, num_offset_per_kpt: int
):
    return {
        "type": "DEKRHead",
        "target_generator": {
            "type": "DEKRGenerator",
            "num_joints": num_joints,
            "pos_dist_thresh": 17,
            "bg_weight": 0.1,
        },
        "criterion": {
            "heatmap": {"type": "WeightedBCECriterion", "weight": 1},
            "offset": {
                "type": "WeightedHuberCriterion",  # or WeightedMSECriterion
                "weight": 0.03,
            },
        },
        "predictor": {
            "type": "DEKRPredictor",
            "num_animals": num_individuals,
            "keypoint_score_type": "combined",
            "max_absorb_distance": 75,
        },
        "heatmap_config": {
            "channels": [
                BACKBONE_OUT_CHANNELS[backbone_type],
                64,
                num_joints + 1,
            ],  # +1 since we need center
            "num_blocks": 1,
            "dilation_rate": 1,
            "final_conv_kernel": 1,
        },
        "offset_config": {
            "channels": [
                BACKBONE_OUT_CHANNELS[backbone_type],
                num_offset_per_kpt * num_joints,
                num_joints,
            ],
            "num_offset_per_kpt": num_offset_per_kpt,
            "num_blocks": 2,
            "dilation_rate": 1,
            "final_conv_kernel": 1,
        },
    }


def make_token_pose_model_cfg(num_joints, backbone_type):
    return {
        "backbone": {
            "type": "HRNet",
            "model_name": backbone_type,
            "pretrained": True,
            "only_high_res": True,
        },
        "neck": {
            "type": "Transformer",
            "feature_size": [64, 64],
            "patch_size": [4, 4],
            "num_keypoints": num_joints,
            "channels": 32,
            "dim": 192,
            "heads": 8,
            "depth": 6,
        },
        "heads": {
            "bodypart": {
                "type": "TransformerHead",
                "target_generator": {
                    "type": "HeatmapPlateauGenerator",
                    "num_heatmaps": num_joints,
                    "pos_dist_thresh": 17,
                    "generate_locref": False,
                },
                "criterion": {"type": "WeightedBCECriterion"},
                "predictor": {"type": "HeatmapOnlyPredictor", "num_animals": 1},
                "dim": 192,
                "hidden_heatmap_dim": 384,
                "heatmap_dim": 4096,
                "apply_multi": True,
                "heatmap_size": [64, 64],
                "apply_init": True,
            }
        },
        "pose_model": {"stride": 4},
    }


def make_detector_cfg(num_individuals: int):
    return {
        "model": {"type": "FasterRCNN"},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "scheduler": {
            "type": "LRListScheduler",
            "params": {"milestones": [90], "lr_list": [[1e-5]]},
        },
        "runner": {
            "type": "DetectorRunner",
            "max_individuals": num_individuals,
        },
        "batch_size": 1,
        "epochs": 500,
        "save_epochs": 100,
        "display_iters": 500,
    }


def make_detector_data_aug() -> dict:
    return {
        "covering": False,
        "gaussian_noise": False,
        "hist_eq": False,
        "hflip": True,
        "motion_blur": False,
        "normalize_images": True,
        "rotation": 30,
        "scale_jitter": [0.5, 1.25],
    }
