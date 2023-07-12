import torch
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions

BACKBONE_OUT_CHANNELS = {
    "resnet-50": 2048,
    "resnet-50": 2048,
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

    """

    single_animal_nets = [
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
    ]

    multi_animal_nets = [
        "dekr_w18",
        "dekr_w32",
        "dekr_w48",
        "token_pose_w18",
        "token_pose_w32",
        "token_pose_w48",
    ]

    bodyparts = auxiliaryfunctions.get_bodyparts(project_config)
    num_joints = len(bodyparts)
    pytorch_config = config_template
    pytorch_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    pytorch_config["method"] = "bu"
    if net_type in single_animal_nets:
        pytorch_config["model"]["heads"] = make_single_head_cfg(num_joints, net_type)
        pytorch_config["model"]["target_generator"]["num_joints"] = num_joints
        pytorch_config["predictor"]["num_animals"] = 1

        if "efficientnet" in net_type:
            raise NotImplementedError("efficientnet config not yet implemented")
        elif "mobilenetv2" in net_type:
            raise NotImplementedError("mobilenet config not yet implemented")
        elif "hrnet" in net_type:
            raise NotImplementedError("hrnet config not yet implemented")

    elif net_type in multi_animal_nets:
        num_animals = len(project_config.get("individuals", [0]))
        if "dekr" in net_type:
            version = net_type.split("_")[-1]
            backbone_type = "hrnet_" + version
            num_offset_per_kpt = 15
            pytorch_config["data"]["auto_padding"] = {
                "min_height": 64,
                "min_width": 64,
                "pad_width_divisor": 32,
                "pad_height_divisor": 32,
            }
            pytorch_config["model"]["backbone"] = {
                "type": "HRNet",
                "model_name": "hrnet_" + version,
            }
            pytorch_config["model"]["heads"] = make_dekr_head_cfg(
                num_joints, backbone_type, num_offset_per_kpt
            )
            pytorch_config["model"]["target_generator"] = {
                "type": "DEKRGenerator",
                "num_joints": num_joints,
                "pos_dist_thresh": 17,
            }

            pytorch_config["predictor"] = {
                "type": "DEKRPredictor",
                "num_animals": num_animals,
            }

            pytorch_config["with_center"] = True
        elif "token_pose" in net_type:
            pytorch_config["method"] = "td"
            version = net_type.split("_")[-1]
            backbone_type = "hrnet_" + version
            pytorch_config['data']['auto_padding'] = {
                'min_height': 64,
                'min_width': 64,
                'pad_width_divisor': 32,
                'pad_height_divisor': 32,
            }
            pytorch_config["detector"] = make_detector_cfg()
            pytorch_config["model"] = make_token_pose_model_cfg(
                num_joints, backbone_type
            )
            pytorch_config["predictor"] = {
                "type": "HeatmapOnlyPredictor",
                "num_animals": 1,
            }
            pytorch_config["criterion"] = {"type": "HeatmapOnlyLoss"}
            pytorch_config["solver"] = {
                "type": "TopDownSolver",
            }
            pytorch_config["with_center"] = False
        else:
            raise NotImplementedError(
                "Currently no other model than dekr and token_pose are implemented"
            )

    else:
        raise ValueError("This net type is not supported by pytorch verison")

    if augmenter_type == None:
        pytorch_config["data"] = {}
    elif augmenter_type != "default" and augmenter_type != None:
        raise NotImplementedError(
            "Other augmentations than default are not implemented"
        )

    return pytorch_config


def make_single_head_cfg(num_joints: int, net_type: str):
    head_configs = []
    heatmap_heag_cfg, locref_head_cfg = {}, {}

    if "resnet" in net_type:
        heatmap_heag_cfg = {
            "type": "SimpleHead",
            "channels": [2048, 1024, num_joints],
            "kernel_size": [2, 2],
            "strides": [2, 2],
        }
        head_configs.append(heatmap_heag_cfg)

        locref_head_cfg = {
            "type": "SimpleHead",
            "channels": [2048, 1024, 2 * num_joints],
            "kernel_size": [2, 2],
            "strides": [2, 2],
        }
        head_configs.append(locref_head_cfg)

    return head_configs


def make_dekr_head_cfg(num_joints: int, backbone_type: str, num_offset_per_kpt: int):
    head_configs = []
    heatmap_heag_cfg, offset_head_cfg = {}, {}

    heatmap_heag_cfg = {
        "type": "HeatmapDEKRHead",
        "channels": [
            BACKBONE_OUT_CHANNELS[backbone_type],
            64,
            num_joints + 1,
        ],  # +1 since we need center
        "num_blocks": 1,
        "dilation_rate": 1,
        "final_conv_kernel": 1,
    }
    head_configs.append(heatmap_heag_cfg)

    offset_head_cfg = {
        "type": "OffsetDEKRHead",
        "channels": [
            BACKBONE_OUT_CHANNELS[backbone_type],
            num_offset_per_kpt * num_joints,
            num_joints,
        ],
        "num_offset_per_kpt": num_offset_per_kpt,
        "num_blocks": 1,
        "dilation_rate": 1,
        "final_conv_kernel": 1,
    }
    head_configs.append(offset_head_cfg)

    return head_configs


def make_token_pose_model_cfg(num_joints, backbone_type):
    model_cfg = {}
    model_cfg["backbone"] = {
        "type": "HRNetTopDown",
        "model_name": backbone_type,
    }

    model_cfg["neck"] = {
        "type": "Transformer",
        "feature_size": [64, 64],
        "patch_size": [4, 4],
        "num_keypoints": num_joints,
        "channels": 32,
        "dim": 192,
        "heads": 8,
        "depth": 6,
    }

    model_cfg["heads"] = []
    model_cfg["heads"].append(
        {
            "type": "TransformerHead",
            "dim": 192,
            "hidden_heatmap_dim": 384,
            "heatmap_dim": 4096,
            "apply_multi": True,
            "heatmap_size": [64, 64],
            "apply_init": False,
        }
    )

    model_cfg["target_generator"] = {
        "type": "PlateauWithoutLocref",
        "num_joints": num_joints,
        "pos_dist_thresh": 17,
    }

    model_cfg["pose_model"] = {"stride": 4}
    return model_cfg


def make_detector_cfg():
    detector_cfg = {}

    detector_cfg["detector_model"] = {
        "type": "FasterRCNN",
    }

    detector_cfg["detector_optimizer"] = {
        "type": "SGD",
        "params": {"lr": 0.01},
    }

    detector_cfg["detector_scheduler"] = {
        "type": "LRListScheduler",
        "params": {"milestones": [10, 430], "lr_list": [[0.05], [0.005]]},
    }

    detector_cfg["detector_max_epochs"] = 500

    detector_cfg["detector_save_epochs"] = 100

    return detector_cfg
