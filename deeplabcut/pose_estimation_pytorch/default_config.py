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
from typing import Any

pytorch_cfg_template: dict[str, Any] = {
    "cfg_path": "/data/quentin/datasets/daniel3mouse/config.yaml",
    "seed": 42,
    "device": "cuda:0",
    "display_iters": 1000,
    "save_epochs": 50,
    "data": {
        "scale_jitter": [0.5, 1.25],
        "rotation": 30,
        "hist_eq": True,
        "motion_blur": True,
        "covering": True,
        "gaussian_noise": 0.05 * 255,
        "normalize_images": True,
    },
    "model": {
        "backbone": {
            "type": "ResNet",
            "pretrained": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        },
        "heatmap_head": {
            "type": "SimpleHead",
            "channels": [2048, 1024, -1],  # -1 acts as undefined here
            "kernel_size": [2, 2],
            "strides": [2, 2],
        },
        "locref_head": {
            "type": "SimpleHead",
            "channels": [2048, 1024, -1],  # -1 acts as undefined here
            "kernel_size": [2, 2],
            "strides": [2, 2],
        },
        "target_generator": {
            "type": "HeatmapPlateauGenerator",
            "num_heatmaps": -1,
            "pos_dist_thresh": 17,
            "generate_locref": True,
            "locref_std": 7.2801,
        },
        "pose_model": {"stride": 8},
    },
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
    "scheduler": {
        "type": "LRListScheduler",
        "params": {"milestones": [90, 120], "lr_list": [[1e-5], [1e-6]]},
    },
    "runner": {"type": "PoseRunner"},
    "with_center_keypoints": False,
    "batch_size": 1,
    "epochs": 200,
}

if __name__ == "__main__":
    import yaml

    with open("pytorch_config.yaml", "w") as f:
        yaml.safe_dump(pytorch_cfg_template, f)
