pytorch_cfg_template = {}

pytorch_cfg_template["cfg_path"] = "/data/quentin/datasets/daniel3mouse/config.yaml"
pytorch_cfg_template["seed"] = 42
pytorch_cfg_template["device"] = "cuda:0"
pytorch_cfg_template["display_iters"] = 1000
pytorch_cfg_template["save_epochs"] = 50  # not iterations, epochs

pytorch_cfg_template["data"] = {
    "scale_jitter": [0.5, 1.25],
    "rotation": 30,
    "translation": 40,
    "hist_eq": True,
    "motion_blur": True,
    "covering": True,
    "gaussian_noise": 0.05 * 255,
    "normalize_images": True,
}

pytorch_cfg_template["model"] = {
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
        "type": "PlateauGenerator",
        "locref_stdev": 7.2801,
        "num_joints": -1,
        "pos_dist_thresh": 17,
    },
    "pose_model": {
        "stride": 8,
    },
}

pytorch_cfg_template["optimizer"] = {
    "type": "AdamW",
    "params": {
        "lr": 1e-4,
    },
}

pytorch_cfg_template["scheduler"] = {
    "type": "LRListScheduler",
    "params": {
        "milestones": [
            90,
            120,
        ],
        "lr_list": [[1e-5], [1e-6]],
    },
}

pytorch_cfg_template["predictor"] = {
    "type": "SinglePredictor",
    "num_animals": -1,
    "location_refinement": True,
    "locref_stdev": 7.2801,
}

pytorch_cfg_template["criterion"] = {
    "type": "PoseLoss",
    "loss_weight_locref": 0.02,
    "locref_huber_loss": True,
}

pytorch_cfg_template["solver"] = {"type": "BottomUpSingleAnimalSolver"}

pytorch_cfg_template["pos_dist_thresh"] = 17
pytorch_cfg_template["with_center"] = False
pytorch_cfg_template["batch_size"] = 1
pytorch_cfg_template["epochs"] = 200

if __name__ == "__main__":
    import yaml

    with open("pytorch_config.yaml", "w") as f:
        yaml.safe_dump(pytorch_cfg_template, f)
