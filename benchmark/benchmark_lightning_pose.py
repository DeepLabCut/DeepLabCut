"""Code to make an ablation study with different image augmentation parameters"""

from __future__ import annotations

from pathlib import Path
from deeplabcut.utils import get_bodyparts

from benchmark_run_experiments import (
    AUG_TRAIN,
    DEFAULT_OPTIMIZER,
    DEFAULT_SCHEDULER,
    HRNET_BACKBONE,
    HRNET_BACKBONE_INCRE,
    HRNET_BACKBONE_INTER,
    main,
    RESNET_BACKBONE,
    RESNET_OPTIMIZER,
    RESNET_SCHEDULER,
)
from utils import Project, WandBConfig
from utils_models import HeadConfig, ModelConfig


LP_DLC_DATA_ROOT = Path("/home/niels/datasets/lightning-pose")
LP_DLC_BENCHMARKS = {
    "mirrorFish": Project(
        root=LP_DLC_DATA_ROOT,
        name="mirror-fish-rick-2023-10-26",
        iteration=1,  # ITERATION 0 IS THE PAPER DATA
    ),
    "mirrorMouse": Project(
        root=LP_DLC_DATA_ROOT,
        name="mirror-mouse-rick-2022-12-02",
        iteration=1,  # ITERATION 0 IS THE PAPER DATA
    ),
    "iblPaw": Project(
        root=LP_DLC_DATA_ROOT,
        name="ibl-paw-mic-2023-01-09",
        iteration=1,  # ITERATION 0 IS THE PAPER DATA
    ),
}


if __name__ == "__main__":
    # Project parameters
    PROJECT_NAME = "mirrorFish"
    PROJECT_BENCHMARKED = LP_DLC_BENCHMARKS[PROJECT_NAME]
    SPLITS_PATH = LP_DLC_DATA_ROOT / "lightning_pose_splits.json"
    CFG = PROJECT_BENCHMARKED.cfg
    NUM_BPT = len(get_bodyparts(CFG))

    # Train parameters
    EPOCHS = 200
    SAVE_EPOCHS = 25
    RESNET_BATCH_SIZE = 8
    HRNET_BATCH_SIZE = 4

    # logging params
    WANDB_PROJECT = "dlc3-benchmark-dev"
    BASE_TAGS = (f"project={PROJECT_NAME}", "server=m0")
    GROUP_UID = "base"

    model_configs = [
        ModelConfig(
            net_type="resnet_50",
            batch_size=RESNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            train_aug=AUG_TRAIN,
            inference_aug=None,
            backbone_config=RESNET_BACKBONE,
            head_config=HeadConfig.build_plateau_head(
                c_in=2048,
                c_out=NUM_BPT,
                deconv=[(NUM_BPT, 3, 2)],
                final_conv=False,
            ),
            optimizer_config=RESNET_OPTIMIZER,
            scheduler_config=RESNET_SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-resnet50",
                group=f"{PROJECT_NAME}-{GROUP_UID}-resnet50",
                tags=(*BASE_TAGS, "arch=resnet50", "ndeconv=1"),
            ),
        ),
        ModelConfig(
            net_type="hrnet_w32",
            batch_size=HRNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            train_aug=AUG_TRAIN,
            inference_aug=None,
            backbone_config=HRNET_BACKBONE_INTER,
            head_config=HeadConfig.build_plateau_head(
                c_in=480,
                c_out=NUM_BPT,
                deconv=[(NUM_BPT, 3, 2)],
                final_conv=False,
            ),
            optimizer_config=DEFAULT_OPTIMIZER,
            scheduler_config=DEFAULT_SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32-inter",
                group=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32-inter",
                tags=(*BASE_TAGS, "arch=hrnet32-inter", "ndeconv=1"),
            ),
        ),
        ModelConfig(
            net_type="hrnet_w32",
            batch_size=HRNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            train_aug=AUG_TRAIN,
            inference_aug=None,
            backbone_config=HRNET_BACKBONE,
            head_config=HeadConfig.build_plateau_head(
                c_in=32,
                c_out=NUM_BPT,
                deconv=[(NUM_BPT, 3, 2)],
                final_conv=False,
            ),
            optimizer_config=DEFAULT_OPTIMIZER,
            scheduler_config=DEFAULT_SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-base-hrnet32",
                group=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32",
                tags=(*BASE_TAGS, "arch=hrnet32", "ndeconv=1"),
            ),
        ),
    ]

    main(
        project=PROJECT_BENCHMARKED,
        splits_file=SPLITS_PATH,
        trainset_index=0,
        train_fraction=0.81,
        models_to_train=model_configs,
        splits_to_train=(0, 1, 2, 3, 4),
    )
