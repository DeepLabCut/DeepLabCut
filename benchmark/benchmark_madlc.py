"""Benchmark script for maDLC models"""
from __future__ import annotations

import torch
import wandb
from deeplabcut.utils import get_bodyparts

from benchmark_run_experiments import (
    AUG_INFERENCE,
    AUG_TRAIN,
    CropSampling,
    DEFAULT_OPTIMIZER,
    DEFAULT_SCHEDULER,
    DetectorConfig,
    HeadConfig,
    main,
    ModelConfig,
    WandBConfig,
)
from projects import MA_DLC_BENCHMARKS, MA_DLC_DATA_ROOT


if __name__ == "__main__":
    PROJECT_NAME = "fish"  # "trimouse", "fish", "marmosets", "parenting"
    PROJECT_BENCHMARKED = MA_DLC_BENCHMARKS[PROJECT_NAME]
    SPLIT_FILE = MA_DLC_DATA_ROOT / "maDLC_benchmarking_splits.json"
    NUM_BPT = len(get_bodyparts(PROJECT_BENCHMARKED.cfg))

    # Train parameters
    DETECTOR_EPOCHS = 1
    DETECTOR_SAVE_EPOCHS = 1
    DETECTOR_BATCH_SIZE = 1

    DEFAULT_OPTIMIZER = {"type": "AdamW", "params": {"lr": 5e-4}}
    DEFAULT_SCHEDULER["params"] = {"lr_list": [[1e-4], [1e-5]], "milestones": [170, 200]}

    EPOCHS = 210
    SAVE_EPOCHS = 30
    DEKR_BATCH_SIZE = 8
    TD_HRNET_BATCH_SIZE = 8
    CTD_HRNET_BATCH_SIZE = 32

    # logging params
    WANDB_PROJECT = "dlc3-buctd-test"
    BASE_TAGS = (f"project={PROJECT_NAME}", "server=4")
    GROUP_UID = "base"

    model_configs = [
        # ModelConfig(
        #     net_type="dekr_w32",
        #     batch_size=DEKR_BATCH_SIZE,
        #     epochs=EPOCHS,
        #     save_epochs=SAVE_EPOCHS,
        #     train_aug=AUG_TRAIN,
        #     inference_aug=AUG_INFERENCE,
        #     optimizer_config=DEFAULT_OPTIMIZER,
        #     scheduler_config=DEFAULT_SCHEDULER,
        #     wandb_config=WandBConfig(
        #         project=WANDB_PROJECT,
        #         run_name=f"{PROJECT_NAME}-{GROUP_UID}-dekr32",
        #         group=f"{PROJECT_NAME}-{GROUP_UID}-dekr32",
        #         tags=(*BASE_TAGS, "arch=dekr32", "ndeconv=1"),
        #     ),
        # ),
        # (
        #     DetectorConfig(
        #         batch_size=DETECTOR_BATCH_SIZE,
        #         epochs=DETECTOR_EPOCHS,
        #         save_epochs=DETECTOR_SAVE_EPOCHS,
        #         train_aug=None,
        #         inference_aug=None,
        #         optimizer_config=None,
        #         scheduler_config=None,
        #     ),
        #     ModelConfig(
        #         net_type="top_down_hrnet_w32",
        #         batch_size=TD_HRNET_BATCH_SIZE,
        #         epochs=EPOCHS,
        #         save_epochs=SAVE_EPOCHS,
        #         train_aug=AUG_TRAIN,
        #         inference_aug=AUG_INFERENCE,
        #         backbone_config=None,
        #         head_config=HeadConfig.build_plateau_head(
        #             c_in=32,
        #             c_out=NUM_BPT,
        #             deconv=[],
        #             final_conv=True,
        #         ),
        #         optimizer_config=DEFAULT_OPTIMIZER,
        #         scheduler_config=DEFAULT_SCHEDULER,
        #         wandb_config=WandBConfig(
        #             project=WANDB_PROJECT,
        #             run_name=f"{PROJECT_NAME}-{GROUP_UID}-td-hrnet32",
        #             group=f"{PROJECT_NAME}-{GROUP_UID}-td-hrnet32",
        #             tags=(*BASE_TAGS, "arch=td-hrnet32", "ndeconv=0"),
        #         ),
        #     ),
        # ),
        ModelConfig(
            net_type="ctd_coam_w32",
            batch_size=CTD_HRNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            train_aug=AUG_TRAIN,
            inference_aug=AUG_INFERENCE,
            backbone_config=None,
            head_config=None,
            optimizer_config=DEFAULT_OPTIMIZER,
            scheduler_config=DEFAULT_SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-ctd-hrnet32",
                group=f"{PROJECT_NAME}-{GROUP_UID}-ctd-hrnet32",
                tags=(*BASE_TAGS, "arch=ctd-hrnet32"),
            ),
        ),
    ]

    main(
        project=PROJECT_BENCHMARKED,
        splits_file=SPLIT_FILE,
        trainset_index=0,
        train_fraction=0.94,
        models_to_train=[model_configs[0]],
        splits_to_train=(0, ),
    )
