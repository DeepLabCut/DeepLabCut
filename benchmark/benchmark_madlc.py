"""Benchmark script for maDLC models"""
from __future__ import annotations

from deeplabcut.utils import get_bodyparts

from benchmark_run_experiments import main, DEFAULT_SCHEDULER
from projects import MA_DLC_BENCHMARKS, MA_DLC_DATA_ROOT
from utils import WandBConfig
from utils_augmentation import (
    AffineAugmentation,
    BatchCollate,
    ImageAugmentations,
)
from utils_models import BackboneConfig, HeadConfig, DetectorConfig, ModelConfig


if __name__ == "__main__":
    for PROJECT_NAME, TRAIN_FRACTION in [
        #("parenting", 0.95),
        #("trimouse", 0.95),
        ("fish", 0.94),
        #("marmoset", 0.95),
    ]:
        PROJECT_BENCHMARKED = MA_DLC_BENCHMARKS[PROJECT_NAME]
        SPLIT_FILE = MA_DLC_DATA_ROOT / "maDLC_benchmarking_splits.json"
        NUM_BPT = len(get_bodyparts(PROJECT_BENCHMARKED.cfg))

        # AUG_TRAIN = ImageAugmentations(
        #     normalize=True,
        #     covering=True,
        #     gaussian_noise=12.75,
        #     hist_eq=True,
        #     motion_blur=True,
        #     affine=AffineAugmentation(
        #         p=0.5,
        #         rotation=30,
        #         scale=(1, 1),
        #         translation=40,
        #     ),
        #     collate=BatchCollate(
        #         min_scale=0.4,
        #         max_scale=1.0,
        #         min_short_side=256,
        #         max_short_side=1152,
        #         multiple_of=32,
        #     ),
        # )
        AUG_TRAIN_TD = ImageAugmentations(
            normalize=True,
            #covering=True,
            covering=False,
            gaussian_noise=12.75,
            #hist_eq=True,
            hist_eq=False,
            #motion_blur=True,
            motion_blur=False,
            hflip=False,
            #hflip=True,
            affine=AffineAugmentation(
                p=0.5,
                #p=0.9,
                rotation=30,
                #scale=(0.5, 1.25),
                scale=(1.0, 1.0),
                #translation=1,
                translation=0,
            ),
            collate=None,
        )

        HRNET_VERSION = "w32"
        #HRNET_VERSION = "w48"

        DEFAULT_OPTIMIZER = {"type": "AdamW", "params": {"lr": 5e-4}}
        #DEFAULT_OPTIMIZER = {"type": "Adam", "params": {"lr": 1e-3}}
        DEFAULT_SCHEDULER["params"] = {"lr_list": [[1e-4], [1e-5]], "milestones": [170, 200]}

        EPOCHS = 210
        SAVE_EPOCHS = 30
        #DEKR_BATCH_SIZE = 8
        #TD_HRNET_BATCH_SIZE = 8
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
                net_type=f"ctd_coam_{HRNET_VERSION}",
                batch_size=CTD_HRNET_BATCH_SIZE,
                epochs=EPOCHS,
                save_epochs=SAVE_EPOCHS,
                dataloader_workers=2,
                dataloader_pin_memory=True,
                #train_aug=AUG_TRAIN,
                train_aug=AUG_TRAIN_TD,
                inference_aug=None,
                backbone_config=None,
                head_config=None,
                optimizer_config=DEFAULT_OPTIMIZER,
                scheduler_config=DEFAULT_SCHEDULER,
                wandb_config=WandBConfig(
                    project=WANDB_PROJECT,
                    run_name=f"{PROJECT_NAME}-{GROUP_UID}-ctd-hrnet{HRNET_VERSION}",
                    group=f"{PROJECT_NAME}-{GROUP_UID}-ctd-hrnet{HRNET_VERSION}",
                    tags=(*BASE_TAGS, f"arch=ctd-hrnet{HRNET_VERSION}"),
                ),
            ),
        ]

        main(
            project=PROJECT_BENCHMARKED,
            splits_file=SPLIT_FILE,
            trainset_index=0,
            train_fraction=TRAIN_FRACTION,
            models_to_train=[model_configs[0]],
            splits_to_train=(0, ),
            train=True,
            evaluate=True,
            manual_shuffle_index=None,
        )
