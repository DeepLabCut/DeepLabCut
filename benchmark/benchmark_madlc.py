"""Benchmark script for maDLC models"""
from __future__ import annotations

from deeplabcut.utils import get_bodyparts

from benchmark_run_experiments import main
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
        ("parenting", 0.95),
        ("trimouse", 0.95),
        ("fish", 0.94),
        ("marmoset", 0.95),
    ]:
        PROJECT_BENCHMARKED = MA_DLC_BENCHMARKS[PROJECT_NAME]
        SPLIT_FILE = MA_DLC_DATA_ROOT / "maDLC_benchmarking_splits.json"
        NUM_BPT = len(get_bodyparts(PROJECT_BENCHMARKED.cfg))

        AUG_TRAIN = ImageAugmentations(
            normalize=True,
            covering=True,
            gaussian_noise=12.75,
            hist_eq=True,
            motion_blur=True,
            affine=AffineAugmentation(
                p=0.5,
                rotation=30,
                scale=(1, 1),
                translation=40,
            ),
            collate=BatchCollate(
                min_scale=0.4,
                max_scale=1.0,
                min_short_side=256,
                max_short_side=1152,
                multiple_of=32,
            ),
        )

        # Optimization parameters
        OPTIMIZER = {"type": "AdamW", "params": {"lr": 1e-4}}
        SCHEDULER = {
            "type": "LRListScheduler",
            "params": {"lr_list": [[1e-5], [1e-6]], "milestones": [140, 190]},
        }

        # Train parameters
        DETECTOR_EPOCHS = 250
        DETECTOR_SAVE_EPOCHS = 50
        DETECTOR_BATCH_SIZE = 8

        EPOCHS = 200
        SAVE_EPOCHS = 25
        RESNET_BATCH_SIZE = 8
        DEKR_BATCH_SIZE = 4
        TD_HRNET_BATCH_SIZE = 16

        # logging params
        WANDB_PROJECT = "dlc3-benchmark-dev"
        BASE_TAGS = (f"project={PROJECT_NAME}", "server=m0")
        GROUP_UID = "base"

        RESNET_PAF = ModelConfig(
            net_type="resnet_50",
            batch_size=RESNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            dataloader_workers=2,
            dataloader_pin_memory=True,
            backbone_config=BackboneConfig(
                model_name="resnet50_gn",
                output_stride=16,
                freeze_bn_stats=True,
                freeze_bn_weights=False,
            ),
            train_aug=AUG_TRAIN,
            inference_aug=None,  # uses default
            optimizer_config=OPTIMIZER,
            scheduler_config=SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-resnet50PAF",
                group=f"{PROJECT_NAME}-{GROUP_UID}-resnet50PAF",
                tags=(*BASE_TAGS, "arch=resnet50PAF"),
            ),
        )
        DEKR_W32 = ModelConfig(
            net_type="dekr_w32",
            batch_size=DEKR_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            dataloader_workers=2,
            dataloader_pin_memory=True,
            train_aug=AUG_TRAIN,
            inference_aug=None,  # uses default
            optimizer_config=OPTIMIZER,
            scheduler_config=SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-dekr32",
                group=f"{PROJECT_NAME}-{GROUP_UID}-dekr32",
                tags=(*BASE_TAGS, "arch=dekr32"),
            ),
        )
        TD_HRNET_W32 = (
            DetectorConfig(
                batch_size=DETECTOR_BATCH_SIZE,
                epochs=DETECTOR_EPOCHS,
                save_epochs=DETECTOR_SAVE_EPOCHS,
                dataloader_workers=2,
                dataloader_pin_memory=True,
                train_aug=AUG_TRAIN,
                inference_aug=None,
                optimizer_config=None,
                scheduler_config=None,
            ),
            ModelConfig(
                net_type="top_down_hrnet_w32",
                batch_size=TD_HRNET_BATCH_SIZE,
                epochs=EPOCHS,
                save_epochs=SAVE_EPOCHS,
                dataloader_workers=2,
                dataloader_pin_memory=True,
                train_aug=AUG_TRAIN,
                inference_aug=None,
                backbone_config=None,
                head_config=HeadConfig.build_plateau_head(
                    c_in=32,
                    c_out=NUM_BPT,
                    deconv=[],
                    final_conv=True,
                ),
                optimizer_config=OPTIMIZER,
                scheduler_config=SCHEDULER,
                wandb_config=WandBConfig(
                    project=WANDB_PROJECT,
                    run_name=f"{PROJECT_NAME}-{GROUP_UID}-td-hrnet32",
                    group=f"{PROJECT_NAME}-{GROUP_UID}-td-hrnet32",
                    tags=(*BASE_TAGS, "arch=td-hrnet32", "ndeconv=0"),
                ),
            ),
        )

        main(
            project=PROJECT_BENCHMARKED,
            splits_file=SPLIT_FILE,
            trainset_index=0,
            train_fraction=TRAIN_FRACTION,
            models_to_train=[RESNET_PAF, DEKR_W32, TD_HRNET_W32],
            splits_to_train=(0, ),
        )
