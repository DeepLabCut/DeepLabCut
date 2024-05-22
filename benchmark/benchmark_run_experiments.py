"""Code to make an ablation study with different image augmentation parameters"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import wandb

from deeplabcut.utils import get_bodyparts

from benchmark_train import EvalParameters, run_dlc, RunParameters
from projects import SA_DLC_BENCHMARKS, SA_DLC_DATA_ROOT
from utils import create_shuffles, Project, Shuffle, WandBConfig
from utils_augmentation import (
    AffineAugmentation,
    BatchCollate,
    ImageAugmentations,
)
from utils_models import (
    BackboneConfig,
    DetectorConfig,
    HeadConfig,
    ModelConfig,
)


def main(
    project: Project,
    splits_file: Path,
    trainset_index: int,
    train_fraction: float,
    models_to_train: list[ModelConfig | tuple[DetectorConfig, ModelConfig]],
    splits_to_train: tuple[int, ...] = (0, 1, 2),
    eval_params: EvalParameters | None = None,
    train: bool = True,
    evaluate: bool = True,
    manual_shuffle_index: int | None = None,
):
    if eval_params is None:
        eval_params = EvalParameters(snapshotindex="all", plotting=False)
        #eval_params = EvalParameters(snapshotindex=-1, plotting=True)

    project.update_iteration_in_config()
    for config in models_to_train:
        if wandb.run is not None:  # TODO: Finish wandb run in DLC
            wandb.finish()

        if isinstance(config, tuple):
            detector_config, model_config = config
            assert isinstance(detector_config, DetectorConfig)
            assert isinstance(model_config, ModelConfig)
        else:
            detector_config = None
            model_config = config
            assert isinstance(model_config, ModelConfig)

        run_name = ""
        tags: tuple[str, ...] = ()
        if model_config.wandb_config is not None:
            run_name = model_config.wandb_config.run_name
            tags = model_config.wandb_config.tags

        print(100 * "-")
        if detector_config is not None:
            print(f"Detector config: {detector_config}")
        print(f"Backbone config: {model_config.backbone_config}")
        print(f"Head config: {model_config.head_config}")
        print(f"Train Augmentation: {model_config.train_aug}")
        print(f"Inference Augmentation: {model_config.inference_aug}")

        shuffle_indices = create_shuffles(
            project, splits_file, trainset_index, model_config.net_type
        )
        if manual_shuffle_index:
            shuffle_indices = [manual_shuffle_index]
        shuffles_to_train = [shuffle_indices[i] for i in splits_to_train]
        print(f"training shuffles {shuffles_to_train}")
        for split_idx, shuffle_idx in zip(splits_to_train, shuffles_to_train):
            if wandb.run is not None:  # TODO: Finish wandb run in DLC
                wandb.finish()

            if detector_config is not None:
                print("  DetectorParameters")
                for k, v in asdict(detector_config).items():
                    print(f"    {k}: {v}")

            print("  ModelParameters")
            for k, v in asdict(model_config).items():
                print(f"    {k}: {v}")

            print("  Train kwargs")
            for k, v in model_config.train_kwargs().items():
                print(f"    {k}: {v}")

            if model_config.wandb_config is not None:
                i = project.iteration
                model_config.wandb_config.run_name = f"{run_name}-it{i}-shuf{shuffle_idx}"
                model_config.wandb_config.tags = (*tags, f"split={split_idx}")

            run_dlc(
                parameters=RunParameters(
                    shuffle=Shuffle(
                        project=project,
                        train_fraction=train_fraction,
                        index=shuffle_idx,
                        model_prefix="",
                    ),
                    train=train,
                    evaluate=evaluate,
                    device="cuda",
                    train_params=model_config,
                    detector_train_params=detector_config,
                    eval_params=eval_params,
                )
            )


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
RESNET_BACKBONE = BackboneConfig(
    model_name="resnet50_gn",
    output_stride=16,
    freeze_bn_stats=True,
    freeze_bn_weights=False,
)
HRNET_BACKBONE = BackboneConfig(  # output strides [4, 8, 16, 32]
    model_name="hrnet_w32",
    freeze_bn_stats=True,
    freeze_bn_weights=False,
)
HRNET_BACKBONE_INTER = BackboneConfig(  # output strides [4, 8, 16, 32]
    model_name="hrnet_w32",
    freeze_bn_stats=True,
    freeze_bn_weights=False,
    kwargs=dict(interpolate_branches=True),
)
HRNET_BACKBONE_INCRE = BackboneConfig(  # output strides [4, 8, 16, 32]
    model_name="hrnet_w32",
    freeze_bn_stats=True,
    freeze_bn_weights=False,
    kwargs=dict(increased_channel_count=True),
)

RESNET_OPTIMIZER = {"type": "AdamW", "params": {"lr": 1e-3}}
RESNET_SCHEDULER = {
    "type": "LRListScheduler",
    "params": {"lr_list": [[1e-4], [1e-5]], "milestones": [90, 120]},
}
DEFAULT_OPTIMIZER = {"type": "AdamW", "params": {"lr": 5e-4}}
DEFAULT_SCHEDULER = {
    "type": "LRListScheduler",
    "params": {"lr_list": [[1e-4], [1e-5]], "milestones": [90, 120]},
}


if __name__ == "__main__":
    # Project parameters
    PROJECT_NAME = "fly"
    PROJECT_BENCHMARKED = SA_DLC_BENCHMARKS[PROJECT_NAME]
    SPLIT_FILE = SA_DLC_DATA_ROOT / "saDLC_benchmarking_splits.json"
    CFG = PROJECT_BENCHMARKED.cfg
    NUM_BPT = len(get_bodyparts(CFG))

    # Train parameters
    EPOCHS = 150
    SAVE_EPOCHS = 25
    RESNET_BATCH_SIZE = 8
    HRNET_BATCH_SIZE = 8

    # logging params
    WANDB_PROJECT = "dlc3-benchmark-dev"
    BASE_TAGS = (f"project={PROJECT_NAME}", "server=m0")
    GROUP_UID = "base"

    # resize openfield
    if PROJECT_NAME == "openfield":
        AUG_TRAIN.resize = dict(height=640, width=640, keep_ratio=True)

    model_configs = [
        ModelConfig(
            net_type="resnet_50",
            batch_size=RESNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            dataloader_workers=2,
            dataloader_pin_memory=True,
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
            dataloader_workers=2,
            dataloader_pin_memory=True,
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
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32",
                group=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32",
                tags=(*BASE_TAGS, "arch=hrnet32", "ndeconv=1"),
            ),
        ),
        ModelConfig(
            net_type="hrnet_w32",
            batch_size=HRNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            dataloader_workers=2,
            dataloader_pin_memory=True,
            train_aug=AUG_TRAIN,
            inference_aug=None,
            backbone_config=HRNET_BACKBONE_INCRE,
            head_config=HeadConfig.build_plateau_head(
                c_in=128,
                c_out=NUM_BPT,
                deconv=[(NUM_BPT, 3, 2)],
                final_conv=False,
            ),
            optimizer_config=DEFAULT_OPTIMIZER,
            scheduler_config=DEFAULT_SCHEDULER,
            wandb_config=WandBConfig(
                project=WANDB_PROJECT,
                run_name=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32-incre",
                group=f"{PROJECT_NAME}-{GROUP_UID}-hrnet32-incre",
                tags=(*BASE_TAGS, "arch=hrnet32-incre", "ndeconv=1"),
            ),
        ),
        ModelConfig(
            net_type="hrnet_w32",
            batch_size=HRNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            dataloader_workers=2,
            dataloader_pin_memory=True,
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
    ]
    main(
        project=PROJECT_BENCHMARKED,
        splits_file=SPLIT_FILE,
        trainset_index=0,
        train_fraction=0.8,
        models_to_train=model_configs,
        splits_to_train=(0, 1, 2),
    )
