"""Code to make an ablation study with different image augmentation parameters"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import wandb

from deeplabcut.utils import get_bodyparts

from benchmark_train import EvalParameters, RunParameters, TrainParameters, run_dlc
from projects import SA_DLC_BENCHMARKS, SA_DLC_DATA_ROOT
from utils import Project, Shuffle, create_shuffles


@dataclass
class WandBConfig:
    project: str
    run_name: str

    def data(self) -> dict:
        return {
            "type": "WandbLogger",
            "project_name": self.project,
            "run_name": self.run_name,
        }


@dataclass
class BackboneConfig:
    """
    Attributes:
        model_name: one of "resnet50", "resnet50_gn"
        output_stride: 8, 16 or 32
        freeze_bn_weights:
        freeze_bn_stats:
    """
    model_name: str = "resnet50"
    output_stride: int | None = None
    freeze_bn_weights: bool | None = None
    freeze_bn_stats: bool | None = None
    drop_path_rate: float | None = None
    drop_block_rate: float | None = None

    def to_dict(self) -> dict:
        config = asdict(self)
        for k in list(config.keys()):
            if config[k] is None:
                config.pop(k)
        return config


@dataclass
class HeadConfig:
    plateau_targets: bool
    heatmap_config: dict
    locref_config: dict | None

    def to_dict(self) -> dict:
        output_channels = self.heatmap_config["channels"][-1]
        if self.heatmap_config.get("final_conv") is not None:
            output_channels = self.heatmap_config["final_conv"]["out_channels"]
        predictor = dict(
            type="HeatmapPredictor",
            location_refinement=self.locref_config is not None,
            locref_std=7.2801,
        )
        target_generator = dict(
            type="HeatmapPlateauGenerator" if self.plateau_targets else "HeatmapGaussianGenerator",
            num_heatmaps=output_channels,
            pos_dist_thresh=17,
            heatmap_mode="KEYPOINT",
            generate_locref=self.locref_config is not None,
            locref_std=7.2801,
        )
        criterion = dict(heatmap=dict(type="WeightedBCECriterion", weight=1.0))
        if self.locref_config is not None:
            criterion["locref"] = dict(
                type="WeightedHuberCriterion", weight=0.05
            )

        return dict(
            type="HeatmapHead",
            predictor=predictor,
            target_generator=target_generator,
            criterion=criterion,
            heatmap_config=self.heatmap_config,
            locref_config=self.locref_config,
        )


@dataclass
class ImageAugmentations:
    """
    The default augmentation only normalizes images.

    Examples:
        gaussian_noise: 12.75
        resize: {height: 800, width: 800, keep_ratio: true}
        rotation: 30
        scale_jitter: (0.5, 1.25)
        translation: 40
    """
    normalize: bool = True
    covering: bool = False
    gaussian_noise: float = 0.0
    hist_eq: bool = False
    motion_blur: bool = False
    resize: dict | None = None
    rotation: int = 0
    scale_jitter: tuple[float, float] | None = None
    translation: int = 0

    def data(self) -> dict:
        augmentations = {
            "normalize_images": self.normalize,
            "covering": self.covering,
            "gaussian_noise": self.gaussian_noise,
            "hist_eq": self.hist_eq,
            "motion_blur": self.motion_blur,
            "rotation": self.rotation,
            "scale_jitter": False,
            "translation": self.translation,
        }
        if self.resize:
            augmentations["resize"] = self.resize
        if self.scale_jitter:
            augmentations["scale_jitter"] = self.scale_jitter
        return augmentations


@dataclass
class ModelConfig(TrainParameters):
    net_type: str = "resnet_50"
    augmentations: ImageAugmentations | None = None
    backbone_config: BackboneConfig | None = None
    head_config: HeadConfig | None = None
    optimizer_config: dict | None = None
    scheduler_config: dict | None = None
    wandb_config: WandBConfig | None = None

    def train_kwargs(self) -> dict:
        kwargs = super().train_kwargs()
        if self.augmentations is not None:
            kwargs["data"] = self.augmentations.data()
        if self.backbone_config is not None:
            kwargs["model"] = dict(backbone=self.backbone_config.to_dict())
        if self.head_config is not None:
            model_config = kwargs.get("model", {})
            model_config["heads"] = dict(bodypart=self.head_config.to_dict())
            kwargs["model"] = model_config
        if self.wandb_config is not None:
            kwargs["logger"] = self.wandb_config.data()
        if self.optimizer_config is not None:
            kwargs["optimizer"] = self.optimizer_config
        if self.scheduler_config is not None:
            kwargs["scheduler"] = self.scheduler_config
        return kwargs


def main(
    project: Project,
    splits_file: Path,
    trainset_index: int,
    train_fraction: float,
    models_to_train: list[ModelConfig],
    splits_to_train: tuple[int, ...] = (0, 1, 2),
):
    project.update_iteration_in_config()
    for config in models_to_train:
        if wandb.run is not None:  # TODO: Finish wandb run in DLC
            wandb.finish()

        print(100 * "-")
        print(f"Backbone config: {config.backbone_config}")
        print(f"Head config: {config.head_config}")
        print(f"Augmentation: {config.augmentations}")

        shuffle_indices = create_shuffles(
            project, splits_file, trainset_index, config.net_type
        )
        shuffles_to_train = [shuffle_indices[i] for i in splits_to_train]
        print(f"training shuffles {shuffles_to_train}")
        for shuffle_idx in shuffles_to_train:
            if wandb.run is not None:  # TODO: Finish wandb run in DLC
                wandb.finish()

            print("  ModelParameters")
            for k, v in asdict(config).items():
                print(f"    {k}: {v}")
            print("  Train kwargs")
            for k, v in config.train_kwargs().items():
                print(f"    {k}: {v}")

            if config.wandb_config is not None:
                config.wandb_config.run_name += f"-it{project.iteration}-shuf{shuffle_idx}"

            run_dlc(
                parameters=RunParameters(
                    shuffle=Shuffle(
                        project=project,
                        train_fraction=train_fraction,
                        index=shuffle_idx,
                        model_prefix="",
                    ),
                    train=True,
                    evaluate=True,
                    device="cuda:0",
                    train_params=config,
                    eval_params=EvalParameters(snapshotindex="all", plotting=False)
                )
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
    project_benchmarked = SA_DLC_BENCHMARKS["fly"]
    splits_file = (SA_DLC_DATA_ROOT / "saDLC_benchmarking_splits.json")
    cfg = project_benchmarked.cfg
    num_bodyparts = len(get_bodyparts(cfg))

    FULL_AUG = ImageAugmentations(
        covering=True,
        gaussian_noise=12.75,
        hist_eq=True,
        motion_blur=True,
        rotation=30,
        scale_jitter=(0.5, 1.25),
        translation=40,
    )
    model_configs = [
        ModelConfig(
            net_type="resnet_50",
            batch_size=8,
            epochs=125,
            save_epochs=25,
            augmentations=FULL_AUG,
            backbone_config=BackboneConfig(
                model_name="resnet50_gn",
                freeze_bn_stats=True,
                freeze_bn_weights=False,
            ),
            head_config=HeadConfig(
                plateau_targets=True,
                heatmap_config=dict(
                    channels=[2048, num_bodyparts],
                    kernel_size=[3],
                    strides=[2],
                    final_conv=None,
                ),
                locref_config=dict(
                    channels=[2048, 2 * num_bodyparts],
                    kernel_size=[3],
                    strides=[2],
                    final_conv=None,
                ),
            ),
            optimizer_config=RESNET_OPTIMIZER,
            scheduler_config=RESNET_SCHEDULER,
            wandb_config=WandBConfig(project="dlc3_hrnet", run_name="resnet_single_deconv"),
        ),
        ModelConfig(
            net_type="hrnet_w32",
            batch_size=8,
            epochs=125,
            save_epochs=25,
            augmentations=FULL_AUG,
            backbone_config=BackboneConfig(
                model_name="hrnet_w32",
                freeze_bn_stats=True,
                freeze_bn_weights=False,
            ),
            head_config=HeadConfig(
                plateau_targets=False,
                heatmap_config=dict(
                    channels=[32],
                    kernel_size=[],
                    strides=[],
                    final_conv=dict(out_channels=num_bodyparts, kernel_size=1),
                ),
                locref_config=None
            ),
            optimizer_config=DEFAULT_OPTIMIZER,
            scheduler_config=DEFAULT_SCHEDULER,
            wandb_config=WandBConfig(project="dlc3_hrnet", run_name="hrnet_gauss"),
        ),
    ]
    main(
        project=project_benchmarked,
        splits_file=splits_file,
        trainset_index=0,
        train_fraction=0.8,
        models_to_train=model_configs,
        splits_to_train=(0, 1, 2),
    )
