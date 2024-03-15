"""Code to make an ablation study with different image augmentation parameters"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import wandb
from deeplabcut.utils import get_bodyparts

from benchmark_train import EvalParameters, run_dlc, RunParameters, TrainParameters
from projects import SA_DLC_BENCHMARKS, SA_DLC_DATA_ROOT
from utils import create_shuffles, Project, Shuffle


@dataclass
class WandBConfig:
    project: str
    run_name: str
    save_code: bool = True
    tags: tuple[str, ...] | None = None
    group: str | None = None

    def data(self) -> dict:
        return dict(
            type="WandbLogger",
            project_name=self.project,
            run_name=self.run_name,
            save_code=self.save_code,
            tags=self.tags,
            group=self.group,
        )


@dataclass
class BackboneConfig:
    """
    Attributes:
        model_name: the timm model name ("resnet50", "resnet50_gn", "hrnet_w18", ...)
        output_stride: 8, 16 or 32 (HRNet only supports 32)
        freeze_bn_weights: freeze batch norm weights
        freeze_bn_stats: freeze batch norm stats
        kwargs: any keyword-arguments for the backbone type that was selected, e.g.
            HRNet: ``only_high_res: bool`` only use the high-resolution branch as the
                image features (otherwise, in DEKR style all branches are interpolated
                to the same shape and concatenated).
    """

    model_name: str = "resnet50"
    output_stride: int | None = None
    freeze_bn_weights: bool | None = None
    freeze_bn_stats: bool | None = None
    drop_path_rate: float | None = None
    drop_block_rate: float | None = None
    kwargs: dict | None = None

    def to_dict(self) -> dict:
        config = asdict(self)
        config.pop("kwargs")
        for k in list(config.keys()):
            if config[k] is None:
                config.pop(k)
        if self.kwargs is not None:
            for k, v in self.kwargs.items():
                config[k] = v
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
            type=(
                "HeatmapPlateauGenerator"
                if self.plateau_targets
                else "HeatmapGaussianGenerator"
            ),
            num_heatmaps=output_channels,
            pos_dist_thresh=17,
            heatmap_mode="KEYPOINT",
            generate_locref=self.locref_config is not None,
            locref_std=7.2801,
        )
        criterion = dict(heatmap=dict(type="WeightedBCECriterion", weight=1.0))
        if self.locref_config is not None:
            criterion["locref"] = dict(type="WeightedHuberCriterion", weight=0.05)

        return dict(
            type="HeatmapHead",
            predictor=predictor,
            target_generator=target_generator,
            criterion=criterion,
            heatmap_config=self.heatmap_config,
            locref_config=self.locref_config,
        )

    @staticmethod
    def build_plateau_head(
        c_in: int,
        c_out: int,
        deconv: list[tuple[int, int, int]],  # channel, kernel, stride
        final_conv: bool = False,
    ) -> HeadConfig:
        heatmap = dict(channels=[c_in], kernel_size=[], strides=[], final_conv=None)
        locref = dict(channels=[c_in], kernel_size=[], strides=[], final_conv=None)
        for c, k, s in deconv:
            for config in (heatmap, locref):
                config["channels"].append(c)
                config["kernel_size"].append(k)
                config["strides"].append(s)

        if final_conv:
            heatmap["final_conv"] = dict(out_channels=c_out, kernel_size=1)
            locref["final_conv"] = dict(out_channels=2 * c_out, kernel_size=1)
        else:
            assert deconv[-1][0] == c_out
            locref["channels"][-1] = 2 * c_out

        return HeadConfig(
            plateau_targets=True,
            heatmap_config=heatmap,
            locref_config=locref,
        )


@dataclass
class AffineAugmentation:
    """An affine image augmentation"""

    p: float = 0.9
    rotation: int = 0
    scale: tuple[float, float] | None = None
    translation: int = 0

    def data(self) -> dict:
        affine = {}
        if self.p > 0:
            affine["p"] = self.p
        if self.scale is not None:
            affine["scaling"] = self.scale
        if self.rotation > 0:
            affine["rotation"] = self.rotation
        if self.translation > 0:
            affine["translation"] = self.translation
        return affine


@dataclass
class CropSampling:
    """Random crop around keypoints"""
    width: int
    height: int
    max_shift: float = 0.4
    mode: str = "uniform"  # "uniform", "keypoints", "density", "hybrid"

    def __post_init__(self):
        assert self.mode in ("uniform", "keypoints", "density", "hybrid")
        assert 0 <= self.max_shift <= 1

    def data(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "max_shift": self.max_shift,
            "mode": self.mode,
        }


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
    affine: AffineAugmentation | None = None
    covering: bool = False
    gaussian_noise: float | bool = False
    hist_eq: bool = False
    motion_blur: bool = False
    resize: dict | None = None
    crop_sampling: CropSampling | None = None

    def data(self) -> dict:
        augmentations = {
            "normalize_images": self.normalize,
            "covering": self.covering,
            "gaussian_noise": self.gaussian_noise,
            "hist_eq": self.hist_eq,
            "motion_blur": self.motion_blur,
        }
        if self.affine is not None:
            augmentations["affine"] = self.affine.data()
        if self.resize is not None:
            augmentations["resize"] = self.resize
        if self.crop_sampling is not None:
            augmentations["crop_sampling"] = self.crop_sampling.data()
        return augmentations


@dataclass
class DetectorConfig(TrainParameters):
    train_aug: ImageAugmentations | None = None
    inference_aug: ImageAugmentations | None = None
    optimizer_config: dict | None = None
    scheduler_config: dict | None = None

    def train_kwargs(self) -> dict:
        kwargs = super().train_kwargs()
        if self.train_aug is not None:
            kwargs["data"]["train"] = self.train_aug.data()
        if self.inference_aug is not None:
            kwargs["data"]["inference"] = self.inference_aug.data()
        if self.optimizer_config is not None:
            kwargs["runner"]["optimizer"] = self.optimizer_config
        if self.scheduler_config is not None:
            kwargs["runner"]["scheduler"] = self.scheduler_config
        return kwargs


@dataclass
class ModelConfig(TrainParameters):
    net_type: str = "resnet_50"
    train_aug: ImageAugmentations | None = None
    inference_aug: ImageAugmentations | None = None
    backbone_config: BackboneConfig | None = None
    head_config: HeadConfig | None = None
    optimizer_config: dict | None = None
    scheduler_config: dict | None = None
    wandb_config: WandBConfig | None = None

    def train_kwargs(self) -> dict:
        kwargs = super().train_kwargs()
        if self.train_aug is not None:
            data = kwargs.get("data", {})
            data["train"] = self.train_aug.data()
            kwargs["data"] = data
        if self.inference_aug is not None:
            data = kwargs.get("data", {})
            data["inference"] = self.inference_aug.data()
            kwargs["data"] = data
        if self.backbone_config is not None:
            kwargs["model"] = dict(backbone=self.backbone_config.to_dict())
        if self.head_config is not None:
            model_config = kwargs.get("model", {})
            model_config["heads"] = dict(bodypart=self.head_config.to_dict())
            kwargs["model"] = model_config
        if self.wandb_config is not None:
            kwargs["logger"] = self.wandb_config.data()
        if self.optimizer_config is not None:
            runner = kwargs.get("runner", {})
            runner["optimizer"] = self.optimizer_config
            kwargs["runner"] = runner
        if self.scheduler_config is not None:
            runner = kwargs.get("runner", {})
            runner["scheduler"] = self.scheduler_config
            kwargs["runner"] = runner
        return kwargs


def main(
    project: Project,
    splits_file: Path,
    trainset_index: int,
    train_fraction: float,
    models_to_train: list[ModelConfig | tuple[DetectorConfig, ModelConfig]],
    splits_to_train: tuple[int, ...] = (0, 1, 2),
    eval_params: EvalParameters | None = None,
):
    if eval_params is None:
        eval_params = EvalParameters(snapshotindex="all", plotting=False)

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
                    train=True,
                    evaluate=True,
                    device="cuda:0",
                    train_params=model_config,
                    detector_train_params=detector_config,
                    eval_params=eval_params,
                )
            )


AUG_INFERENCE = ImageAugmentations(normalize=True)
AUG_TRAIN = ImageAugmentations(
    normalize=True,
    covering=True,
    gaussian_noise=12.75,
    hist_eq=True,
    motion_blur=True,
    affine=AffineAugmentation(
        p=0.9,
        rotation=30,
        scale=(0.5, 1.25),
        translation=40,
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
    "params": {"lr_list": [[1e-4], [1e-5]], "milestones": [160, 190]},
}
DEFAULT_OPTIMIZER = {"type": "AdamW", "params": {"lr": 5e-4}}
DEFAULT_SCHEDULER = {
    "type": "LRListScheduler",
    "params": {"lr_list": [[1e-4], [1e-5]], "milestones": [160, 190]},
}


if __name__ == "__main__":
    # Project parameters
    PROJECT_NAME = "fly"
    PROJECT_BENCHMARKED = SA_DLC_BENCHMARKS[PROJECT_NAME]
    SPLIT_FILE = SA_DLC_DATA_ROOT / "saDLC_benchmarking_splits.json"
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

    # resize openfield
    if PROJECT_NAME == "openfield":
        AUG_TRAIN.resize = dict(height=640, width=640, keep_ratio=True)

    model_configs = [
        ModelConfig(
            net_type="resnet_50",
            batch_size=RESNET_BATCH_SIZE,
            epochs=EPOCHS,
            save_epochs=SAVE_EPOCHS,
            train_aug=AUG_TRAIN,
            inference_aug=AUG_INFERENCE,
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
            inference_aug=AUG_INFERENCE,
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
            train_aug=AUG_TRAIN,
            inference_aug=AUG_INFERENCE,
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
            train_aug=AUG_TRAIN,
            inference_aug=AUG_INFERENCE,
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
