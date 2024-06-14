from __future__ import annotations

from dataclasses import dataclass, asdict

from benchmark_train import TrainParameters
from utils import WandBConfig
from utils_augmentation import ImageAugmentations


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
class DetectorConfig(TrainParameters):
    train_aug: ImageAugmentations | None = None
    inference_aug: ImageAugmentations | None = None
    optimizer_config: dict | None = None
    scheduler_config: dict | None = None

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
