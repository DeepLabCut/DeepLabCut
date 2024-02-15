"""Code to make an ablation study with different image augmentation parameters"""
from __future__ import annotations

from pathlib import Path

from deeplabcut.utils import get_bodyparts

from benchmark_run_experiments import (
    main,
    BackboneConfig,
    HeadConfig,
    ImageAugmentations,
    ModelConfig,
    WandBConfig,
    DEFAULT_OPTIMIZER,
    DEFAULT_SCHEDULER,
    RESNET_OPTIMIZER,
    RESNET_SCHEDULER,
)
from utils import Project


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
        iteration=2,  # ITERATION 0 IS THE PAPER DATA
    ),
}


if __name__ == "__main__":
    project_benchmarked = LP_DLC_BENCHMARKS["mirrorFish"]
    splits_file = (LP_DLC_DATA_ROOT / "lightning_pose_splits.json")
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
                output_stride=16,
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
            wandb_config=WandBConfig(project="dlc3-mirror-mouse-dev", run_name="resnet_single_deconv"),
        )
    ]

    main(
        project=project_benchmarked,
        splits_file=splits_file,
        trainset_index=0,
        train_fraction=0.81,
        models_to_train=model_configs,
        splits_to_train=(0, 1, 2, 3, 4),
    )
