#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Testscript for single animal PyTorch projects"""
from __future__ import annotations

from pathlib import Path

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.compat import Engine
from deeplabcut.pose_estimation_pytorch.config.utils import is_model_top_down

from utils import (
    cleanup,
    create_fake_project,
    log_step,
    run,
    SyntheticProjectParameters,
)


def main(
    net_types: list[str],
    params: SyntheticProjectParameters,
    epochs: int = 1,
    top_down_epochs: int = 1,
    detector_epochs: int = 1,
    save_epochs: int = 1,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    max_snapshots_to_keep: int = 5,
    device: str = "cpu",
    logger: dict | None = None,
    create_labeled_videos: bool = False,
    delete_after_test_run: bool = False,
) -> None:
    project_path = Path("synthetic-data-niels-multi-animal").resolve()
    config_path = project_path / "config.yaml"
    create_fake_project(path=project_path, params=params)

    engine = Engine.PYTORCH
    cfg = af.read_config(config_path)
    trainset_index = 0
    train_frac = cfg["TrainingFraction"][trainset_index]
    try:
        for net_type in net_types:
            epochs_ = epochs
            if is_model_top_down(net_type):
                epochs_ = top_down_epochs
            try:
                run(
                    config_path=config_path,
                    train_fraction=train_frac,
                    trainset_index=trainset_index,
                    net_type=net_type,
                    videos=[str(project_path / "videos" / "video.mp4")],
                    device=device,
                    engine=engine,
                    pytorch_cfg_updates={
                        "train_settings.display_iters": 50,
                        "train_settings.epochs": epochs_,
                        "train_settings.batch_size": batch_size,
                        "runner.device": device,
                        "runner.snapshots.save_epochs": save_epochs,
                        "runner.snapshots.max_snapshots": max_snapshots_to_keep,
                        "detector.train_settings.display_iters": 1,
                        "detector.train_settings.epochs": detector_epochs,
                        "detector.train_settings.batch_size": detector_batch_size,
                        "detector.runner.snapshots.save_epochs": save_epochs,
                        "detector.runner.snapshots.max_snapshots": max_snapshots_to_keep,
                        "logger": logger,
                    },
                    create_labeled_videos=create_labeled_videos,
                )
            except Exception as err:
                log_step(f"FAILED TO RUN {net_type}")
                log_step(str(err))
                log_step("Continuing to next model")
                raise err

    finally:
        if delete_after_test_run:
            cleanup(project_path)


if __name__ == "__main__":
    wandb_logger = {
        "type": "WandbLogger",
        "project_name": "testscript-dev",
        "run_name": "test-logging",
    }
    main(
        net_types=["top_down_resnet_50", "resnet_50", "dekr_w32", "rtmpose_m"],
        params=SyntheticProjectParameters(
            multianimal=True,
            num_bodyparts=4,
            num_individuals=3,
            num_unique=0,
            num_frames=25,
            frame_shape=(256, 256),
        ),
        batch_size=2,
        detector_batch_size=2,
        epochs=8,
        top_down_epochs=2,
        detector_epochs=10,
        save_epochs=4,
        max_snapshots_to_keep=2,
        device="cpu",  # "cpu", "cuda:0", "mps"
        logger=None,
        create_labeled_videos=True,
        delete_after_test_run=True,
    )
