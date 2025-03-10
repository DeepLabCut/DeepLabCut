"""Testscript for single animal PyTorch projects"""

from __future__ import annotations

from pathlib import Path

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.compat import Engine

from utils import (
    cleanup,
    copy_project_for_test,
    create_fake_project,
    log_step,
    run,
    SyntheticProjectParameters,
)


def main(
    synthetic_data: bool,
    net_types: list[str],
    epochs: int = 1,
    save_epochs: int = 1,
    max_snapshots_to_keep: int = 5,
    batch_size: int = 1,
    device: str = "cpu",
    logger: dict | None = None,
    synthetic_data_params: SyntheticProjectParameters = SyntheticProjectParameters(
        multianimal=False,
        num_bodyparts=6,
    ),
    create_labeled_videos: bool = False,
    delete_after_test_run: bool = False,
) -> None:
    engine = Engine.PYTORCH
    if synthetic_data:
        project_path = Path("synthetic-data-niels-single-animal").resolve()
        videos = [str(project_path / "videos" / "video.mp4")]
        create_fake_project(path=project_path, params=synthetic_data_params)

    else:
        project_path = copy_project_for_test()
        videos = [str(project_path / "videos" / "m3v1mp4.mp4")]

    config_path = project_path / "config.yaml"
    cfg = af.read_config(config_path)
    trainset_index = 0
    train_frac = cfg["TrainingFraction"][trainset_index]
    try:
        for net_type in net_types:
            try:
                run(
                    config_path=config_path,
                    train_fraction=train_frac,
                    trainset_index=trainset_index,
                    net_type=net_type,
                    videos=videos,
                    device=device,
                    engine=engine,
                    pytorch_cfg_updates={
                        "train_settings.display_iters": 50,
                        "train_settings.epochs": epochs,
                        "train_settings.batch_size": batch_size,
                        "runner.device": device,
                        "runner.snapshots.save_epochs": save_epochs,
                        "runner.snapshots.max_snapshots": max_snapshots_to_keep,
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
        synthetic_data=True,
        net_types=["cspnext_m", "resnet_50", "hrnet_w32"],
        batch_size=4,
        epochs=8,
        save_epochs=2,
        max_snapshots_to_keep=2,
        device="cpu",  # "cpu", "cuda:0", "mps"
        logger=None,
        synthetic_data_params=SyntheticProjectParameters(
            multianimal=False,
            num_bodyparts=4,
            num_individuals=1,
            num_unique=0,
            num_frames=12,
            frame_shape=(128, 128),
        ),
        create_labeled_videos=True,
        delete_after_test_run=True,
    )
