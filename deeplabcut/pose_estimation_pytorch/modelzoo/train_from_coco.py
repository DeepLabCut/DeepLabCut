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
"""File to train a model on a COCO dataset"""

from __future__ import annotations

import copy
from pathlib import Path

from deeplabcut.pose_estimation_pytorch import COCOLoader, utils
from deeplabcut.pose_estimation_pytorch.apis.train import train
from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging
from deeplabcut.pose_estimation_pytorch.task import Task


def adaptation_train(
    project_root: str | Path,
    model_folder: str | Path,
    train_file: str,
    test_file: str,
    model_config_path: str | Path,
    device: str | None,
    epochs: int | None,
    save_epochs: int | None,
    detector_epochs: int | None,
    detector_save_epochs: int | None,
    snapshot_path: str | None,
    detector_path: str | None,
    batch_size: int = 8,
    detector_batch_size: int = 8,
    eval_interval: int | None = None,
):
    setup_file_logging(Path(model_folder) / "log.txt")
    loader = COCOLoader(
        project_root=project_root,
        model_config_path=model_config_path,
        train_json_filename=train_file,
        test_json_filename=test_file,
    )

    utils.fix_seeds(loader.model_cfg["train_settings"]["seed"])

    updates = dict(
        detector=dict(
            model=dict(freeze_bn_stats=True),
            runner=dict(snapshots=dict(max_snapshots=5, save_epochs=1)),
            train_settings=dict(batch_size=detector_batch_size, epochs=4),
        ),
        model=dict(backbone=dict(freeze_bn_stats=True)),
        runner=dict(snapshots=dict(max_snapshots=5, save_epochs=1)),
        train_settings=dict(batch_size=batch_size, epochs=4),
    )

    if epochs is not None:
        updates["train_settings"]["epochs"] = epochs
    if save_epochs is not None:
        updates["runner"]["snapshots"]["save_epochs"] = save_epochs
    if detector_epochs is not None:
        updates["detector"]["train_settings"]["epochs"] = detector_epochs
    if detector_save_epochs is not None:
        updates["detector"]["runner"]["snapshots"]["save_epochs"] = detector_save_epochs

    if eval_interval is not None:
        updates["runner"]["eval_interval"] = eval_interval

    loader.update_model_cfg(updates)

    pose_task = Task(loader.model_cfg["method"])
    if pose_task == Task.TOP_DOWN:
        logger_config = None
        if loader.model_cfg.get("logger"):
            logger_config = copy.deepcopy(loader.model_cfg["logger"])
            logger_config["run_name"] += "-detector"

        if loader.model_cfg["detector"]["train_settings"]["epochs"] > 0:
            train(
                loader=loader,
                run_config=loader.model_cfg["detector"],
                task=Task.DETECT,
                device=device,
                logger_config=logger_config,
                snapshot_path=detector_path,
            )

    train(
        loader=loader,
        run_config=loader.model_cfg,
        task=pose_task,
        device=device,
        logger_config=loader.model_cfg.get("logger"),
        snapshot_path=snapshot_path,
    )
