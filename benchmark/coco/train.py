"""File to train a model on a COCO dataset"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from deeplabcut.pose_estimation_pytorch import COCOLoader, utils
from deeplabcut.pose_estimation_pytorch.apis.train import train
from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging
from deeplabcut.pose_estimation_pytorch.task import Task
from collections import defaultdict

def main(
    project_root: str,
    train_file: str,
    test_file: str,
    model_config_path: str,
    device: str | None,
    epochs: int | None,
    save_epochs: int | None,
    detector_epochs: int | None,
    detector_save_epochs: int | None,
    snapshot_path: str | None,
    detector_path: str | None,
    batch_size: int = 64,
    dataloader_workers: int = 12,
    detector_batch_size: int = 64,
    detector_dataloader_workers: int = 12,
):
    log_path = Path(model_config_path).parent / "log.txt"
    setup_file_logging(log_path)

    loader = COCOLoader(
        project_root=project_root,
        model_config_path=model_config_path,
        train_json_filename=train_file,
        test_json_filename=test_file,
    )
    
    utils.fix_seeds(loader.model_cfg["train_settings"]["seed"])

    if epochs is None:
        epochs = loader.model_cfg["train_settings"]["epochs"]
    if save_epochs is None:
        save_epochs = loader.model_cfg["runner"]["snapshots"]["save_epochs"]

    updates = dict(
        runner=dict(snapshots=dict(save_epochs=save_epochs)),
        train_settings=dict(
            batch_size=batch_size,
            dataloader_workers=dataloader_workers,
            epochs=epochs,
        ),
    )

    det_cfg = loader.model_cfg("detector")
    if det_cfg is not None:
        if detector_epochs is None:
            detector_epochs = det_cfg["train_settings"]["epochs"]
        if detector_save_epochs is None:
            detector_save_epochs = det_cfg["runner"]["snapshots"]["save_epochs"]
        updates_detector = dict(
            runner=dict(snapshots=dict(save_epochs=detector_save_epochs)),
            train_settings=dict(
                batch_size=detector_batch_size,
                dataloader_workers=detector_dataloader_workers,
            ),
        )
        updates["detector"] = updates_detector

    loader.update_model_cfg(updates)

    pose_task = Task(loader.model_cfg["method"])
    if pose_task == Task.TOP_DOWN:
        logger_config = None
        if loader.model_cfg.get("logger"):
            logger_config = copy.deepcopy(loader.model_cfg["logger"])
            logger_config["run_name"] += "-detector"

        # skipping detector training if a detector_path is given 
        if args.detector_path is None and detector_epochs > 0:
            train(
                loader=loader,
                run_config=loader.model_cfg["detector"],
                task=Task.DETECT,
                device=device,
                logger_config=logger_config,
                snapshot_path=detector_path,
            )

    if epochs > 0:
        train(
            loader=loader,
            run_config=loader.model_cfg,
            task=pose_task,
            device=device,
            logger_config=loader.model_cfg.get("logger"),
            snapshot_path=snapshot_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("pytorch_config")
    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-epochs", type=int, default=None)
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--detector-save-epochs", type=int, default=None)
    parser.add_argument("--snapshot_path", default=None)
    parser.add_argument("--detector_path", default=None)
    args = parser.parse_args()
    main(
        args.project_root,
        args.train_file,
        args.test_file,
        args.pytorch_config,
        args.device,
        args.epochs,
        args.save_epochs,
        args.detector_epochs,
        args.detector_save_epochs,
        args.snapshot_path,
        args.detector_path,
    )
