"""File to train a model on a COCO dataset"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from deeplabcut.pose_estimation_pytorch import COCOLoader
from deeplabcut.pose_estimation_pytorch.apis.train import train
from deeplabcut.pose_estimation_pytorch.runners import Task
from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging


def main(
    project_root: str,
    train_file: str,
    test_file: str,
    pytorch_config: str,
    device: str | None,
    epochs: int | None,
    save_epochs: int | None,
    detector_epochs: int | None,
    detector_save_epochs: int | None,
    snapshot_path: str | None,
    detector_path: str | None,
):
    model_folder = Path(pytorch_config).parent.parent
    log_path = Path(pytorch_config).parent / "log.txt"
    setup_file_logging(log_path)

    loader = COCOLoader(
        project_root=project_root,
        model_config_path=pytorch_config,
        train_json_filename=train_file,
        test_json_filename=test_file,
    )
    pytorch_config = loader.model_cfg
    if device is not None:
        pytorch_config["device"] = device

    if epochs is not None:
        pytorch_config["epochs"] = epochs
    if save_epochs is not None:
        pytorch_config["save_epochs"] = save_epochs

    pose_task = Task(pytorch_config.get("method", "bu"))
    if pytorch_config.get("method", "bu").lower() == "td":
        logger_config = None
        if pytorch_config.get("logger"):
            logger_config = copy.deepcopy(pytorch_config["logger"])
            logger_config["run_name"] += "-detector"

        if detector_epochs is not None:
            pytorch_config["detector"]["epochs"] = detector_epochs
        if detector_save_epochs is not None:
            pytorch_config["detector"]["save_epochs"] = detector_save_epochs

        if detector_epochs > 0:
            train(
                loader=loader,
                model_folder=str(model_folder),
                run_config=pytorch_config["detector"],
                task=Task.DETECT,
                device=pytorch_config["device"],
                transform_config=pytorch_config["data_detector"],
                logger_config=logger_config,
                snapshot_path=detector_path,
                transform=None,  # Load transform from config
            )

    train(
        loader=loader,
        model_folder=str(model_folder),
        run_config=pytorch_config,
        task=pose_task,
        device=pytorch_config["device"],
        transform_config=pytorch_config["data"],
        logger_config=pytorch_config.get("logger"),
        snapshot_path=snapshot_path,
        transform=None,  # Load transform from config
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
