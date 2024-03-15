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
from __future__ import annotations

import argparse
import copy
import logging

import albumentations as A
from torch.utils.data import DataLoader

import deeplabcut.pose_estimation_pytorch.config as torch_config
import deeplabcut.pose_estimation_pytorch.utils as utils
from deeplabcut.pose_estimation_pytorch.data import build_transforms, DLCLoader, Loader
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.runners import build_training_runner
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.runners.logger import (
    LOGGER,
    destroy_file_logging,
    setup_file_logging,
)


def train(
    loader: Loader,
    run_config: dict,
    task: Task,
    device: str = "cpu",
    logger_config: dict | None = None,
    snapshot_path: str | None = None,
    transform: A.BaseCompose | None = None,
    inference_transform: A.BaseCompose | None = None,
    max_snapshots_to_keep: int | None = None,
) -> None:
    """Builds a model from a configuration and fits it to a dataset

    Args:
        loader: the loader containing the data to train on/validate with
        run_config: the model and run configuration
        task: the task to train the model for
        device: the torch device to train on (such as "cpu", "cuda", "mps")
        logger_config: the configuration of a logger to use
        snapshot_path: if continuing to train from a snapshot, the path containing the
            weights to load
        transform: if defined, overwrites the transform defined in the model config
        inference_transform: if defined, overwrites the inference transform defined in
            the model config
        max_snapshots_to_keep: the maximum number of snapshots to store for each model
    """
    if task == Task.DETECT:
        model = DETECTORS.build(run_config["model"])
    else:
        model = PoseModel.build(run_config["model"])

    if max_snapshots_to_keep is not None:
        run_config["snapshots"]["max_snapshots"] = max_snapshots_to_keep

    logger = None
    if logger_config is not None:
        logger = LOGGER.build(dict(**logger_config, model=model))
        logger.log_config(run_config)

    if device is None:
        device = utils.resolve_device(run_config)

    model.to(device)  # Move model before giving its parameters to the optimizer
    runner = build_training_runner(
        runner_config=run_config["runner"],
        model_folder=loader.model_folder,
        task=task,
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        logger=logger,
    )

    if transform is None:
        transform = build_transforms(run_config["data"]["train"])
    if inference_transform is None:
        inference_transform = build_transforms(run_config["data"]["inference"])

    logging.info("Data Transforms:")
    logging.info(f"  Training:   {transform}")
    logging.info(f"  Validation: {inference_transform}")

    train_dataset = loader.create_dataset(transform=transform, mode="train", task=task)
    valid_dataset = loader.create_dataset(
        transform=inference_transform, mode="test", task=task
    )
    logging.info(
        f"Using {len(train_dataset)} images to train {task} and {len(valid_dataset)}"
        f" for testing"
    )

    batch_size = run_config["train_settings"]["batch_size"]
    num_workers = run_config["train_settings"]["dataloader_workers"]
    pin_memory = run_config["train_settings"]["dataloader_pin_memory"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    runner.fit(
        train_dataloader,
        valid_dataloader,
        epochs=run_config["train_settings"]["epochs"],
        display_iters=run_config["train_settings"]["display_iters"],
    )


def train_network(
    config: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    device: str | None = None,
    snapshot_path: str | None = None,
    detector_path: str | None = None,
    max_snapshots_to_keep: int | None = None,
    **kwargs,
) -> None:
    """Trains a network for a project

    Args:
        config : path to the yaml config file of the project
        shuffle : index of the shuffle we want to train on
        trainingsetindex : training set index
        modelprefix: directory containing the deeplabcut configuration files to use
            to train the network (and where snapshots will be saved). By default, they
             are assumed to exist in the project folder.
        device: the torch device to train on (such as "cpu", "cuda", "mps")
        snapshot_path: if resuming training, used to specify the snapshot from which to resume
        detector_path: if resuming training of a top-down model, used to specify the
            detector snapshot from which to resume
        max_snapshots_to_keep: the maximum number of snapshots to save for each model
        **kwargs : could be any entry of the pytorch_config dictionary. Examples are
            to see the full list see the pytorch_cfg.yaml file in your project folder
    """
    loader = DLCLoader(
        config=config,
        shuffle=shuffle,
        trainset_index=trainingsetindex,
        modelprefix=modelprefix,
    )
    loader.update_model_cfg(kwargs)
    setup_file_logging(loader.model_folder / "train.txt")

    logging.info("Training with configuration:")
    torch_config.pretty_print(loader.model_cfg, print_fn=logging.info)

    # fix seed for reproducibility
    utils.fix_seeds(loader.model_cfg["train_settings"]["seed"])

    # get the pose task
    pose_task = Task(loader.model_cfg.get("method", "bu"))
    if (
        pose_task == Task.TOP_DOWN
        and loader.model_cfg["detector"]["train_settings"]["epochs"] > 0
    ):
        logger_config = None
        if loader.model_cfg.get("logger"):
            logger_config = copy.deepcopy(loader.model_cfg["logger"])
            logger_config["run_name"] += "-detector"
        train(
            loader=loader,
            run_config=loader.model_cfg["detector"],
            task=Task.DETECT,
            device=device,
            logger_config=logger_config,
            snapshot_path=detector_path,
            max_snapshots_to_keep=max_snapshots_to_keep,
        )

    train(
        loader=loader,
        run_config=loader.model_cfg,
        task=pose_task,
        device=device,
        logger_config=loader.model_cfg.get("logger"),
        snapshot_path=snapshot_path,
        max_snapshots_to_keep=max_snapshots_to_keep,
    )

    destroy_file_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--train-ind", type=int, default=0)
    parser.add_argument("--modelprefix", type=str, default="")
    args = parser.parse_args()
    train_network(
        config=args.config_path,
        shuffle=args.shuffle,
        trainingsetindex=args.train_ind,
        modelprefix=args.modelprefix,
    )
