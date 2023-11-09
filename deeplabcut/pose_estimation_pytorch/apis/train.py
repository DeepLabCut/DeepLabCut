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
from pathlib import Path

import albumentations as A
from torch.utils.data import DataLoader

import deeplabcut.pose_estimation_pytorch as dlc
import deeplabcut.pose_estimation_pytorch.runners.utils as runner_utils
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch import Loader
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_inference_transform,
    build_runner,
    build_transforms,
    update_config_parameters,
)
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.runners.logger import (
    destroy_file_logging,
    LOGGER,
    setup_file_logging,
)


def _train(
    loader: Loader,
    model_folder: str,
    run_config: dict,
    task: str,
    device: str,
    transform_config: dict,
    logger_config: dict | None = None,
    snapshot_path: str | None = None,
    transform: A.BaseCompose | None = None,
) -> None:
    """Builds a model from a configuration and fits it to a dataset

    Args:
        loader: the loader containing the data to train on/validate with
        model_folder: the folder where the models should be saved
        run_config: the model and run configuration
        task: {"TD", "BU", "DT"} the task to train the model for
        device: the device to train on
        transform_config: the configuration of the data augmentation to use. Ignored if
            a transform is given
        logger_config: the configuration of a logger to use
        snapshot_path: if continuing to train from a snapshot, the path containing the
            weights to load
        transform: if None, a transform is loaded with the given configuration.
            Otherwise, this transform is used.
    """
    if task == "DT":
        model = DETECTORS.build(run_config["model"])
    else:
        model = PoseModel.from_cfg(run_config["model"])

    # TODO: Log the configuration file
    #  Model should not be needed when building the logger
    logger = None
    if logger_config is not None:
        logger = LOGGER.build(dict(**logger_config, model=model))

    runner = build_runner(
        run_cfg=run_config,
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        logger=logger,
    )

    batch_size = run_config.get("batch_size", 1)
    epochs = run_config.get("epochs", 200)
    save_epochs = run_config.get("save_epochs", 50)
    display_iters = run_config.get("display_iters", 50)

    if transform is None:
        logging.info(f"No transform passed to augment images for {task}, using default")
        transform = build_transforms(transform_config, augment_bbox=True)
    valid_transform = build_inference_transform(transform_config, augment_bbox=True)

    train_dataset = loader.create_dataset(transform=transform, mode="train", task=task)
    valid_dataset = loader.create_dataset(
        transform=valid_transform, mode="test", task=task
    )
    print(
        f"Using {len(train_dataset)} images to train {task} and {len(valid_dataset)}"
        f" for testing"
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    runner.fit(
        train_dataloader,
        valid_dataloader,
        model_folder=model_folder,
        epochs=epochs,
        save_epochs=save_epochs,
        display_iters=display_iters,
    )


def train_network(
    config: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    transform: A.BaseCompose | None = None,
    transform_cropped: A.BaseCompose | None = None,
    modelprefix: str = "",
    snapshot_path: str | None = "",
    detector_path: str | None = "",
    **kwargs,
) -> None:
    """Trains a network for a project

    TODO: max_snapshots_to_keep

    Args:
        config : path to the yaml config file of the project
        shuffle : index of the shuffle we want to train on
        trainingsetindex : training set index
        transform: Augmentation pipeline for the images
            if None, the augmentation pipeline is built from config files
            Advice if you want to use custom transformations:
                Keep in mind that in order for transfer learning to be efficient, your
                data statistical distribution should resemble the one used to pretrain your backbone

                In most cases (e.g backbone was pretrained on ImageNet), that means it should be Normalized with
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        transform_cropped: Augmentation pipeline for the cropped images around animals
            if None, the augmentation pipeline is built from config files
            Advice if you want to use custom transformations:
                Keep in mind that in order for transfer learning to be efficient, your
                data statistical distribution should resemble the one used to pretrain your backbone
                In most cases (e.g backbone was pretrained on ImageNet), that means it should be Normalized with
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        modelprefix: directory containing the deeplabcut configuration files to use
            to train the network (and where snapshots will be saved). By default, they
             are assumed to exist in the project folder.
        snapshot_path: if resuming training, used to specify the snapshot from which to resume
        detector_path: if resuming training of a top down model, used to specify the detector snapshot from
            which to resume
        **kwargs : could be any entry of the pytorch_config dictionary. Examples are
            to see the full list see the pytorch_cfg.yaml file in your project folder
    """
    cfg = auxiliaryfunctions.read_config(config)
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    model_folder = runner_utils.get_model_folder(
        str(Path(config).parent), cfg, train_fraction, shuffle, modelprefix
    )
    train_folder = Path(model_folder) / "train"
    log_path = train_folder / "log.txt"
    model_config_path = str(train_folder / "pytorch_config.yaml")

    setup_file_logging(log_path)
    pytorch_config = auxiliaryfunctions.read_plainconfig(model_config_path)

    update_config_parameters(pytorch_config=pytorch_config, **kwargs)  # TODO: improve
    if transform is None:
        logging.info("No transform specified... using default")
        transform = build_transforms(dict(pytorch_config["data"]), augment_bbox=True)

    dlc.fix_seeds(pytorch_config["seed"])
    loader = dlc.DLCLoader(
        project_root=pytorch_config["project_path"],
        model_config_path=model_config_path,
        shuffle=shuffle,
    )

    pose_task = "BU"
    transform_config = pytorch_config["data"]
    if pytorch_config.get("method", "bu").lower() == "td":
        pose_task = "TD"
        transform_config = pytorch_config["data_detector"]
        logger_config = None
        if pytorch_config.get("logger"):
            logger_config = copy.deepcopy(pytorch_config["logger"])
            logger_config["run_name"] += "-detector"
        _train(
            loader=loader,
            model_folder=model_folder,
            run_config=pytorch_config["detector"],
            task="DT",
            device=pytorch_config["device"],
            transform_config=pytorch_config["data"],
            logger_config=logger_config,
            snapshot_path=detector_path,
            transform=transform_cropped,
        )

    _train(
        loader=loader,
        model_folder=model_folder,
        run_config=pytorch_config,
        task=pose_task,
        device=pytorch_config["device"],
        transform_config=transform_config,
        logger_config=pytorch_config.get("logger"),
        snapshot_path=snapshot_path,
        transform=transform,
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
