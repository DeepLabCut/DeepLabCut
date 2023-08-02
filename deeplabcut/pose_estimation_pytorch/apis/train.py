import argparse
import os
from typing import Optional, Union

import albumentations as A
import deeplabcut.pose_estimation_pytorch as dlc
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_solver,
    build_transforms,
    update_config_parameters,
)
from deeplabcut.pose_estimation_pytorch.solvers.base import Solver
from torch.utils.data import DataLoader


def train_network(
    config: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    transform: Union[A.BaseCompose, A.BasicTransform] = None,
    transform_cropped: Union[A.BaseCompose, A.BasicTransform] = None,
    modelprefix: str = "",
    snapshot_path: Optional[str] = "",
    detector_path: Optional[str] = "",
    **kwargs
) -> Solver:
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

    Returns:
        solver: solver used for training, stores data about losses during training
    """
    cfg = auxiliaryfunctions.read_config(config)
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    modelfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_model_folder(
            train_fraction,
            shuffle,
            cfg,
            modelprefix=modelprefix,
        ),
    )
    pytorch_config = auxiliaryfunctions.read_plainconfig(os.path.join(modelfolder, "train", "pytorch_config.yaml"))
    update_config_parameters(pytorch_config=pytorch_config, **kwargs)
    if transform is None:
        print("No transform specified... using default")
        transform = build_transforms(dict(pytorch_config["data"]), augment_bbox=True)

    batch_size = pytorch_config["batch_size"]
    epochs = pytorch_config["epochs"]

    dlc.fix_seeds(pytorch_config["seed"])
    project_train = dlc.DLCProject(
        proj_root=pytorch_config["project_path"], shuffle=shuffle
    )
    project_valid = dlc.DLCProject(
        proj_root=pytorch_config["project_path"], shuffle=shuffle
    )
    train_dataset = dlc.PoseDataset(project_train, transform=transform, mode="train")
    valid_dataset = dlc.PoseDataset(project_valid, transform=transform, mode="test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    solver = build_solver(pytorch_config, snapshot_path, detector_path)
    if pytorch_config.get("method", "bu").lower() == "td":
        if transform_cropped is None:
            print(
                "No transform passed to augment cropped images, using default augmentations"
            )
            transform_cropped = build_transforms(
                pytorch_config["cropped_data"], augment_bbox=False
            )

        detector_epochs = pytorch_config["detector"].get("detector_max_epochs", epochs)
        train_cropped_dataset = dlc.CroppedDataset(
            project_train, transform=transform_cropped, mode="train"
        )
        valid_cropped_dataset = dlc.CroppedDataset(
            project_valid, transform=transform_cropped, mode="test"
        )
        train_cropped_dataloader = DataLoader(
            train_cropped_dataset, batch_size=batch_size, shuffle=True
        )

        valid_cropped_dataloader = DataLoader(
            valid_cropped_dataset, batch_size=batch_size, shuffle=False
        )
        solver.fit(
            train_dataloader,
            valid_dataloader,
            train_cropped_dataloader,
            valid_cropped_dataloader,
            train_fraction=train_fraction,
            epochs=epochs,
            detector_epochs=detector_epochs,
            shuffle=shuffle,
            model_prefix=modelprefix,
        )
    elif pytorch_config.get("method", "bu").lower() == "bu":
        solver.fit(
            train_dataloader,
            valid_dataloader,
            train_fraction=train_fraction,
            epochs=epochs,
            shuffle=shuffle,
            model_prefix=modelprefix,
        )
    else:
        raise ValueError(
            "Method not supported, should be either 'bu' (Bottom Up) or 'td' (Top Down)"
        )
    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--train-ind", type=int, default=0)
    parser.add_argument("--modelprefix", type=str, default="")
    args = parser.parse_args()
    _ = train_network(
        config=args.config_path,
        shuffle=args.shuffle,
        trainingsetindex=args.train_ind,
        modelprefix=args.modelprefix,
    )
