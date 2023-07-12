import argparse
import deeplabcut.pose_estimation_pytorch as dlc
import os
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_solver,
    build_transforms,
    update_config_parameters,
)
from deeplabcut.pose_estimation_pytorch.solvers.base import Solver
from torch.utils.data import DataLoader
import albumentations as A
from typing import Union


def train_network(
    config_path: str,
    shuffle: int = 1,
    training_set_index: Union[int, str] = 0,
    transform: Union[A.BaseCompose, A.BasicTransform] = None,
    transform_cropped: Union[A.BaseCompose, A.BasicTransform] = None,
    model_prefix: str = "",
    **kwargs
) -> Solver:
    """
        Trains a network for a project

    Args:
        - config_path : path to the yaml config file of the project
        - shuffle : index of the shuffle we want to train on
        - training_set_index : training set index

        - transform: Augmentation pipeline for the images
            if None, the augmentation pipeline is built from config files
            Advice if you want to use custom transformations:
                Keep in mind that in order for transfer leanring to be efficient, your
                data statistical distribution should resemble the one used to pretrain your backbone

                In most cases (e.g bacbone was pretrained on ImageNet), that means it should be Normalized with
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        - transform_cropped: Augmentation pipeline for the cropped images around animals
            if None, the augmentation pipeline is built from config files
            Advice if you want to use custom transformations:
                Keep in mind that in order for transfer leanring to be efficient, your
                data statistical distribution should resemble the one used to pretrain your backbone

                In most cases (e.g bacbone was pretrained on ImageNet), that means it should be Normalized with
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        - model_prefix: model prefix
        - **kwargs : could be any entry of the pytorch_config dictionary
            to see the full list see the pytorch_cfg.yaml file in your project folder

    Returns:
        solver: solver used for training, stores data about losses during training
    """

    cfg = auxiliaryfunctions.read_config(config_path)
    if training_set_index == "all":
        train_fraction = cfg["TrainingFraction"]
    else:
        train_fraction = [cfg["TrainingFraction"][training_set_index]]
    modelfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_model_folder(
            train_fraction[0],
            shuffle,
            cfg,
            modelprefix=model_prefix,
        ),
    )
    pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
    pytorch_config = auxiliaryfunctions.read_plainconfig(pytorch_config_path)
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

    solver = build_solver(pytorch_config)
    if pytorch_config.get("method", "bu").lower() == "td":
        if transform_cropped == None:
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
            train_fraction=train_fraction[0],
            epochs=epochs,
            detector_epochs=detector_epochs,
            shuffle=shuffle,
            model_prefix=model_prefix,
        )
    elif pytorch_config.get("method", "bu").lower() == "bu":
        solver.fit(
            train_dataloader,
            valid_dataloader,
            train_fraction=train_fraction[0],
            epochs=epochs,
            shuffle=shuffle,
            model_prefix=model_prefix,
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
    solver = train_network(
        config_path=args.config_path,
        shuffle=args.shuffle,
        training_set_index=args.train_ind,
        model_prefix=args.modelprefix,
    )
