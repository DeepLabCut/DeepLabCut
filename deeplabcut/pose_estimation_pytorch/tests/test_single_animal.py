import os
import pickle
import time

import albumentations as A
import numpy as np
import torch
import yaml

import deeplabcut
from deeplabcut.pose_estimation_pytorch.apis.utils import build_pose_model
from deeplabcut.pose_estimation_pytorch.models.criterion import PoseLoss
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_prediction
from deeplabcut.pose_estimation_pytorch.solvers.utils import (
    get_paths,
    get_results_filename,
)


def read_yaml(path):
    try:
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    except:
        raise FileNotFoundError("An eero occured whilereading the file")


def get_training_set_length(cfg, train_fraction, shuffle):
    training_folder = os.path.join(
        cfg["project_path"], deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg)
    )
    train_idx_path = os.path.join(
        training_folder,
        f'Documentation_data-{cfg["Task"]}_{int(train_fraction*100)}shuffle{shuffle}.pickle',
    )

    with open(train_idx_path, "rb") as file:
        meta = pickle.load(file)

    print(
        f"length of the training set {len(meta[1])}, length of the test set {len(meta[2])}"
    )
    return len(meta[1])


def load_model(cfg, pytorch_config, shuffle, model_prefix="", train_iteration=-1):
    names = get_paths(
        train_fraction=cfg["TrainingFraction"][0],
        model_prefix=model_prefix,
        shuffle=shuffle,
        cfg=cfg,
        train_iterations=train_iteration,
    )
    print(names["model_path"])

    results_filename = get_results_filename(
        names["evaluation_folder"],
        names["dlc_scorer"],
        names["dlc_scorer_legacy"],
        names["model_path"][:-3],
    )

    pose_cfg = deeplabcut.auxiliaryfunctions.read_config(
        pytorch_config["pose_cfg_path"]
    )
    model = build_pose_model(pytorch_config["model"], pose_cfg)
    model.load_state_dict(torch.load(names["model_path"]))

    return model


def evaluate_network_custom(
    config_path, shuffle, model_prefix="", transform=None, train_iteration=-1
):
    cfg = read_yaml(config_path)
    train_fraction = cfg["TrainingFraction"][0]
    model_folder = os.path.join(
        cfg["project_path"],
        deeplabcut.auxiliaryfunctions.get_model_folder(
            train_fraction, shuffle, cfg, modelprefix=model_prefix
        ),
    )
    pytorch_config_path = os.path.join(model_folder, "train", "pytorch_config.yaml")
    pytorch_config = read_yaml(pytorch_config_path)
    pose_cfg = deeplabcut.auxiliaryfunctions.read_config(
        pytorch_config["pose_cfg_path"]
    )

    batch_size = pytorch_config["batch_size"]
    project = deeplabcut.pose_estimation_pytorch.DLCProject(
        shuffle=shuffle, proj_root=pytorch_config["project_root"]
    )

    valid_dataset = deeplabcut.pose_estimation_pytorch.PoseDataset(
        project, transform=transform, mode="train"
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )

    model = load_model(cfg, pytorch_config, shuffle, model_prefix, train_iteration)
    model.to("cuda")
    model.eval()
    criterion = PoseLoss(locref_huber_loss=True)

    with torch.no_grad():
        losses = []
        rmses = []
        for i, item in enumerate(valid_dataloader):
            _, keypoints = item
            if isinstance(item, tuple) or (isinstance, list):
                item = item[0].to("cuda")
            output = model(item)

            scale_factor = (
                item.shape[2] / output[0].shape[2],
                item.shape[3] / output[0].shape[3],
            )

            gt = model.get_target(keypoints, output[0].shape[2:], scale_factor)
            for key in gt:
                if gt[key] is not None:
                    gt[key] = gt[key].to("cuda")

            predictions = get_prediction(pose_cfg, output, scale_factor)

            rmse = keypoints.numpy() - predictions[:, :, :2]
            rmse *= rmse
            rmse = np.sqrt(rmse.sum(axis=2))
            rmses.append(np.nanmean(rmse))

            losses.append(criterion(output, gt)[0].cpu().numpy())

        print(np.mean(losses), np.nanmean(rmses))
        return np.mean(losses), np.nanmean(rmses)


def runBenchmark(path_dataset, train_fraction, shuffle, transform=None):
    """Trains the model and evaluates it on a given dataset"""
    config_path = os.path.join(path_dataset, "config.yaml")

    # Training the network
    print("Training started")
    start_time = time.time()
    deeplabcut.pose_estimation_pytorch.apis.train.train_network(
        config_path, shuffle=shuffle, transform=transform
    )
    delta_time = time.time() - start_time
    print("Training ended")

    # #evaluate the nework
    print("Starting evaluation of the last saved model")
    evaluate_network_custom(config_path, shuffle, transform=transform)


class CustomHorizontalFlip(A.HorizontalFlip):
    def __init__(self, flipped_keypoints, always_apply=False, p=0.5):
        """
        flipped_keypoints : list of the new order of keypoints
        """
        super().__init__(always_apply=always_apply, p=p)
        self.flipped_keypoints = flipped_keypoints

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = list(super().apply_to_keypoints(keypoints, **params))

        return [keypoints[i] for i in self.flipped_keypoints]


if __name__ == "__main__":
    path_dataset = "/home/quentin/datasets/Openfield_pytorch"
    config_path = os.path.join(path_dataset, "config.yaml")

    cfg = read_yaml(config_path)
    if cfg.get("flipped_keypoints"):
        flip_transform = CustomHorizontalFlip(cfg["flipped_keypoints"])
    else:
        flip_transform = A.HorizontalFlip()

    transform = A.Compose(
        [
            flip_transform,
            A.RandomScale(scale_limit=[-0.25, 0.25]),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=10),
            A.MotionBlur(),
            A.PixelDropout(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    runBenchmark(path_dataset, 0.95, 1, transform=transform)
