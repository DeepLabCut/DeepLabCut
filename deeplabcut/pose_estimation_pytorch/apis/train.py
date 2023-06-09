import argparse
import deeplabcut.pose_estimation_pytorch as dlc
import os
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import build_solver, build_transforms
from deeplabcut.pose_estimation_pytorch.models.target_generators import TARGET_GENERATORS
from torch.utils.data import DataLoader
import albumentations as A
from typing import Union


def train_network(
        config_path: str,
        shuffle: int = 1,
        training_set_index: Union[int, str] = 0,
        transform = None,
        model_prefix: str = ""):
    cfg = auxiliaryfunctions.read_config(config_path)
    if training_set_index == "all":
        train_fraction = cfg["TrainingFraction"]
    else:
        train_fraction = [cfg["TrainingFraction"][training_set_index]]
    modelfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_model_folder(
            train_fraction[0], shuffle, cfg, modelprefix=model_prefix,
        ),
    )
    pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
    pytorch_config = auxiliaryfunctions.read_plainconfig(pytorch_config_path)


    transform = build_transforms(dict(pytorch_config['data']))
    batch_size = pytorch_config['batch_size']
    epochs = pytorch_config['epochs']

    dlc.fix_seeds(pytorch_config['seed'])
    project_train = dlc.DLCProject(proj_root=pytorch_config['project_path'], shuffle=shuffle)
    project_valid = dlc.DLCProject(proj_root=pytorch_config['project_path'], shuffle=shuffle)
    train_dataset = dlc.PoseDataset(project_train,
                                    transform=transform,
                                    mode='train')
    valid_dataset = dlc.PoseDataset(project_valid,
                                    transform=transform,
                                    mode='test')

    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    valid_dataloader = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    solver = build_solver(pytorch_config)
    solver.fit(
        train_dataloader,
        valid_dataloader,
        train_fraction=train_fraction[0],
        epochs=epochs,
        shuffle=shuffle,
        model_prefix=model_prefix,
    )
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--train-ind", type=int, default=0)
    parser.add_argument("--modelprefix", type=str, default="")
    args = parser.parse_args()
    solver = train_network(
        config_path = args.config_path,
        shuffle=args.shuffle,
        training_set_index=args.train_ind,
        model_prefix=args.modelprefix,
    )
