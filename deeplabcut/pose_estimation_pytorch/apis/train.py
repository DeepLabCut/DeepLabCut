import argparse
import deeplabcut.pose_estimation_pytorch as dlc
import os
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import build_solver
from torch.utils.data import DataLoader
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
    config = auxiliaryfunctions.read_config(pytorch_config_path)
    batch_size = config['batch_size']
    epochs = config['epochs']

    dlc.fix_seeds(config['seed'])
    project_train = dlc.DLCProject(proj_root=config['project_root'], shuffle=shuffle)
    project_valid = dlc.DLCProject(proj_root=config['project_root'], shuffle=shuffle)
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

    solver = build_solver(config)
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
