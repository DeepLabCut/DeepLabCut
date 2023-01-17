import deeplabcut.pose_estimation_pytorch as dlc
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_prediction
from deeplabcut.pose_estimation_pytorch.apis.utils import build_pose_model

import torch
from torch.utils.data import DataLoader


config = auxiliaryfunctions.read_config('pytorch_config.yaml')
batch_size = config['batch_size']
device = config['device']

transform = None
dlc.fix_seeds(config['seed'])

### Load data
project = dlc.Project(proj_root=config['project_root'])
project.train_test_split()
valid_dataset = dlc.PoseDataset(project,
                                transform=transform,
                                mode='test')
valid_dataloader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=1)

### TODO: anothor option: user can load ood data

checkpoint_path="../../../examples/openfield-Pranav-2018-10-30/dlc-models/iteration-0/openfieldOct30-trainset95shuffle0/train/snapshot-1.pt"
pose_cfg = auxiliaryfunctions.read_config(config['pose_cfg_path'])
model = build_pose_model(config['model'], pose_cfg)
model.load_state_dict(torch.load(checkpoint_path), strict=False)

output = model(valid_dataloader)
predictions = get_prediction(config, output, stride)
