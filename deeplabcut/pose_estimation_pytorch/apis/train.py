from torch.utils.data import DataLoader

import deeplabcut.pose_estimation_pytorch as dlc
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import build_solver

config = auxiliaryfunctions.read_config('config.yaml')
batch_size = config['batch_size']
device = config['device']
epochs = config['epochs']

transform = None
dlc.fix_seeds(config['seed'])
project = dlc.DLCProject(proj_root=config['project_root'])

train_dataset = dlc.PoseDataset(project,
                                transform=transform,
                                mode='train')
valid_dataset = dlc.PoseDataset(project,
                                transform=transform,
                                mode='test')

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

valid_dataloader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)

solver = build_solver(config)

solver.fit(train_dataloader, valid_dataloader, epochs=epochs)

solver.evaluate(valid_dataset,
                plotting=False,
                train_iterations=epochs - 1)
