import deeplabcut.pose_estimation_pytorch as dlc
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import build_solver

config = auxiliaryfunctions.read_config('config.yaml')
batch_size = config['batch_size']
device = config['device']

transform = None
dlc.fix_seeds(config['seed'])
project = dlc.DLCProject(proj_root=config['project_root'])
solver = build_solver(config)

test_dataset = dlc.PoseDataset(project,
                               mode='test')

solver.evaluate(test_dataset,
                train_iterations=49,
                plotting=True)