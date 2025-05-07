from deeplabcut.core.config import read_config_as_dict
from pathlib import Path

import deeplabcut.pose_estimation_pytorch as dlc_torch

train_dir = Path("/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/dlc-models-pytorch/iteration-0/uk_harNov4-trainset95shuffle1/train/")
pytorch_config_path = train_dir / "pytorch_config.yaml"
snapshot_path = train_dir / "snapshot-050.pt"

# for top-down models, otherwise None
detector_snapshot_path = train_dir / "snapshot-detector-050.pt"

# video and inference parameters
max_num_animals = 5
batch_size = 16
detector_batch_size = 8

# read model configuration
model_cfg = read_config_as_dict(pytorch_config_path)
pose_task = dlc_torch.Task(model_cfg["method"])
pose_runner = dlc_torch.get_pose_inference_runner(
    model_config=model_cfg,
    snapshot_path=snapshot_path,
    max_individuals=max_num_animals,
    batch_size=batch_size,
)

detector_runner = dlc_torch.get_detector_inference_runner(
    model_config=model_cfg,
    snapshot_path=detector_snapshot_path,
    max_individuals=max_num_animals,
    batch_size=detector_batch_size,
)

predictions = dlc_torch.analyze_image_folder(
    model_cfg=pytorch_config_path,
    images=Path("/home/max/tmp/foxes/"),
    snapshot_path=Path(snapshot_path),
    detector_path=Path(detector_snapshot_path),
)