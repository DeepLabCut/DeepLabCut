# Code Examples

This guide provides practical examples for common DeepLabCut PyTorch workflows.

## Training a Model on a COCO Dataset

```python
from pathlib import Path

import deeplabcut.pose_estimation_pytorch as dlc_torch

# Specify project paths
project_root = Path("/path/to/my/COCOProject")
train_json_filename = "train.json"
test_json_filename = "test.json"

loader = dlc_torch.COCOLoader(
    project_root=project_root,
    model_config_path="/path/to/my/project/experiments/pytorch_config.yaml",
    train_json_filename=train_json_filename,
    test_json_filename=test_json_filename,
)
dlc_torch.train(
    loader=loader,
    run_config=loader.model_cfg,
    task=dlc_torch.Task(loader.model_cfg["method"]),
    device="cuda:2",
    logger_config=dict(
        type="WandbLogger",
        project_name="MyWandbProject",
        tags=["model=hrnet_w32"],
    ),
    snapshot_path=None,
)
```

## Running Video Analysis outside a DeepLabCut Project

DeepLabCut provides high-level APIs (via the GUI or the python package) to analyze your data. The usage of this API assumes the existence of a DLC project (with `config.yaml` file, etc.).

Sometimes it might be more convenient to just run a model on your data via a low-level API. We also use this API under the hood, in particular for the Model Zoo. Check out the example below:

```python
from deeplabcut.core.config import read_config_as_dict
from pathlib import Path

import deeplabcut.pose_estimation_pytorch as dlc_torch

train_dir = Path("/Users/Jaylen/my-dlc-models/train")
pytorch_config_path = train_dir / "pytorch_config.yaml"
snapshot_path = train_dir / "snapshot-100.pt"

# for top-down models, otherwise None
detector_snapshot_path = train_dir / "detector-snapshot-100.pt"

# video and inference parameters
video_path = Path("/Users/Jaylen/my-dlc-models/videos/test-video.mp4")
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

detector_runner = None
if pose_task == dlc_torch.Task.TOP_DOWN:
    detector_runner = dlc_torch.get_detector_inference_runner(
        model_config=model_cfg,
        snapshot_path=detector_snapshot_path,
        max_individuals=max_num_animals,
        batch_size=detector_batch_size,
    )

predictions = dlc_torch.video_inference(
    video=video_path,
    pose_runner=pose_runner,
    detector_runner=detector_runner,
)
```

## Running Top-Down Video Analysis with Existing Bounding Boxes

When `deeplabcut.pose_estimation_pytorch.apis.videos.video_inference` is called with a top-down model, it is assumed that a detector snapshot is given as well to obtain bounding boxes with which to run pose estimation. It's possible that you've already obtained bounding boxes for your video (with another object detector or through some other means), and you want to reuse those bounding boxes instead of running an object detector again.

You can easily do so by writing a bit of custom code, as shown in the example below:

```python
from deeplabcut.core.config import read_config_as_dict
from pathlib import Path

import numpy as np
import deeplabcut.pose_estimation_pytorch as dlc_torch
from tqdm import tqdm

# create an iterator for your video
video = dlc_torch.VideoIterator("/Users/Jayson/my-cool-video.mp4")

# dummy bboxes - you can load yours from a file or in another way
#  the bboxes should be in `xywh` format, i.e. (x_top_left, y_top_left, width, height)
bounding_boxes = [
    dict(  # frame 0 bounding boxes
        bboxes=np.array([[12, 37, 120, 78]]),
    ),
    dict(  # frame 1 bounding boxes
        bboxes=np.array([[17, 45, 128, 73], [532, 34, 117, 87]]),
    ),
    # ...
    dict(  # frame N bboxes -> must be equal to the number of frames in the video!
        bboxes=np.array([[17, 45, 128, 73], [532, 34, 117, 87]]),
    ),
]
video.set_context(bounding_boxes)
max_individuals = np.max([len(context["bboxes"]) for context in bounding_boxes])

# run inference!
model_cfg = read_config_as_dict("/Users/Jayson/pytorch_config.yaml")
pose_runner = dlc_torch.get_pose_inference_runner(
    model_config=model_cfg,
    snapshot_path=Path("/Users/Jayson/model-snapshot.pt"),
    max_individuals=max_individuals,
    batch_size=32,
)

# your predictions will be a list, containing the predictions made for each frame
#  as a dict (with keys for "bodyparts" but also "bboxes")!
predictions = pose_runner.inference(images=tqdm(video))
```

