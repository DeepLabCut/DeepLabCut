# Training Models

The [`deeplabcut.pose_estimation_pytorch`][] package provides tools for training pose estimation models.

## Training on a DeepLabCut Project

For standard DeepLabCut projects, use [`train_network`][deeplabcut.pose_estimation_pytorch.apis.train_network]:

```python
import deeplabcut

deeplabcut.train_network(
    config="/path/to/project/config.yaml",
    shuffle=1,
    trainingsetindex=0,
    max_epochs=200,
    displayiters=100,
    saveiters=10000,
)
```

## Training on COCO Datasets

You can train directly on COCO-format datasets:

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
