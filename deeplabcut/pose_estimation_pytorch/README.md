# PyTorch DeepLabCut API

This is written primarily for maintainers and expert users. It details the logic for the
DLC3.0 PyTorch code.

**Structure of the PyTorch DLC code:**

[Models](#models)   
[Runners](#runners)   
[Data](#data)  
[API](#API)

## API

High-level API methods are implemented in `deeplabcut.pose_estimations_pytorch.apis`.
This includes methods to train and evaluate models on DeepLabCut projects, and analyze 
videos or folders of images. While some of the methods are implemented to work directly
from DeepLabCut projects (i.e. by specifying the path to the project config file and the
shuffle number), internally they call methods that allow more flexibility.

## Data

The `deeplabcut.pose_estimations_pytorch.data` package contains all code for pytorch
dataset creation and test/train splitting.

`Project` class provides train and test splitting and converts dataset to required 
format. For instance, to 
[COCO](https://medium.com/@manuktiwary/coco-format-what-and-how-5c7d22cf5301) format.

Example:

```python3
import deeplabcut.pose_estimation_pytorch as dlc

project = dlc.Project(proj_root=config['project_root'])
project.train_test_split()
```

`PoseDataset` class is an instance of [torch.utils.Dataset](https://pytorch.org/docs/stable/data.html), which converts raw 
images and keypoints to a tensor dataset for training and evaluation.

Example:

```python3
from deeplabcut.pose_estimation_pytorch.data import (
    build_transforms,
    DLCLoader,
)
from deeplabcut.pose_estimation_pytorch.task import Task

loader = DLCLoader(
    config="/path/to/my/project/config.yaml",
    trainset_index=0,
    shuffle=1,
)
train_dataset = loader.create_dataset(
    transform=build_transforms(loader.model_cfg["data"]["train"]),
    mode="train",
    task=Task.BOTTOM_UP,
)
valid_dataset = loader.create_dataset(
transform=build_transforms(loader.model_cfg["data"]["train"]),
    mode="test",
    task=Task.BOTTOM_UP,
)
```


## Models

The `deeplabcut.pose_estimations_pytorch.models` package contains all components related
to building a model with `backbone`, `neck` (optional) and `head`.

We provide state-of-the-art models such as DLCRNet, HRNet, DEKR and more are coming 
(BUCTD is in the works ;))!

Some object detectors are also available (and implemented in 
`deeplabcut.pose_estimations_pytorch.models.detectors`).

If you want to add a novel model, you need to divide it into a model backbone, neck and
head. Often the 'neck' will be just the identity function.

## Runners

The `deeplabcut.pose_estimations_pytorch.runners` contains code to get models,
