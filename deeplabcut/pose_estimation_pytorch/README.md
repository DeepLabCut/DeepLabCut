# PyTorch DeepLabCut API

This is written primarily for maintainers and expert users. It details the logic for the
DLC3.0 PyTorch code.

## Structure of the PyTorch DLC code

[API](#API)
[Models](#models)
[Data](#data)
[Runners](#runners)

### API

High-level API methods are implemented in `deeplabcut.pose_estimations_pytorch.apis`.
This includes methods to train and evaluate models on DeepLabCut projects, and analyze 
videos or folders of images. While some of the methods are implemented to work directly
from DeepLabCut projects (i.e. by specifying the path to the project config file and the
shuffle number), internally they call methods that allow more flexibility.

### Models

The `deeplabcut.pose_estimations_pytorch.models` package contains all components related
to building a model with `backbone`, `neck` (optional) and `head`.

We provide state-of-the-art models such as DLCRNet, HRNet, DEKR and more are coming 
(BUCTD is in the works ;))!

Object detection models are also available (and implemented in 
`deeplabcut.pose_estimations_pytorch.models.detectors`).

If you want to add a novel model, you'll need the following parts implemented:
- a backbone (such as a ResNet or HRNet)
- a head (such as a HeatmapHead)
- a predictor (transforming model outputs into keypoint locations)
- a target generator (creating the targets for your head outputs from your labels)

Some models can also define a neck. You'll also need some loss criterions, but usually 
you'll be able to use existing ones. You can either use existing classes and only 
replace some elements, or rewrite everything you need for your model!

#### Model Configuration Files

Model architectures are built according to a configuration specified in a `yaml` file.
This file (named `pytorch_cfg.yaml`) describes the architecture of the model you want to
train (but also hyperparameters, optimizer, ...). All code to manipulate PyTorch 
configuration files is in `deeplabcut.pose_estimations_pytorch.config`.

To generate a model configuration, you can call `make_pytorch_pose_config`. Note that 
this does not save the configuration to a given filepath - it just returns it as a 
dictionary. However, you can save it with `write_config`. 

During a typical DeepLabCut project management workflow, these methods don't need to be 
called, as `create_training_dataset` will create this configuration file and save it to 
disk.

```python
from pathlib import Path

from deeplabcut.pose_estimation_pytorch.config import (
    make_pytorch_pose_config,
    write_config,
)

project_cfg = { "Task": "mice", ... }  # the configuration for your DLC project
pose_config_path = Path("/path/to/my/config/pytorch_cfg.yaml")
model_cfg = make_pytorch_pose_config(
    project_config=project_cfg,
    pose_config_path=pose_config_path,
    net_type="hrnet_w32",
    top_down=True,
)
write_config(pose_config_path, model_cfg)
```

#### Model Registry

Registries are created for all model building blocks to make it easy to add new models.
All you need to do is add the decorator `REGISTRY.register_module` to be able to load 
your model from a configuration file. Available registries are `BACKBONES`, `NECKS`,
`HEADS`, `PREDICTORS` and `TARGET_GENERATORS`. Each building block has a base class
that should be inherited by the class added to the model registry (`BaseBackbone`,
`BaseNeck`, `BaseHead`, `BasePredictor` and `BaseGenerator` respectively).

Let's illustrate that with a small example. We'll create a dummy backbone, which simply
applies a max-pool to the input:

```python
import torch
import torch.nn.functional as F

from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone


@BACKBONES.register_module
class DummyBackbone(BaseBackbone):
    """A dummy backbone, simply max-pooling the input"""
    
    def __init__(self, kernel_size: int = 2):
        super().__init__(stride=kernel_size)
        self.kernel_size = kernel_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size)


backbone_config = dict(type="DummyBackbone", kernel_size=3)
backbone = BACKBONES.build(backbone_config)  # will create a DummyBackbone
```

### Data

The `deeplabcut.pose_estimations_pytorch.data` package contains all code for PyTorch
dataset creation and test/train splitting. The `DLCLoader` class is used to load the
labeled data for a specific shuffle.

```python3
from deeplabcut.pose_estimation_pytorch.data import DLCLoader

loader = DLCLoader(
    config="/path/to/my/project/config.yaml",
    trainset_index=0,
    shuffle=1,
)

# print the path to the model folder (where the config file is stored)
print(loader.model_folder)
# print the path to the evaluation folder
print(loader.evaluation_folder)

# display the DataFrame containing the dataset
print(loader.df)

# display the DataFrames containing the train/test data respectively
print(loader.df_train)
print(loader.df_test)
```

The `PoseDataset` class is an instance of 
[torch.utils.Dataset](https://pytorch.org/docs/stable/data.html), which converts raw 
images and keypoints to a tensor dataset for training and evaluation. You can generate 
an instance of training/test dataset with your `DLCLoader`:

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

A `COCOLoader` is also available, and allows you train models in DeepLabCut on 
[COCO-format](https://medium.com/@manuktiwary/coco-format-what-and-how-5c7d22cf5301)
datasets. This essentially consists of having a folder containing your dataset in the 
format:

```
COCOProject
└───annotations
│   │   train.json
│   │   test.json
│   
└───images
    │   img0000.png
    │   img0001.png
    │           ...
```

In your `train.json` and `test.json` files, you can either specify your image 
`"file_name"` with a relative path or with an absolute path. If a relative path is 
used (e.g. `img0000.png` or `subfolder/img0000.png`), it will be resolved to the 
`images` folder in your project (i.e. `/path/to/COCOProject/images/img0000.png` or 
`/path/to/COCOProject/images/subfolder/img0000.png`).

If you specify an absolute path, the path to the image will not be resolved, and the 
image will be loaded from the specified path. This allows you to keep data on different
disks, or reuse the same images in different projects without having to duplicate them.

To train a DeepLabCut model on a COCO-format dataset, you'll need to specify a model 
configuration file (as described in [#model_configuration_files]).

```python3
from pathlib import Path

from deeplabcut.pose_estimation_pytorch.config import (
    make_pytorch_pose_config,
    write_config,
)
from deeplabcut.pose_estimation_pytorch.data import (
    build_transforms,
    COCOLoader,
)
from deeplabcut.pose_estimation_pytorch.task import Task

# Specify project paths
project_root = Path("/path/to/my/COCOProject")
train_json_filename = "train.json"
test_json_filename = "test.json"

# Parse information about the project
train_dict = COCOLoader.load_json(project_root, filename=train_json_filename)
max_num_individuals, bodyparts = COCOLoader.get_project_parameters(train_dict)

# Generate a configuration file for your PyTorch model
# In this case, it's for a Top-Down HRNet_w32
experiment_path = project_root / "experiments" / "hrnet_w32"
model_cfg_path = experiment_path / "train" / "pytorch_cfg.yaml"
model_cfg = make_pytorch_pose_config(
    project_config={
        "project_path": str(project_root.resolve()),
        "multianimalproject": max_num_individuals > 1,
        "bodyparts": bodyparts,
        "multianimalbodyparts": bodyparts,
        "uniquebodyparts": [],
        "individuals": [f"idv{i}" for i in range(max_num_individuals)],
    },
    pose_config_path=experiment_path,
    net_type="hrnet_w32",
    top_down=True,
)
write_config(config_path=model_cfg_path, config=model_cfg)
task = Task(model_cfg["method"])

# Create the loader for the COCO dataset
loader = COCOLoader(
    project_root=project_root,
    model_config_path="/path/to/my/project/experiments/pytorch_config.yaml",
    train_json_filename=train_json_filename,
    test_json_filename=test_json_filename,
)
train_dataset = loader.create_dataset(
    transform=build_transforms(loader.model_cfg["data"]["train"]),
    mode="train",
    task=task,
)
valid_dataset = loader.create_dataset(
    transform=build_transforms(loader.model_cfg["data"]["train"]),
    mode="test",
    task=task,
)
```

### Runners

The `deeplabcut.pose_estimations_pytorch.runners` contains code to get models, load 
pretrained weights, and either train them or run inference with them.

## Code Examples

### Training a Model on a COCO Dataset

```python
from pathlib import Path

from deeplabcut.pose_estimation_pytorch.apis.train import train
from deeplabcut.pose_estimation_pytorch.data import COCOLoader
from deeplabcut.pose_estimation_pytorch.task import Task

# Specify project paths
project_root = Path("/path/to/my/COCOProject")
train_json_filename = "train.json"
test_json_filename = "test.json"

loader = COCOLoader(
    project_root=project_root,
    model_config_path="/path/to/my/project/experiments/pytorch_config.yaml",
    train_json_filename=train_json_filename,
    test_json_filename=test_json_filename,
)
train(
    loader=loader,
    run_config=loader.model_cfg,
    task=Task(loader.model_cfg["method"]),
    device="cuda:2",
    logger_config=dict(
        type="WandbLogger",
        project_name="MyWandbProject",
        tags=["model=hrnet_w32"],
    ),
    snapshot_path=None,
)
```

### Running Video Analysis outside a DeepLabCut Project

```python
from pathlib import Path

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import video_inference
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

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
bodyparts = model_cfg["metadata"]["bodyparts"]
unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
with_identity = model_cfg["metadata"].get("with_identity", False)

pose_task = Task(model_cfg["method"])
pose_runner, detector_runner = get_inference_runners(
    model_config=model_cfg,
    snapshot_path=snapshot_path,
    max_individuals=max_num_animals,
    num_bodyparts=len(bodyparts),
    num_unique_bodyparts=len(unique_bodyparts),
    batch_size=batch_size,
    with_identity=with_identity,
    transform=None,
    detector_batch_size=detector_batch_size,
    detector_path=detector_snapshot_path,
    detector_transform=None,
)

predictions = video_inference(
    video_path=video_path,
    task=pose_task,
    pose_runner=pose_runner,
    detector_runner=detector_runner,
    with_identity=False,
)
```
