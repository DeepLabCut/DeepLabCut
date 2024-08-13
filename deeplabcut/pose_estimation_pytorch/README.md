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

Some object detectors are also available (and implemented in 
`deeplabcut.pose_estimations_pytorch.models.detectors`).

If you want to add a novel model, you need to divide it into a model backbone, neck and
head. Often the 'neck' will be just the identity function.

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

TODO

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
    task=Task.BOTTOM_UP,
)
valid_dataset = loader.create_dataset(
    transform=build_transforms(loader.model_cfg["data"]["train"]),
    mode="test",
    task=Task.BOTTOM_UP,
)
```

### Runners

The `deeplabcut.pose_estimations_pytorch.runners` contains code to get models,
