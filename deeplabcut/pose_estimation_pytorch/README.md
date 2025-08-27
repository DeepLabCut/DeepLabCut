# PyTorch DeepLabCut API

This overview is primarily written for maintainers and expert users. 

Here we detail the logic and structure for the DLC3.* PyTorch code. Furthermore, we
provide many practical examples to illustrate the usage of the code for developers. 

## Structure of the PyTorch DLC code

[API](#API)

[Models](#models)

[Data](#data)

[Runners](#runners)

### API

High-level API methods are implemented in `deeplabcut.pose_estimations_pytorch.apis`.
This folder includes methods to train and evaluate models on DeepLabCut projects, and
analyze videos or folders (of images). While some of the methods are implemented to work
directly from DeepLabCut projects (i.e. by specifying the path to the project config
file and the shuffle number), internally they call methods that allow more flexibility.
Thus, they are also ideally suited for developers.

### Models

We provide state-of-the-art pose estimation models such as DLCRNet, HRNet, DEKR, BUCTD
and more are coming! Object detection models are also available (and implemented in 
`deeplabcut.pose_estimations_pytorch.models.detectors`).

The `deeplabcut.pose_estimations_pytorch.models` package contains all components related
to building a model. Models are flexibly build from modular components: `backbone`, 
`neck` (optional) and `head` (as discussed below). 

You can check available models by running:

```python
import deeplabcut.pose_estimation_pytorch

# Available pose estimation models
print(deeplabcut.pose_estimation_pytorch.available_models())

# Available object detection models
print(deeplabcut.pose_estimation_pytorch.available_detectors())
```

### Model Configuration Files

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

import deeplabcut.pose_estimation_pytorch as dlc_torch

project_cfg = { "Task": "mice", ... }  # the configuration for your DLC project
pose_config_path = Path("/path/to/my/config/pytorch_cfg.yaml")
model_cfg = dlc_torch.config.make_pytorch_pose_config(
    project_config=project_cfg,
    pose_config_path=pose_config_path,
    net_type="hrnet_w32",
    top_down=True,
    save=True,
)
```

### Adding Models

If you want to add a novel model, you'll ideally build them from the following
implemented parts:

- a backbone (such as a ResNet or HRNet)
- a head (such as a HeatmapHead)
- a predictor (transforming model outputs into keypoint locations)
- a target generator (creating the targets for your head outputs from your labels)

Some models can also define a neck (model components between the backbone and the head).
You'll also need some loss criterions, but usually you'll be able to use existing ones.

You can either use existing classes and only replace some elements, or rewrite
everything you need for your model. We use Model Registries to simplify the process of
adding models.

### Model Registry

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

Another example would be creating a custom head for our model. In this case, let's make
a head which takes as input the output of a backbone (which has shape `(num_channels,
H', W')`) and put it through a kernel-size 1 convolution, simply changing the number of
channels.

Heads can output multiple tensors (such as heatmaps and location refinement fields). 
Therefore, their `forward(...)` method outputs a dictionary mapping strings to tensors.
Here, we return the `heatmap` and `locref` tensors.

A head must contain different: a `target_generator` to generate targets for
its outputs and a `predictor` to convert model outputs to pose. Make sure that the keys
output by the `target_generator` and the `head` match! Some `criterion` also needs to be
defined to compute the loss between the outputs and targets. When more than one output 
is specified (such as in this case, where we're generating heatmaps and location 
refinement fields), a loss aggregator must also be given to combine all losses into one
(this should simply be a `WeightedLossAggregator`, indicating the weight for each loss).

```python
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.criterions import (
    BaseCriterion,
    BaseLossAggregator,
    WeightedHuberCriterion,
    WeightedLossAggregator,
    WeightedMSECriterion,
)
from deeplabcut.pose_estimation_pytorch.models.heads import HEADS, BaseHead
from deeplabcut.pose_estimation_pytorch.models.predictors import (
    BasePredictor,
    HeatmapPredictor,
)
from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    BaseGenerator,
    HeatmapGaussianGenerator,
)


@HEADS.register_module
class DummyHead(BaseHead):
    """A dummy backbone, simply max-pooling the input"""
    
    def __init__(
        self,
        num_input_channels: int,
        num_bodyparts: int,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion],
        aggregator: BaseLossAggregator,
    ):
        super().__init__(
            stride=1,
            predictor=predictor,
            target_generator=target_generator,
            criterion=criterion,
            aggregator=aggregator
        )
        self.conv_heatmap = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_bodyparts,
            kernel_size=1,
            stride=1,
        )
        self.locref_heatmap = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=2 * num_bodyparts,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "heatmap": self.conv_heatmap(x),
            "locref": self.locref_heatmap(x),
        }


head_config = dict(
    type="DummyHead",
    num_input_channels=2048,
    num_bodyparts=5,
    predictor=HeatmapPredictor(location_refinement=True, locref_std= 7.2801),
    target_generator=HeatmapGaussianGenerator(
        num_heatmaps=5,
        pos_dist_thresh=17,
        heatmap_mode=HeatmapGaussianGenerator.Mode.KEYPOINT,
        generate_locref=True,
    ),
    criterion={
        "heatmap": WeightedMSECriterion(),
        "locref": WeightedHuberCriterion(),
    },
    aggregator=WeightedLossAggregator(weights={"heatmap": 1, "locref": 0.05}),
)
head = HEADS.build(head_config)
```

### Data

The `deeplabcut.pose_estimations_pytorch.data` package contains all code for PyTorch
dataset creation and test/train splitting. The `DLCLoader` class is used to load the
labeled data for a specific shuffle.

```python3
import deeplabcut.pose_estimation_pytorch as dlc_torch

loader = dlc_torch.DLCLoader(
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
import deeplabcut.pose_estimation_pytorch as dlc_torch

loader = dlc_torch.DLCLoader(
    config="/path/to/my/project/config.yaml",
    trainset_index=0,
    shuffle=1,
)
train_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["train"]),
    mode="train",
    task=loader.pose_task,
)
valid_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["inference"]),
    mode="test",
    task=loader.pose_task,
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

import deeplabcut.pose_estimation_pytorch as dlc_torch

# Specify project paths
project_root = Path("/path/to/my/COCOProject")
train_json_filename = "train.json"
test_json_filename = "test.json"

# Parse information about the project
train_dict = dlc_torch.COCOLoader.load_json(project_root, filename=train_json_filename)
max_num_individuals, bodyparts = dlc_torch.COCOLoader.get_project_parameters(train_dict)

# Generate a configuration file for your PyTorch model
# In this case, it's for a Top-Down HRNet_w32
experiment_path = project_root / "experiments" / "hrnet_w32"
model_cfg_path = experiment_path / "train" / "pytorch_cfg.yaml"
model_cfg = dlc_torch.config.make_pytorch_pose_config(
    project_config=dlc_torch.config.make_basic_project_config(
        dataset_path=str(project_root.resolve()),
        bodyparts=bodyparts,
        max_individuals=max_num_individuals,
        multi_animal=True,
    ),
    pose_config_path=experiment_path,
    net_type="hrnet_w32",
    top_down=True,
    save=True,
)

# Create the loader for the COCO dataset
loader = dlc_torch.COCOLoader(
    project_root=project_root,
    model_config_path="/path/to/my/project/experiments/pytorch_config.yaml",
    train_json_filename=train_json_filename,
    test_json_filename=test_json_filename,
)
train_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["train"]),
    mode="train",
    task=loader.pose_task,
)
valid_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["inference"]),
    mode="test",
    task=loader.pose_task,
)
```

### Runners

The `deeplabcut.pose_estimations_pytorch.runners` contains code to get models, load 
pretrained weights, and either train them or run inference with them.

## Code Examples

### Training a Model on a COCO Dataset

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

### Running Video Analysis outside a DeepLabCut Project

DeepLabCut provides high-level APIs (via the GUI or the python package) to analyze your
data. The usage of this API assumes the existence of a DLC project (with `config.yaml`
file, etc.).

Sometimes it might be more convenient to just run a model on your data via a low-level
API. We also use this API under the hood, in particular for the Model Zoo. Check out the
example below:

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


### Running Top-Down Video Analysis with Existing Bounding Boxes

When `deeplabcut.pose_estimation_pytorch.apis.videos.video_inference` is called
with a top-down model, it is assumed that a detector snapshot is given as well to obtain
bounding boxes with which to run pose estimation. It's possible that you've already 
obtained bounding boxes for your video (with another object detector or through some 
other means), and you want to reuse those bounding boxes instead of running an object
detector again.

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
