# Data Preparation and Loading

The [`deeplabcut.pose_estimation_pytorch.data`][] package provides comprehensive tools for dataset creation, train/test splitting, and data augmentation. This guide covers the data loaders, datasets, and transforms available in the PyTorch backend.

## Data Loaders

DeepLabCut provides two main data loader classes for different dataset formats:

### DLCLoader

The [`DLCLoader`][deeplabcut.pose_estimation_pytorch.data.DLCLoader] class loads labeled data from a DeepLabCut project. It handles train/test splitting, configuration loading, and dataset creation for a specific shuffle.

**Basic Usage:**

```python
import deeplabcut.pose_estimation_pytorch as dlc_torch

loader = dlc_torch.DLCLoader(
    config="/path/to/project/config.yaml",
    trainset_index=0,
    shuffle=1,
)

# Access loader properties
print(loader.model_folder)       # Path to model directory
print(loader.evaluation_folder)  # Path to evaluation directory
print(loader.pose_task)          # Task type (BOTTOM_UP, TOP_DOWN, etc.)

# View the data
print(loader.df)        # Complete dataset as DataFrame
print(loader.df_train)  # Training split
print(loader.df_test)   # Test split
```

**Creating Datasets:**

```python
# Create training dataset
train_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["train"]),
    mode="train",
    task=loader.pose_task,
)

# Create validation dataset
valid_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["inference"]),
    mode="test",
    task=loader.pose_task,
)

# Check dataset size
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
```

### COCOLoader

The [`COCOLoader`][deeplabcut.pose_estimation_pytorch.data.COCOLoader] class enables training on datasets in COCO format without creating a DeepLabCut project. This is useful for working with public datasets or custom data pipelines.

**COCO Dataset Structure:**

```
COCOProject/
├── annotations/
│   ├── train.json
│   └── test.json
└── images/
    ├── img0000.png
    ├── img0001.png
    └── ...
```

**Working with COCO Data:**

```python
from pathlib import Path
import deeplabcut.pose_estimation_pytorch as dlc_torch

project_root = Path("/path/to/COCOProject")

# Parse dataset information
train_dict = dlc_torch.COCOLoader.load_json(
    project_root,
    filename="train.json"
)
max_num_individuals, bodyparts = dlc_torch.COCOLoader.get_project_parameters(train_dict)

# Create model configuration
model_cfg = dlc_torch.config.make_pytorch_pose_config(
    project_config=dlc_torch.config.make_basic_project_config(
        dataset_path=str(project_root),
        bodyparts=bodyparts,
        max_individuals=max_num_individuals,
        multi_animal=True,
    ),
    pose_config_path=project_root / "experiments" / "hrnet_w32" / "train",
    net_type="hrnet_w32",
    top_down=True,
    save=True,
)

# Create loader
loader = dlc_torch.COCOLoader(
    project_root=project_root,
    model_config_path=project_root / "experiments" / "hrnet_w32" / "train" / "pytorch_cfg.yaml",
    train_json_filename="train.json",
    test_json_filename="test.json",
)

# Create datasets
train_dataset = loader.create_dataset(
    transform=dlc_torch.build_transforms(loader.model_cfg["data"]["train"]),
    mode="train",
    task=loader.pose_task,
)
```

**Image Path Resolution:**

COCO JSON files can specify image paths in two ways:

1. **Relative paths**: Resolved to the `images/` folder
   - `"file_name": "img0000.png"` → `/path/to/COCOProject/images/img0000.png`
   - `"file_name": "subfolder/img0000.png"` → `/path/to/COCOProject/images/subfolder/img0000.png`

2. **Absolute paths**: Used directly without resolution
   - `"file_name": "/data/disk2/images/img0000.png"` → `/data/disk2/images/img0000.png`

This allows you to keep images on different disks or reuse images across projects without duplication.

## PoseDataset

The [`PoseDataset`][deeplabcut.pose_estimation_pytorch.data.PoseDataset] class extends `torch.utils.data.Dataset` and converts raw images and keypoints into tensors for training and evaluation.

- Loads images and annotations
- Applies data augmentation transforms
- Generates training targets using the model's target generator
- Handles multi-animal and single-animal data
- Supports dynamic cropping for top-down models
