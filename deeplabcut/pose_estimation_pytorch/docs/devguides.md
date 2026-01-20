# Developer Guides

This section provides comprehensive guides for developers working with the DeepLabCut PyTorch backend. Whether you're training models, running inference, or building custom architectures, these guides will help you understand and extend the framework.

## Guide Overview

### Model Architecture

The [Model Architecture](models.md) guide explains the modular design of DeepLabCut models. Learn about:

- **Backbones**: Feature extraction networks (ResNet, HRNet, CSPNext)
- **Necks**: Optional intermediate processing layers
- **Heads**: Task-specific output layers for pose estimation
- **Predictors**: Converting model outputs to keypoint predictions
- **Target Generators**: Creating training targets from annotations
- **Available Models**: Pre-configured architectures for various tasks

This guide is essential for understanding how components work together to form complete pose estimation systems.

### Configuration Files

The [Configuration Files](configuration.md) guide covers model and training configuration. Topics include:

- Creating configuration files with [`make_pytorch_pose_config`][deeplabcut.pose_estimation_pytorch.config.make_pytorch_pose_config]
- YAML configuration structure and parameters
- Model architecture specification
- Training hyperparameters (optimizer, scheduler, batch size)
- Data augmentation settings
- Reading and writing configurations
- Using configuration templates

Understanding configuration is crucial for customizing model behavior and training settings.

### Data Preparation

The [Data Preparation](dataprep.md) guide explains data loading and augmentation. Learn about:

- [`DLCLoader`][deeplabcut.pose_estimation_pytorch.data.DLCLoader] for DeepLabCut projects
- [`COCOLoader`][deeplabcut.pose_estimation_pytorch.data.COCOLoader] for COCO-format datasets
- [`PoseDataset`][deeplabcut.pose_estimation_pytorch.data.PoseDataset] for PyTorch data loading
- Data augmentation transforms
- Train/test splitting
- Custom collate functions for multi-animal scenarios
- Snapshot management

Proper data preparation is key to training robust models.

### Training Models

The [Training Models](training.md) guide covers the complete training workflow. Topics include:

- High-level training APIs: [`train`][deeplabcut.pose_estimation_pytorch.apis.train] and [`train_network`][deeplabcut.pose_estimation_pytorch.apis.train_network]
- Training on DeepLabCut projects and COCO datasets
- [`PoseTrainingRunner`][deeplabcut.pose_estimation_pytorch.runners.PoseTrainingRunner] for custom training loops
- Optimizers and learning rate schedulers
- Checkpointing and resuming training
- Logging with Weights & Biases
- Training top-down models (detector and pose estimation)
- Distributed training across multiple GPUs
- Performance optimization tips

This guide helps you train models efficiently and monitor progress effectively.

### Inference and Analysis

The [Inference & Analysis](inference.md) guide explains how to run predictions. Learn about:

- High-level APIs: [`analyze_videos`][deeplabcut.pose_estimation_pytorch.apis.analyze_videos] and [`analyze_images`][deeplabcut.pose_estimation_pytorch.apis.analyze_images]
- Low-level [`video_inference`][deeplabcut.pose_estimation_pytorch.apis.video_inference] API
- [`PoseInferenceRunner`][deeplabcut.pose_estimation_pytorch.runners.PoseInferenceRunner] for pose estimation
- [`DetectorInferenceRunner`][deeplabcut.pose_estimation_pytorch.runners.DetectorInferenceRunner] for object detection
- [`VideoIterator`][deeplabcut.pose_estimation_pytorch.apis.VideoIterator] for efficient video processing
- Running inference outside DeepLabCut projects
- Using custom bounding boxes with top-down models
- Model evaluation and metrics
- Visualization and labeled video creation
- Batch processing and performance optimization

Master inference to efficiently analyze your data and evaluate model performance.

### Adding Custom Models

The [Adding Custom Models](custom_models.md) guide shows how to extend the framework. Topics include:

- The model registry system (BACKBONES, NECKS, HEADS, PREDICTORS, TARGET_GENERATORS)
- Creating custom backbones for feature extraction
- Implementing custom heads for new output types
- Building custom predictors for keypoint extraction
- Designing custom target generators for training
- Creating custom necks for feature processing
- Using custom components in configuration files
- Best practices for extensibility

This guide empowers you to implement novel architectures and techniques.

## Core Concepts

### Modular Design

DeepLabCut PyTorch uses a modular architecture where models are composed of interchangeable components.

### Registry System

Components are registered using decorators, making them available for instantiation from configuration files:

```python
from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone

@BACKBONES.register_module
class MyBackbone(BaseBackbone):
    def __init__(self, ...):
        super().__init__(stride=32)
        # Implementation
    
    def forward(self, x):
        # Forward pass
        return x
```

This pattern is used for all model components (backbones, necks, heads, predictors, target generators).

### Configuration-Driven

Models and training are controlled through YAML configuration files:

```yaml
model:
  backbone:
    type: "HRNet"
    variant: "w32"
  head:
    type: "HeatmapHead"
    num_bodyparts: 17

train:
  optimizer:
    type: "AdamW"
    lr: 0.0001
  epochs: 200
```

This approach separates architecture definition from code, making experimentation easier.

### Task-Based Design

Different pose estimation approaches are handled through the [`Task`][deeplabcut.pose_estimation_pytorch.task.Task] enum:

- `BOTTOM_UP`: Detect all keypoints then group into individuals
- `TOP_DOWN`: Detect individuals then estimate pose for each
- `COND_TOP_DOWN`: Conditional top-down approach

The same APIs work across tasks, with the framework handling task-specific logic internally.

## API Patterns

### High-Level vs Low-Level APIs

DeepLabCut provides both high-level and low-level APIs:

- **High-Level**: Simple functions for common tasks (training, video analysis)
- **Low-Level**: Detailed control through runners and components

Start with high-level APIs and move to low-level when you need more control.

### Loaders

Data loaders abstract dataset access:

- Use [`DLCLoader`][deeplabcut.pose_estimation_pytorch.data.DLCLoader] for standard DeepLabCut projects
- Use [`COCOLoader`][deeplabcut.pose_estimation_pytorch.data.COCOLoader] for COCO-format data
- Both provide consistent interfaces for dataset creation

### Runners

Runners encapsulate training and inference logic:

- [`PoseTrainingRunner`][deeplabcut.pose_estimation_pytorch.runners.PoseTrainingRunner]: Manages training loops
- [`PoseInferenceRunner`][deeplabcut.pose_estimation_pytorch.runners.PoseInferenceRunner]: Handles pose estimation
- [`DetectorInferenceRunner`][deeplabcut.pose_estimation_pytorch.runners.DetectorInferenceRunner]: Handles object detection

Runners can be created from configuration or instantiated directly for custom workflows.