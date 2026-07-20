# DeepLabCut PyTorch API Documentation

<div align="center">

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1628250004229-KVYD7JJVHYEFDJ32L9VJ/DLClogo2021.jpg?format=1000w" width="95%">
</p>

</div>

[![PyPI](https://img.shields.io/pypi/v/deeplabcut?label=PyPI)](https://pypi.org/project/deeplabcut)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deeplabcut)
[![License](https://img.shields.io/github/license/DeepLabCut/DeepLabCut)](https://github.com/DeepLabCut/DeepLabCut/blob/main/LICENSE)

This documentation is designed for maintainers, developers, and expert users who want to understand and extend the PyTorch backend of DeepLabCut 3.0+. It provides detailed information about the architecture, APIs, and practical examples for building and training state-of-the-art pose estimation models.

## Overview

The [`deeplabcut.pose_estimation_pytorch`][] package provides a complete framework for training and deploying deep learning models for pose estimation. The API is designed to be modular, flexible, and extensible, allowing developers to easily add new models, customize training pipelines, and integrate with existing workflows.

## Core Components

The PyTorch DeepLabCut codebase is organized into four main components:

### High-Level APIs

The [`deeplabcut.pose_estimation_pytorch.apis`][] module contains high-level methods for training, evaluation, and inference. These methods work seamlessly with DeepLabCut projects but can also be used independently for maximum flexibility.

Key functions include:

- [`train`][deeplabcut.pose_estimation_pytorch.apis.train] - Train pose estimation models
- [`evaluate`][deeplabcut.pose_estimation_pytorch.apis.evaluate] - Evaluate model performance
- [`analyze_videos`][deeplabcut.pose_estimation_pytorch.apis.analyze_videos] - Run inference on video files
- [`analyze_images`][deeplabcut.pose_estimation_pytorch.apis.analyze_images] - Run inference on image files
- [`video_inference`][deeplabcut.pose_estimation_pytorch.apis.video_inference] - Low-level video inference API

### Models

The [`deeplabcut.pose_estimation_pytorch.models`][] package provides state-of-the-art pose estimation architectures including DLCRNet, HRNet, DEKR, BUCTD, and RTMPose. Models are built from modular components:

- **Backbones**: Feature extraction networks like ResNet and HRNet (see [`deeplabcut.pose_estimation_pytorch.models.backbones`][])
- **Necks**: Optional intermediate layers between backbone and head (see [`deeplabcut.pose_estimation_pytorch.models.necks`][])
- **Heads**: Task-specific output layers for pose prediction (see [`deeplabcut.pose_estimation_pytorch.models.heads`][])
- **Predictors**: Convert model outputs to keypoint locations (see [`deeplabcut.pose_estimation_pytorch.models.predictors`][])
- **Target Generators**: Create training targets from annotations (see [`deeplabcut.pose_estimation_pytorch.models.target_generators`][])

Object detection models for top-down pose estimation are available in [`deeplabcut.pose_estimation_pytorch.models.detectors`][].

You can check available models programmatically:

```python
import deeplabcut.pose_estimation_pytorch as dlc_torch

# List available pose estimation models
print(dlc_torch.available_models())

# List available object detection models
print(dlc_torch.available_detectors())
```

### Data Loading

The [`deeplabcut.pose_estimation_pytorch.data`][] package handles dataset creation, train/test splitting, and data augmentation. Two main data loaders are provided:

- [`DLCLoader`][deeplabcut.pose_estimation_pytorch.data.DLCLoader] - Load data from DeepLabCut projects
- [`COCOLoader`][deeplabcut.pose_estimation_pytorch.data.COCOLoader] - Load data in COCO format

The [`PoseDataset`][deeplabcut.pose_estimation_pytorch.data.PoseDataset] class extends `torch.utils.data.Dataset` to provide tensor-based datasets for training and evaluation.

### Training and Inference Runners

The [`deeplabcut.pose_estimation_pytorch.runners`][] module provides classes for model training and inference:

- [`PoseTrainingRunner`][deeplabcut.pose_estimation_pytorch.runners.PoseTrainingRunner] - Handles the training loop
- [`PoseInferenceRunner`][deeplabcut.pose_estimation_pytorch.runners.PoseInferenceRunner] - Runs pose estimation inference
- [`DetectorInferenceRunner`][deeplabcut.pose_estimation_pytorch.runners.DetectorInferenceRunner] - Runs object detection for top-down models
