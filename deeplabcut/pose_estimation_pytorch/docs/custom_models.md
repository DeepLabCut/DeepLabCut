# Adding Custom Models

DeepLabCut's PyTorch backend uses a registry system that makes it easy to add custom model components. This guide explains how to extend the framework with your own backbones, necks, heads, predictors, and target generators.

## Model Registry System

The registry system allows you to register custom components that can be instantiated from configuration files. Five registries are available for different model building blocks:

- [`BACKBONES`][deeplabcut.pose_estimation_pytorch.models.backbones] - Feature extraction networks
- [`NECKS`][deeplabcut.pose_estimation_pytorch.models.necks] - Intermediate processing layers
- [`HEADS`][deeplabcut.pose_estimation_pytorch.models.heads] - Task-specific output layers
- [`PREDICTORS`][deeplabcut.pose_estimation_pytorch.models.predictors] - Output-to-keypoint converters
- [`TARGET_GENERATORS`][deeplabcut.pose_estimation_pytorch.models.target_generators] - Annotation-to-target converters