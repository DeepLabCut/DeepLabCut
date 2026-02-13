# Model Architecture

The [`deeplabcut.pose_estimation_pytorch.models`][] package provides a modular framework for building pose estimation models. Models are composed of several key components that can be mixed and matched to create custom architectures.

## Model Components

DeepLabCut models are built from the following modular components:

### Backbones

Backbones are feature extraction networks that process input images and produce multi-scale feature representations. The [`deeplabcut.pose_estimation_pytorch.models.backbones`][] module provides several state-of-the-art backbone architectures.

**Available Backbones:**

- **ResNet** (ResNet-50, ResNet-101): Deep residual networks
- **HRNet** (HRNet-W18, HRNet-W32, HRNet-W48): High-resolution networks that maintain high-resolution representations throughout
- **CSPNext** (CSPNext-S, CSPNext-M, CSPNext-X): Cross Stage Partial networks for efficient feature extraction

All backbones inherit from [`BaseBackbone`][deeplabcut.pose_estimation_pytorch.models.backbones.BaseBackbone] and must define a stride property indicating the downsampling factor.

**Example:**
```python
from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES

# Build a backbone from configuration
backbone_config = dict(type="HRNet", variant="w32")
backbone = BACKBONES.build(backbone_config)
```

### Necks

Necks are optional intermediate layers between the backbone and head. They typically perform feature aggregation, refinement, or transformation. The [`deeplabcut.pose_estimation_pytorch.models.necks`][] module provides various neck architectures.

All necks inherit from [`BaseNeck`][deeplabcut.pose_estimation_pytorch.models.necks.BaseNeck].

### Heads

Heads are task-specific output layers that produce the final predictions. The [`deeplabcut.pose_estimation_pytorch.models.heads`][] module contains various head architectures for different pose estimation approaches.

Each head contains:
- A **predictor** to convert model outputs to keypoint locations
- A **target generator** to create training targets from annotations
- A **criterion** to compute the loss
- An **aggregator** to combine multiple losses

All heads inherit from [`BaseHead`][deeplabcut.pose_estimation_pytorch.models.heads.BaseHead] and output a dictionary mapping output names to tensors.

**Example:**
```python
from deeplabcut.pose_estimation_pytorch.models.heads import HEADS

head_config = dict(
    type="HeatmapHead",
    predictor=dict(type="HeatmapPredictor", location_refinement=True, locref_std=7.2801),
    target_generator=dict(
        type="HeatmapGaussianGenerator",
        num_heatmaps=17,
        pos_dist_thresh=17,
        generate_locref=True,
    ),
    criterion=dict(type="WeightedMSECriterion"),
    aggregator=None,
    heatmap_config=dict(
        num_input_channels=256,
        num_output_channels=17,
        num_deconv_layers=1,
    ),
)
head = HEADS.build(head_config)
```

### Predictors

Predictors transform model outputs into final keypoint predictions. The [`deeplabcut.pose_estimation_pytorch.models.predictors`][] module provides various prediction strategies.

**Available Predictors:**

- [`HeatmapPredictor`][deeplabcut.pose_estimation_pytorch.models.predictors.HeatmapPredictor]: Extract keypoints from heatmaps
- [`DEKRPredictor`][deeplabcut.pose_estimation_pytorch.models.predictors.DEKRPredictor]: DEKR-style predictions with offsets
- [`SimCCPredictor`][deeplabcut.pose_estimation_pytorch.models.predictors.SimCCPredictor]: SimCC coordinate classification
- [`PartAffinityFieldPredictor`][deeplabcut.pose_estimation_pytorch.models.predictors.PartAffinityFieldPredictor]: Bottom-up PAF-based assembly

All predictors inherit from [`BasePredictor`][deeplabcut.pose_estimation_pytorch.models.predictors.BasePredictor].

### Target Generators

Target generators create training targets from ground truth annotations. The [`deeplabcut.pose_estimation_pytorch.models.target_generators`][] module provides generators for different output types.

All target generators inherit from [`BaseGenerator`][deeplabcut.pose_estimation_pytorch.models.target_generators.BaseGenerator].