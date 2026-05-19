# Adding Custom Models

DeepLabCut's PyTorch backend uses a registry system that makes it easy to add custom model components. This guide explains how to extend the framework with your own backbones, necks, heads, predictors, and target generators.

## Model Registry System

The registry system allows you to register custom components that can be instantiated from configuration files. Five registries are available for different model building blocks:

- [`BACKBONES`][deeplabcut.pose_estimation_pytorch.models.backbones] - Feature extraction networks
- [`NECKS`][deeplabcut.pose_estimation_pytorch.models.necks] - Intermediate processing layers
- [`HEADS`][deeplabcut.pose_estimation_pytorch.models.heads] - Task-specific output layers
- [`PREDICTORS`][deeplabcut.pose_estimation_pytorch.models.predictors] - Output-to-keypoint converters
- [`TARGET_GENERATORS`][deeplabcut.pose_estimation_pytorch.models.target_generators] - Annotation-to-target converters

## Custom Backbones

A backbone extracts features from input images. Subclass [`BaseBackbone`][deeplabcut.pose_estimation_pytorch.models.backbones.BaseBackbone] and decorate with `@BACKBONES.register_module`:

```python
import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone


@BACKBONES.register_module
class MyBackbone(BaseBackbone):
    def __init__(self, out_channels: int = 256, pretrained: bool = False):
        # stride is the total downsampling factor of the network
        super().__init__(stride=32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)
        # ... rest of architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x
```

Use the custom backbone in a configuration file:

```yaml
model:
  backbone:
    type: MyBackbone
    out_channels: 256
    pretrained: false
```

## Custom Necks

A neck processes the backbone's feature map before it reaches the head. Subclass [`BaseNeck`][deeplabcut.pose_estimation_pytorch.models.necks.BaseNeck] and register with `@NECKS.register_module`:

```python
import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.models.necks import NECKS, BaseNeck


@NECKS.register_module
class MyNeck(BaseNeck):
    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)
```

Reference in configuration:

```yaml
model:
  neck:
    type: MyNeck
    in_channels: 256
    out_channels: 128
```

## Custom Heads

A head produces task-specific predictions from the neck's or backbone's output. Subclass [`BaseHead`][deeplabcut.pose_estimation_pytorch.models.heads.BaseHead] and register with `@HEADS.register_module`:

```python
import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.models.heads import HEADS, BaseHead
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator


@HEADS.register_module
class MyHead(BaseHead):
    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict,
        aggregator=None,
        in_channels: int = 256,
        num_keypoints: int = 17,
    ):
        super().__init__(
            predictor=predictor,
            target_generator=target_generator,
            criterion=criterion,
            aggregator=aggregator,
        )
        self.conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        heatmap = self.conv(x)
        return {"heatmap": heatmap}
```

## Custom Predictors

A predictor converts a head's raw output tensors into keypoint coordinates. Subclass [`BasePredictor`][deeplabcut.pose_estimation_pytorch.models.predictors.BasePredictor] and register with `@PREDICTORS.register_module`:

```python
import torch
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS, BasePredictor


@PREDICTORS.register_module
class MyPredictor(BasePredictor):
    def forward(
        self,
        stride: float,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        heatmap = outputs["heatmap"]
        # Locate the maximum value in each heatmap channel
        batch, num_kpts, h, w = heatmap.shape
        flat = heatmap.flatten(2)
        idx = flat.argmax(dim=-1)
        x = (idx % w).float() * stride
        y = (idx // w).float() * stride
        coords = torch.stack([x, y], dim=-1)  # (batch, num_kpts, 2)
        scores = flat.max(dim=-1).values
        return {"coordinates": coords, "scores": scores}
```

## Custom Target Generators

A target generator converts ground-truth annotations into training target tensors. Subclass [`BaseGenerator`][deeplabcut.pose_estimation_pytorch.models.target_generators.BaseGenerator] and register with `@TARGET_GENERATORS.register_module`:

```python
import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    TARGET_GENERATORS,
    BaseGenerator,
)


@TARGET_GENERATORS.register_module
class MyTargetGenerator(BaseGenerator):
    def __init__(self, num_heatmaps: int, sigma: float = 2.0):
        super().__init__()
        self.num_heatmaps = num_heatmaps
        self.sigma = sigma

    def forward(
        self,
        annotations: dict,
        stride: float,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        # Build Gaussian heatmaps for each keypoint annotation
        # Returns a dict matching the keys produced by the head
        ...
        return {"heatmap": target_heatmaps, "heatmap_mask": mask}
```

## Using Custom Components in Configuration

Once registered, custom components are referenced by class name in `pytorch_cfg.yaml`. Make sure the module containing your class is imported before the configuration is loaded (e.g. at the top of your training script):

```python
# my_components.py — import this before calling dlc_torch.train()
from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone
import torch.nn as nn


@BACKBONES.register_module
class MyBackbone(BaseBackbone):
    ...
```

Then reference it in the configuration:

```yaml
model:
  backbone:
    type: MyBackbone   # matched by class name in the registry
    out_channels: 256
  head:
    type: HeatmapHead  # built-in head, paired with your backbone
    ...
```
