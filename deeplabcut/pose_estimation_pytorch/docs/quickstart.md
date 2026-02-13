# Quick Start

## Standard API (High-Level)

The high-level API remains unchanged. PyTorch models work seamlessly with existing DeepLabCut workflows:

```python
import deeplabcut

config = "/path/to/project/config.yaml"

# Standard workflow works with both engines
deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config, shuffle=1)
deeplabcut.evaluate_network(config, shuffle=[1])
deeplabcut.analyze_videos(config, videos=["/path/to/video.mp4"])
```

See the [standard guide](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide) for single-animal projects and [multi-animal guide](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide) for multi-animal tracking.

## Low-Level PyTorch API

For advanced usage, directly interact with the PyTorch backend:

```python
import deeplabcut.pose_estimation_pytorch as dlc_torch

# Create a data loader
loader = dlc_torch.DLCLoader(
    config="/path/to/project/config.yaml",
    trainset_index=0,
    shuffle=1,
)

# Train with custom configuration
dlc_torch.train(
    loader=loader,
    run_config=loader.model_cfg,
    task=loader.pose_task,
    device="cuda",
    max_epochs=200,
    batch_size=8,
)

# Run inference
predictions = dlc_torch.analyze_videos(
    config="/path/to/project/config.yaml",
    videos=["/path/to/video.mp4"],
    shuffle=1,
)
```

## Listing Available Models

Query supported architectures:

```python
from deeplabcut.pose_estimation_pytorch import available_models, available_detectors

# List all pose estimation models
models = available_models()
print(models)
# Output: ['dlcrnet_ms5', 'hrnet_w18', 'hrnet_w32', 'hrnet_w48', 'dekr_w32', ...]

# List all object detectors (for top-down models)
detectors = available_detectors()
print(detectors)
# Output: ['fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'yolox_s', ...]
```