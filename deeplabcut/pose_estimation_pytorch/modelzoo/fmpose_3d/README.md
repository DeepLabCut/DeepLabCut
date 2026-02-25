# FMPose3D

## Overview

**FMPose3D** is a monocular 3D pose estimation library that lifts 2D keypoints from images into 3D poses using *flow matching* — a generative modeling technique based on ODE sampling. It supports two pipelines:

- **Human pose estimation** — Uses HRNet + YOLO for 2D detection (17 H36M joints), then a flow-matching 3D lifter with optional flip test-time augmentation and camera-to-world transformation.
- **Animal pose estimation** — Uses DeepLabCut SuperAnimal for 2D detection (26 Animal3D joints), then a flow-matching 3D lifter with limb regularization post-processing.

Model weights are hosted on HuggingFace Hub and are downloaded automatically when no local path is provided. The library is installable via `pip install fmpose3d` and requires Python >= 3.8.

For a full overview and documentation on the API, see https://github.com/AdaptiveMotorControlLab/FMPose3D. 
