# DeepLabCut Live GUI

A graphical application for **real-time pose estimation with DeepLabCut** using one or more cameras.

This GUI is designed for **scientists and experimenters** who want to preview, run inference, and record synchronized video with pose overlays—without writing code.

## Table of Contents

- {doc}`Installation <./quickstart/install>`
- {doc}`Overview <./user_guide/overview>`
- {doc}`Camera setup and backends <./user_guide/cameras_backends/camera_support>`

---

## Description

### What this software does

- **Live camera preview** from one or multiple cameras
- **Real-time pose inference** using DeepLabCut Live models
- **Multi-camera support** with tiled display
- **Video recording** (raw or with pose and bounding-box overlays)
- **Session-based data organization** with reproducible naming
- **Optional processor plugins** to extend behavior (e.g. remote control, triggers)

The application is built with **PySide6 (Qt)** and is intended for interactive experimental use rather than offline batch processing.
<!-- TODO mention and link to DLC live standalone -->

### Typical workflow

1. **Install** the application and required camera backends
2. **Configure cameras** (single or multi-camera)
3. **Select a DeepLabCut Live model**
4. **Start preview** and verify frame rate
5. **Run pose inference** on a selected camera
6. **Record video** (optionally with overlays)
   - With **organized results** by session and run

Each of these steps is covered in the *{doc}`Quickstart <quickstart/install>`*
and *{doc}`User Guide <user_guide/overview>`* sections of this documentation.

### Who this is for

- Neuroscience and behavior labs
- Experimentalists running real-time tracking
- Anyone who wants a **GUI-first** workflow for DeepLabCut Live

---

## Current limitations

Before getting started, be aware of the following constraints:

- Pose inference runs on **one selected camera at a time** (even in multi-camera mode)
- Camera synchronization depends on backend capabilities and hardware
  - OpenCV controls for resolution and FPS are "best effort" and may not work with all cameras.
    Expect inconsistencies when setting certain frame rates or resolutions as resolution depends on the device driver.
- DeepLabCut Live models must be **exported and compatible** with the selected backend
  - Some SuperAnimal models from {ref}`file:model-zoo` may not work out of the box.<br>This is currently the case for:
    - SuperHuman model (missing detector)
- Performance depends on camera resolution, frame rate, GPU availability, and codec choice

---

## About

*This project is under active development. Feedback from real experimental use is highly valued.*

Please report issues, suggest features, or contribute to the codebase on GitHub:

- [DLC-Live! GUI](https://github.com/DeepLabCut/DeepLabCut-live-GUI)
