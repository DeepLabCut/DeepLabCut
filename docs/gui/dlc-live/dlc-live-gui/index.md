# DeepLabCut Live GUI

A graphical application for **real-time pose estimation with DeepLabCut** using one or more cameras.

This GUI is designed for **scientists and experimenters** who want to preview, run inference, and record synchronized video with pose overlays—without writing code.

## Table of Contents

:::{toc}
:::

---

## What this software does

- **Live camera preview** from one or multiple cameras
- **Real-time pose inference** using DeepLabCut Live models
- **Multi-camera support** with tiled display
- **Video recording** (raw or with pose and bounding-box overlays)
- **Session-based data organization** with reproducible naming
- **Optional processor plugins** to extend behavior (e.g. remote control, triggers)

The application is built with **PySide6 (Qt)** and is intended for interactive experimental use rather than offline batch processing.

---

## Typical workflow

1. **Install** the application and required camera backends
2. **Configure cameras** (single or multi-camera)
3. **Select a DeepLabCut Live model**
4. **Start preview** and verify frame rate
5. **Run pose inference** on a selected camera
6. **Record video** (optionally with overlays)
7. **Organize results** by session and run

Each of these steps is covered in the *Quickstart* and *User Guide* sections of this documentation.

---

## Who this is for

- Neuroscience and behavior labs
- Experimentalists running real-time tracking
- Users who want a **GUI-first** workflow for DeepLabCut Live

You do **not** need to be a software developer to use this tool.

---

## What this documentation covers

- Installation and first-run setup
- Camera configuration and supported backends
- Pose inference settings and visualization
- Recording options and file organization
- Known limitations of the current release

This documentation intentionally focuses on **end-user operation**.
Developer-oriented material (APIs, internals, extension points) is out of scope for now.

---

## Current limitations (high-level)

Before getting started, be aware of the following constraints:

- Pose inference runs on **one selected camera at a time** (even in multi-camera mode)
- Camera synchronization depends on backend capabilities and hardware
- DeepLabCut Live models must be **exported and compatible** with the selected backend
- Performance depends on camera resolution, frame rate, GPU availability, and codec choice

A detailed and up-to-date list is maintained in the **Limitations** section.


---

## About DeepLabCut Live

DeepLabCut Live enables low-latency, real-time pose estimation using models trained with DeepLabCut.
This GUI provides an accessible interface on top of that ecosystem for interactive experiments.

---

*This project is under active development. Feedback from real experimental use is highly valued.*
