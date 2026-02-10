# Overview

DeepLabCut Live GUI is a **PySide6-based desktop application** for running real-time DeepLabCut pose estimation experiments with **one or multiple cameras**, optional **processor plugins**, and **video recording** (with or without overlays).

This page gives you a **guided tour of the main window**, explains the **core workflow**, and introduces the key concepts used throughout the user guide.

---

## Main window at a glance

When you launch the application (`dlclivegui`), you will see:

- A **Controls panel** (left) for configuring cameras, inference, recording, and overlays
- A **Video panel** (right) showing the live preview (single or tiled multi-camera)
- A **Stats area** (below the video) summarizing camera, inference, and recorder performance

---

## Intended workflow

On startup, the GUI is idle and waiting for you to configure cameras and settings,
as well as pick a model for pose inference.

To start running an experiment, the typical workflow is:

1. **Configure Cameras**
   Use **Configure Cameras…** to select one or more cameras and their parameters.
   <!-- TODO for more details see... -->

2. **Start Preview**
   Click **Start Preview** to begin streaming all selected configured cameras.
   - If multiple cameras are active, the preview becomes a **tiled view**.

3. *(If ready)* **Start Pose Inference**
   Choose a **Model file**, optionally a DLC-live **Processor**, select the **Inference Camera**, then click **Start pose inference**.
   <!-- - For more details about Processors see... -->
   - Toggle **Display pose predictions** to show or hide pose estimation overlays.

4. *(If ready)* **Start Recording**
   Choose an **Output directory**, session/run naming options, and encoding settings, then click **Start recording**.
   - Recording includes **all active cameras** in multi-camera mode in separate files.

5. **Stop**
   Use **Stop Preview**, **Stop pose inference**, and/or **Stop recording** as needed.

```{note}
Pose inference requires the camera preview to be running.
If you start pose inference while preview is stopped, the GUI will automatically start the preview first.
```

---

## Main control panel

:::{figure} ../_static/images/main_window_100226.png
:label: fig:main_window_startup
:alt: Screenshot of the main window
:width: 100%
:align: center

   The main window on startup, showing the Controls panel (left), Video panel (right), and Stats area (below video).
:::

### Camera settings

**Purpose:** Define which cameras are available and active.

- **Configure Cameras…**
  Opens the camera configuration dialog where you can:
  - Add, enable, or disable cameras
  - Select backend and index
  - Adjust camera-specific properties
  - Switch between single- and multi-camera setups

```{important}
Depending on the system, backend and camera model,
settings may vary widely between proper support, partial support, or no support at all.
This is especially true for the generalist OpenCV backend, which may work well with some cameras but not others.
```

- **Active**
  Displays a summary of configured cameras:
  - **Single camera:** `Name [backend:index] @ fps`
  - **Multiple cameras:** `N cameras: camA, camB, …`

```{important}
In multi-camera mode, pose inference runs on **one selected camera at a time** (the *Inference Camera*),
even though preview and recording may include multiple cameras.
```

---

### DLCLive settings

**Purpose:** Configure and run pose inference on the live stream.

- **Model file**
  Path to an exported DeepLabCut-Live model file (e.g. `.pt`, `.pb`).

- **Processor folder / Processor** *(optional)*
  Processor plugins extend functionality (for example, experiment logic or external control).
  This allows for e.g. closed-loop control of devices based on pose estimation results.

- **Inference Camera**
  Select which active camera is used for pose inference.
  In multi-camera preview, pose overlays are drawn only on the corresponding tile.

- **Start pose inference / Stop pose inference**
  The button indicates inference state:
  - *Initializing DLCLive!* → Model loading
  - *DLCLive running!* → Inference active

- **Display pose predictions**
  Toggle visualization of predicted keypoints.

- **Auto-record video on processor command** *(optional)*
  Allows compatible processors to start and stop recording automatically.

- **Processor Status**
  Displays processor-specific status information when available.

---

### Recording

**Purpose:** Save videos from active cameras !

Core settings:
- **Output directory**: Base directory for all recordings
- **Session name**: Logical grouping of runs (e.g. `mouseA_day1`)
- **Use timestamp for run folder name**:
  - Enabled → `run_YYYYMMDD_HHMMSS_mmm`
  - Disabled → `run_0001`, `run_0002`, …

A live preview label shows the *approximate* output path, including camera placeholders.

Encoding options:
- **Container** (e.g. `mp4`, `avi`, `mov`)
- **Codec** (availability depends on OS and hardware)
- **CRF** (quality/compression tradeoff; lower values = higher quality)

Additional controls:
- **Record video with overlays**
  Include pose predictions and/or bounding boxes directly in the recorded video.
  :::{caution}
  This **cannot be easily undone** once the recording is saved.<br>
  Use with caution if you want to preserve raw footage.
  :::
- **Start recording / Stop recording**
- **Open recording folder**

---

### Bounding Box Visualization

**Purpose:** Show a bounding box around the detected subject.

- **Show bounding box**: Enable or disable overlay
- **Coordinates**: `x0`, `y0`, `x1`, `y1`

In multi-camera mode, the bounding box is applied relative to the **inference camera tile**, ensuring correct alignment in tiled previews.

---

## Video Panel and Stats

### Video preview

- Displays a logo screen when idle
- Shows live video when preview is running
- Uses a tiled layout automatically when multiple cameras are active

### Stats panel

Three continuously updated sections:

- **Camera**: Per-camera measured frame rate
- **DLC Processor**: Inference throughput, latency, queue depth, and dropped frames
- **Recorder**: Recording status and write performance

```{tip}
Stats text can be selected and copied directly, which is useful for debugging or reporting performance issues.
```

---

## Menus

### File
- Load configuration…
- Save configuration
- Save configuration as…
- Open recording folder
- Close window

### View → Appearance
- System theme
- Dark theme

---

## Configuration and Persistence

The GUI can restore settings across sessions using:

- Explicitly saved **JSON configuration files**
- A stored snapshot of the most recent configuration
- Remembered paths (e.g. last-used model directory)

```{tip}
For reproducible experiments, prefer saving and versioning JSON configuration files
rather than relying only on remembered application state.
```

---

## Next Steps

- **Camera Setup**: Add and validate cameras, start preview
- **Inference Setup**: Select a model, start pose inference, interpret performance metrics
- **First Recording**: Understand session/run structure and verify recorded output
