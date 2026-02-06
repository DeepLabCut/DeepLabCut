# DeepLabCut-live-GUI User Guide

Complete walkthrough for using the DeepLabCut-live-GUI application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Camera Setup](#camera-setup)
3. [DLCLive Configuration](#dlclive-configuration)
4. [Recording Videos](#recording-videos)
---

## Getting Started

### First Launch

1. Open a terminal/command prompt
2. Run the application:
   ```bash
   dlclivegui
   ```
3. The main window will appear with three control panels and a video display area

### Interface Overview

```
┌─────────────────────────────────────────────────────┐
│ File  Help                                          │
├─────────────┬───────────────────────────────────────┤
│ Camera      │                                       │
│ Settings    │                                       │
│             │                                       │
│ ─────────── │        Video Display                  │
│ DLCLive     │                                       │
│ Settings    │                                       │
│             │                                       │
│ ─────────── │                                       │
│ Recording   │                                       │
│ Settings    │                                       │
│             │                                       │
│ ─────────── │                                       │
│ Bounding    │                                       │
│ Box         │                                       │
│             │                                       │
│ ─────────── │                                       │
│ [Preview]   │                                       │
│ [Stop]      │                                       │
└─────────────┴───────────────────────────────────────┘
│ Status: Ready                                       │
└─────────────────────────────────────────────────────┘
```

---

## Camera Setup

### Step 1: Select Camera Backend

The **Backend** dropdown shows available camera drivers:

| Backend | When to Use |
|---------|-------------|
| **opencv** | Webcams, USB cameras (universal) |
| **gentl** | Industrial cameras (Windows/Linux) |
| **aravis** | GenICam/GigE cameras (Linux/macOS) |
| **basler** | Basler cameras specifically |

**Note**: Unavailable backends appear grayed out. Install required drivers to enable them.

### Step 2: Select Camera

1. Click **Refresh** next to the camera dropdown
2. Wait for camera detection (1-3 seconds)
3. Select your camera from the dropdown

The list shows camera details:
```
0:DMK 37BUX287 (26320523)
│ │             └─ Serial Number
│ └─ Model Name
└─ Index
```

### Step 3: Configure Camera Parameters

#### Frame Rate
- **Range**: 1-240 FPS (hardware dependent)
- **Recommendation**: Start with 30 FPS, increase as needed
- **Note**: Higher FPS = more processing load

#### Exposure Time
- **Auto**: Set to 0 (default)
- **Manual**: Microseconds (e.g., 10000 = 10ms)
- **Tips**:
  - Shorter exposure = less motion blur
  - Longer exposure = better low-light performance
  - Typical range: 5,000-30,000 μs

#### Gain
- **Auto**: Set to 0.0 (default)
- **Manual**: 0.0-100.0
- **Tips**:
  - Higher gain = brighter image but more noise
  - Start low (5-10) and increase if needed
  - Auto mode works well for most cases

#### Cropping (Optional)
Reduce frame size for faster processing:

1. Set crop region: (x0, y0, x1, y1)
   - x0, y0: Top-left corner
   - x1, y1: Bottom-right corner
2. Use Bounding Box visualization to preview
3. Set all to 0 to disable cropping

**Example**: Crop to center 640x480 region of 1280x720 camera:
```
x0: 320
y0: 120
x1: 960
y1: 600
```

#### Rotation
Select if camera is mounted at an angle:
- 0° (default)
- 90° (rotated right)
- 180° (upside down)
- 270° (rotated left)

### Step 4: Start Camera Preview

1. Click **Start Preview**
2. Video feed should appear in the display area
3. Check the **Throughput** metric below camera settings
4. Verify frame rate matches expected value

**Troubleshooting**:
- **No preview**: Check camera connection and permissions
- **Low FPS**: Reduce resolution or increase exposure time
- **Black screen**: Check exposure settings
- **Distorted image**: Verify backend compatibility

---

## DLCLive Configuration

### Prerequisites

1. Exported DLCLive model (see [DLC documentation](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/HelperFunctions.md#model-export-function))
2. DeepLabCut-live installed (`pip install deeplabcut-live`)
3. Camera preview running

### Step 1: Select Model

1. Click **Browse** next to "Model directory"
2. Navigate to your exported DLCLive model folder
3. Select the folder containing:
   - `pose_cfg.yaml`
   - Model weights (`.pb`, `.pth`, etc.)

### Step 2: Choose Model Type
We only support newer, pytorch based models.
- **PyTorch**: PyTorch-based models (requires PyTorch)


**Common options**:
- `processor`: "cpu" or "gpu"
- `resize`: Scale factor (0.5 = half size)
- `pcutoff`: Likelihood threshold
- `cropping`: Crop before inference

### Step 4: Select Processor (Optional)

If using custom pose processors:

1. Click **Browse** next to "Processor folder" (or use default)
2. Click **Refresh** to scan for processors
3. Select processor from dropdown
4. Processor will activate when inference starts

### Step 5: Start Inference

1. Ensure camera preview is running
2. Click **Start pose inference**
3. Button changes to "Initializing DLCLive!" (blue)
4. Wait for model loading (5-30 seconds)
5. Button changes to "DLCLive running!" (green)
6. Check **Performance** metrics

**Performance Metrics**:
```
150/152 frames | inference 42.1 fps | latency 23.5 ms (avg 24.1 ms) | queue 2 | dropped 2
```
- **150/152**: Processed/Total frames
- **inference 42.1 fps**: Processing rate
- **latency 23.5 ms**: Current processing delay
- **queue 2**: Frames waiting
- **dropped 2**: Skipped frames (due to full queue)

### Step 6: Enable Visualization (Optional)

Check **"Display pose predictions"** to overlay keypoints on video.

- Keypoints appear as green circles
- Updates in real-time with video
- Can be toggled during inference

---

## Recording Videos

### Basic Recording

1. **Configure output path**:
   - Click **Browse** next to "Output directory"
   - Select or create destination folder

2. **Set filename**:
   - Enter base filename (e.g., "session_001")
   - Extension added automatically based on container

3. **Select format**:
   - **Container**: mp4 (recommended), avi, mov
   - **Codec**:
     - `h264_nvenc` (NVIDIA GPU - fastest)
     - `libx264` (CPU - universal)
     - `hevc_nvenc` (NVIDIA H.265)

4. **Set quality** (CRF slider):
   - 0-17: Very high quality, large files
   - 18-23: High quality (recommended)
   - 24-28: Medium quality, smaller files
   - 29-51: Lower quality, smallest files

5. **Start recording**:
   - Ensure camera preview is running
   - Click **Start recording**
   - **Stop recording** button becomes enabled

6. **Monitor performance**:
   - Check "Performance" metrics
   - Watch for dropped frames
   - Verify write FPS matches camera FPS

### Advanced Recording Options

#### High-Speed Recording (60+ FPS)

**Settings**:
- Codec: `h264_nvenc` (requires NVIDIA GPU)
- CRF: 28 (higher compression)
- Crop region: Reduce frame size
- Close other applications

#### High-Quality Recording

**Settings**:
- Codec: `libx264` or `h264_nvenc`
- CRF: 18-20
- Full resolution
- Sufficient disk space


### Auto-Recording

Enable automatic recording triggered by processor events:

1. **Select a processor** that supports auto-recording
2. **Enable**: Check "Auto-record video on processor command"
3. **Start inference**: Processor will control recording
4. **Session management**: Files named by processor

---
## Next Steps

- Explore [Features Documentation](features.md) for detailed capabilities
- Review [Camera Backend Guide](camera_support.md) for advanced setup
- Check [Processor System](PLUGIN_SYSTEM.md) for custom processing
- See [Aravis Backend](aravis_backend.md) for Linux industrial cameras

---
