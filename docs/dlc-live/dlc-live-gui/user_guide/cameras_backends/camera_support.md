(file:dlclivegui-camera-support)=
# Camera Support

DeepLabCut-live-GUI supports multiple camera backends for different platforms and camera types:

## Supported Backends

1. **OpenCV** - "Universal" webcam and USB camera support *(all platforms)*
   - Expect some limitations in camera control and performance
2. **GenTL** - Industrial cameras via GenTL producers *(Windows, Linux)*
3. **Aravis** - GenICam/GigE Vision cameras *(Linux, experimental on macOS)*
4. **Basler** - Basler cameras via pypylon *(all platforms)*

## Backend Selection

You can select the backend in the GUI from the "Backend" dropdown, or in your configuration file:

```json
{
  ...
  "camera": {
    ...
    "backend": "aravis",
    ...
  }
}
```

## Platform-Specific Recommendations

### Windows

- **OpenCV compatible cameras**: Best for webcams and simple USB cameras. OpenCV is installed with DeepLabCut-live-GUI.
- **GenTL backend**: Recommended for industrial cameras (The Imaging Source, Basler, etc.) via vendor-provided CTI files.
- **Basler cameras**: Can use either GenTL or pypylon backend.

### Linux

- **OpenCV compatible cameras**: Good for webcams via Video4Linux drivers. Installed with DeepLabCut-live-GUI.
- **Aravis backend**: **Recommended** for GenICam/GigE Vision industrial cameras (The Imaging Source, Basler, Point Grey, etc.)
  - Easy installation via system package manager
  - Better Linux support than GenTL
  - See {ref}`file:dlclivegui-camera-aravis-backend` for details and troubleshooting.
- **GenTL backend**: Alternative for industrial cameras if vendor provides Linux CTI files.

### macOS

- **OpenCV compatible cameras**: For webcams and compatible USB cameras.
- **Aravis backend**: Experimental support for GenICam/GigE Vision cameras.
  Requires Homebrew and PyGObject; functionality depends heavily on camera model and setup.


## Quick Installation Guide

### Aravis (Linux/Ubuntu)
```bash
sudo apt-get install gir1.2-aravis-0.8 python3-gi
```

### Aravis (macOS)
```bash
brew install aravis
pip install pygobject
```

### GenTL (Windows)
Install vendor-provided camera drivers and SDK. CTI files are typically in:
- `C:\Program Files\The Imaging Source Europe GmbH\IC4 GenTL Driver\bin\`

## Backend Comparison

| Feature | OpenCV | GenTL | Aravis | Basler (pypylon) |
|---------|--------|-------|--------|------------------|
| Exposure control | No | Yes | Yes | Yes |
| Gain control | No | Yes | Yes | Yes |
| Windows | ✅ | ✅ | ❌ | ✅ |
| Linux | ✅ | ✅ | ✅ | ✅ |
| macOS | ✅ | ❌ | ⚠️ | ✅ |

## Detailed Backend Documentation

- OpenCV - *"Universal" webcam support - Expect some limitations in camera control and performance*
- {doc}`Aravis <aravis_backend>` - *GenICam/GigE cameras on Linux/macOS*
- **To be added** GenTL - *Industrial cameras via vendor CTI files*
- **To be added** Basler - *Basler cameras via pypylon (cross-platform)*
