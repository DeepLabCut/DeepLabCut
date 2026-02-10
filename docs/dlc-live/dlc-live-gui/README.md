# DeepLabCut-live-GUI Documentation Index

Welcome to the DeepLabCut-live-GUI documentation! This index will help you find the information you need.

## Getting Started

### New Users
1. **[README](../README.md)** - Project overview, installation, and quick start
2. **[User Guide](user_guide.md)** - Step-by-step walkthrough of all features
3. **[Installation Guide](install.md)** - Detailed installation instructions

### Quick References
- **[ARAVIS_QUICK_REF](../ARAVIS_QUICK_REF.md)** - Aravis backend quick reference
- **[Features Overview](features.md)** - Complete feature documentation

## Core Documentation

### Camera Setup
- **[Camera Support](camera_support.md)** - Overview of all camera backends
- **[Aravis Backend](aravis_backend.md)** - Linux/macOS GenICam camera setup
- Platform-specific guides for industrial cameras

### Application Features
- **[Features Documentation](features.md)** - Detailed feature descriptions:
  - Camera control and backends
  - Real-time pose estimation
  - Video recording
  - Configuration management
  - Processor system
  - User interface
  - Performance monitoring
  - Advanced features

### User Guide
- **[User Guide](user_guide.md)** - Complete usage walkthrough:
  - Getting started
  - Camera setup
  - DLCLive configuration
  - Recording videos
  - Configuration management
  - Common workflows
  - Tips and best practices
  - Troubleshooting

## Advanced Topics

### Processor System
- **[Processor Plugins](PLUGIN_SYSTEM.md)** - Custom pose processing
- **[Processor Auto-Recording](processor_auto_recording.md)** - Event-triggered recording
- Socket processor documentation

### Technical Details
- **[Timestamp Format](timestamp_format.md)** - Synchronization and timing
- **[ARAVIS_BACKEND_SUMMARY](../ARAVIS_BACKEND_SUMMARY.md)** - Implementation details

## By Use Case

### I want to...

#### Set up a camera
→ [Camera Support](camera_support.md) → Select backend → Follow setup guide

**By Platform**:
- **Windows**: [README](../README.md#windows-gentl-for-industrial-cameras) → GenTL setup
- **Linux**: [Aravis Backend](aravis_backend.md) → Installation for Ubuntu/Debian
- **macOS**: [Aravis Backend](aravis_backend.md) → Installation via Homebrew

**By Camera Type**:
- **Webcam**: [User Guide](user_guide.md#camera-setup) → OpenCV backend
- **Industrial Camera**: [Camera Support](camera_support.md) → GenTL/Aravis
- **Basler Camera**: [Camera Support](camera_support.md#basler-cameras) → pypylon setup
- **The Imaging Source**: [Aravis Backend](aravis_backend.md) or GenTL

#### Run pose estimation
→ [User Guide](user_guide.md#dlclive-configuration) → Load model → Start inference

#### Record high-speed video
→ [Features](features.md#video-recording) → Hardware encoding → GPU setup
→ [User Guide](user_guide.md#high-speed-recording-60-fps) → Optimization tips

#### Create custom processor
→ [Processor Plugins](PLUGIN_SYSTEM.md) → Plugin architecture → Examples

#### Trigger recording remotely
→ [Features](features.md#auto-recording-feature) → Auto-recording setup
→ Socket processor documentation

#### Optimize performance
→ [Features](features.md#performance-optimization) → Metrics → Adjustments
→ [User Guide](user_guide.md#tips-and-best-practices) → Best practices

## By Topic

### Camera Backends
| Backend | Documentation | Platform |
|---------|---------------|----------|
| OpenCV | [User Guide](user_guide.md#step-1-select-camera-backend) | All |
| GenTL | [Camera Support](camera_support.md) | Windows, Linux |
| Aravis | [Aravis Backend](aravis_backend.md) | Linux, macOS |
| Basler | [Camera Support](camera_support.md#basler-cameras) | All |

### Configuration
- **Basics**: [README](../README.md#configuration)
- **Management**: [User Guide](user_guide.md#working-with-configurations)
- **Templates**: [User Guide](user_guide.md#configuration-templates)
- **Details**: [Features](features.md#configuration-management)

### Recording
- **Quick Start**: [User Guide](user_guide.md#recording-videos)
- **Features**: [Features](features.md#video-recording)
- **Optimization**: [README](../README.md#performance-optimization)
- **Auto-Recording**: [Features](features.md#auto-recording-feature)

### DLCLive
- **Setup**: [User Guide](user_guide.md#dlclive-configuration)
- **Models**: [Features](features.md#model-support)
- **Performance**: [Features](features.md#performance-metrics)
- **Visualization**: [Features](features.md#pose-visualization)

## Troubleshooting

### Quick Fixes
1. **Camera not detected** → [User Guide](user_guide.md#troubleshooting-guide)
2. **Slow inference** → [Features](features.md#performance-optimization)
3. **Dropped frames** → [README](../README.md#troubleshooting)
4. **Recording issues** → [User Guide](user_guide.md#troubleshooting-guide)

### Detailed Troubleshooting
- [User Guide - Troubleshooting Section](user_guide.md#troubleshooting-guide)
- [README - Troubleshooting](../README.md#troubleshooting)
- [Aravis Backend - Troubleshooting](aravis_backend.md#troubleshooting)

## Development

### Architecture
- **Project Structure**: [README](../README.md#development)
- **Backend Development**: [Camera Support](camera_support.md#contributing-new-camera-types)
- **Processor Development**: [Processor Plugins](PLUGIN_SYSTEM.md)

### Implementation Details
- **Aravis Backend**: [ARAVIS_BACKEND_SUMMARY](../ARAVIS_BACKEND_SUMMARY.md)
- **Thread Safety**: [Features](features.md#thread-safety)
- **Resource Management**: [Features](features.md#resource-management)

## Reference

### Configuration Schema
```json
{
  "camera": {
    "name": "string",
    "index": "number",
    "fps": "number",
    "backend": "opencv|gentl|aravis|basler",
    "exposure": "number (μs, 0=auto)",
    "gain": "number (0.0=auto)",
    "crop_x0/y0/x1/y1": "number",
    "max_devices": "number",
    "properties": "object"
  },
  "dlc": {
    "model_path": "string",
    "model_type": "base|pytorch",
    "additional_options": "object"
  },
  "recording": {
    "enabled": "boolean",
    "directory": "string",
    "filename": "string",
    "container": "mp4|avi|mov",
    "codec": "h264_nvenc|libx264|hevc_nvenc",
    "crf": "number (0-51)"
  },
  "bbox": {
    "enabled": "boolean",
    "x0/y0/x1/y1": "number"
  }
}
```

### Performance Metrics
- **Camera FPS**: [Features](features.md#camera-metrics)
- **DLC Metrics**: [Features](features.md#dlc-metrics)
- **Recording Metrics**: [Features](features.md#recording-metrics)

### Keyboard Shortcuts
| Action | Shortcut |
|--------|----------|
| Load configuration | Ctrl+O |
| Save configuration | Ctrl+S |
| Save as | Ctrl+Shift+S |
| Quit | Ctrl+Q |

## External Resources

### DeepLabCut
- [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut)
- [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live)
- [DeepLabCut Documentation](http://deeplabcut.github.io/DeepLabCut/docs/intro.html)

### Camera Libraries
- [Aravis Project](https://github.com/AravisProject/aravis)
- [Harvesters (GenTL)](https://github.com/genicam/harvesters)
- [pypylon (Basler)](https://github.com/basler/pypylon)
- [OpenCV](https://opencv.org/)

### Video Encoding
- [FFmpeg](https://ffmpeg.org/)
- [NVENC (NVIDIA)](https://developer.nvidia.com/nvidia-video-codec-sdk)

## Getting Help

### Support Channels
1. Check relevant documentation (use this index!)
2. Search GitHub issues
3. Review example configurations
4. Contact maintainers

### Reporting Issues
When reporting bugs, include:
- GUI version
- Platform (OS, Python version)
- Camera backend and model
- Configuration file (if applicable)
- Error messages
- Steps to reproduce

## Contributing

Interested in contributing?
- See [README - Contributing](../README.md#contributing)
- Review [Development Section](../README.md#development)
- Check open GitHub issues
- Read coding guidelines

---

## Document Version History

- **v1.0** - Initial comprehensive documentation
  - Complete README overhaul
  - User guide creation
  - Features documentation
  - Camera backend guides
  - Aravis backend implementation

## Quick Navigation

**Popular Pages**:
- [User Guide](user_guide.md) - Most comprehensive walkthrough
- [Features](features.md) - All capabilities detailed
- [Aravis Setup](aravis_backend.md) - Linux industrial cameras
- [Camera Support](camera_support.md) - All camera backends

**By Experience Level**:
- **Beginner**: [README](../README.md) → [User Guide](user_guide.md)
- **Intermediate**: [Features](features.md) → [Camera Support](camera_support.md)
- **Advanced**: [Processor Plugins](PLUGIN_SYSTEM.md) → Implementation details

---

*Last updated: 2025-10-24*
