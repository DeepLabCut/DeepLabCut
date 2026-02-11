(file:dlclivegui-camera-aravis-backend)=
# Aravis Backend

The Aravis backend provides support for GenICam-compatible cameras using the [Aravis](https://github.com/AravisProject/aravis) library.

```{important}
Support for Aravis in our GUI is currently experimental.
Please report any issues or feedback to help us improve this backend.
```

## Features

- Support for GenICam/GigE Vision cameras
- Automatic device detection with `get_device_count()`
- Configurable exposure time and gain
- Support for various pixel formats (Mono8, Mono12, Mono16, RGB8, BGR8)
- Efficient streaming with configurable buffer count
- Timeout handling for robust operation

## Installation

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install gir1.2-aravis-0.8 python3-gi
```

### Linux (Fedora)
```bash
sudo dnf install aravis python3-gobject
```

### Windows
Aravis support on Windows requires building from source or using WSL. For native Windows support, consider using the GenTL backend instead.

### macOS
```bash
brew install aravis
pip install pygobject
```

## Configuration

### Basic Configuration

Select "aravis" as the backend in the GUI or in your configuration file:

```json
{
  "camera": {
    "backend": "aravis",
    "index": 0,
    "fps": 30.0,
    "exposure": 10000,
    "gain": 5.0
  }
}
```

### Advanced Properties

You can configure additional Aravis-specific properties via the `properties` dictionary:

```json
{
  "camera": {
    "backend": "aravis",
    "index": 0,
    "fps": 30.0,
    "exposure": 10000,
    "gain": 5.0,
    "properties": {
      "aravis": {
        "camera_id": "MyCamera-12345",
        "pixel_format": "Mono8",
        "timeout": 2000000,
        "n_buffers": 10
      }
    }
  }
}
```

#### Available Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `device_id` | string | None | Specific camera ID to open (overrides index) |
| `pixel_format` | string | "Mono8" | Pixel format: Mono8, Mono12, Mono16, RGB8, BGR8 |
| `timeout` | int | 2000000 | Frame timeout in microseconds (2 seconds) |
| `n_buffers` | int | 10 | Number of buffers in the acquisition stream |

### Exposure and Gain

The Aravis backend supports exposure time (in microseconds) and gain control:

- **Exposure**: Set via the GUI exposure field or `settings.exposure` (0 = auto, >0 = manual in μs)
- **Gain**: Set via the GUI gain field or `settings.gain` (0.0 = auto, >0.0 = manual value)

When exposure or gain are set to non-zero values, the backend automatically disables auto-exposure and auto-gain.

## Camera Selection

### By Index

The default method is to select cameras by index (0, 1, 2, etc.):

```json
{
  "camera": {
    "backend": "aravis",
    "index": 0
  }
}
```

### By Camera ID

You can also select a specific camera by its ID:

```json
{
  "camera": {
    "backend": "aravis",
    "properties": {
      "aravis": {
        "device_id": "TheImagingSource-12345678"
      }
    }
  }
}
```

## Supported Pixel Formats

The backend automatically converts different pixel formats to BGR format for consistency:

- **Mono8**: 8-bit grayscale → BGR
- **Mono12**: 12-bit grayscale → scaled to 8-bit → BGR
- **Mono16**: 16-bit grayscale → scaled to 8-bit → BGR
- **RGB8**: 8-bit RGB → BGR (color conversion)
- **BGR8**: 8-bit BGR (no conversion needed)

## Performance Tuning

### Buffer Count
Increase `n_buffers` for high-speed cameras or systems with variable latency:

```json
{
  "properties": {
    "n_buffers": 20
  }
}
```

### Timeout
Adjust timeout for slower cameras or network cameras:

```json
{
  "properties": {
    "timeout": 5000000
  }
}
```
(5 seconds = 5,000,000 microseconds)

## Troubleshooting

### No cameras detected

1. Verify Aravis installation: `arv-tool-0.8 -l`
2. Check camera is powered and connected
3. Ensure proper network configuration for GigE cameras
4. Check user permissions for USB cameras

### Timeout errors

- Increase the `timeout` property
- Check network bandwidth for GigE cameras
- Verify camera is properly configured and streaming

### Pixel format errors

- Check camera's supported pixel formats: `arv-tool-0.8 -n <camera-name> features`
- Try alternative formats: Mono8, RGB8, etc.

## Comparison with GenTL Backend

| Feature | Aravis | GenTL |
|---------|--------|-------|
| Platform | Linux (best), macOS (experimental) | Windows (best), Linux |
| Camera Support | GenICam/GigE | GenTL producers |
| Installation | System packages | Vendor CTI files |
| Performance | Excellent | Excellent |
| Auto-detection | Yes | Yes |

## Example: The Imaging Source Camera

```json
{
  "camera": {
    "backend": "aravis",
    "index": 0,
    "fps": 60.0,
    "exposure": 8000,
    "gain": 10.0,
    "properties": {
      "pixel_format": "Mono8",
      "n_buffers": 15,
      "timeout": 3000000
    }
  }
}
```

## Resources

- [Aravis Project](https://github.com/AravisProject/aravis)
- [GenICam Standard](https://www.emva.org/standards-technology/genicam/)
- [Python GObject Documentation](https://pygobject.readthedocs.io/)
