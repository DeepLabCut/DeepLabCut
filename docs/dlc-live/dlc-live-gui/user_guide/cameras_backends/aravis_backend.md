(file:dlclivegui-camera-aravis-backend)=
# Aravis backend

The Aravis backend provides support for GenICam-compatible cameras using the
[Aravis](https://github.com/AravisProject/aravis) library.

```{important}
Support for Aravis in the GUI is currently experimental.
Please report issues on GitHub to help improve this backend.
```

---

## Features

- Support for GenICam / GigE Vision cameras via Aravis 0.8
- Automatic device discovery without opening cameras
- Configurable exposure, gain, frame rate, and resolution
- Support for common mono and color pixel formats
- Efficient streaming with configurable buffer count

## Installation

### Linux (Ubuntu / Debian)

```bash
sudo apt-get install gir1.2-aravis-0.8 python3-gi
```

### Linux (Fedora)

```bash
sudo dnf install aravis python3-gobject
```

### Windows

Aravis support on Windows requires building from source or using WSL.
For native Windows usage, consider the GenTL backend instead.

### macOS

```bash
brew install aravis
pip install pygobject
```

```{note}
On macOS, installing `pygobject` may require additional system
dependencies such as `gobject-introspection` and `cairo`.
```

---

## Basic configuration

Select the Aravis backend either in the GUI or via configuration:

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

---

## Camera selection

### By index (default)

```json
{
  "camera": {
    "backend": "aravis",
    "index": 0
  }
}
```

### By device ID (recommended for stability)

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

```{note}
The backend may automatically populate additional read-only identity fields
(vendor, model, serial, etc.) after a successful open. These are primarily
used internally and by the GUI.
```

---

## Full properties and configuration

Aravis-specific options live under the `properties.aravis` namespace in
the settings used by the GUI and configuration files.

### Related camera settings

```{tip}
These values are accessible directly in the GUI and are shared for all backends.
```

| Property | Type | Description |
|--------|------|-------------|
| `width` | int | Requested image width (optional) |
| `height` | int | Requested image height (optional) |
| `fps` | float | Target acquisition frame rate |
| `exposure` | float | Exposure time in microseconds |
| `gain` | float | Camera gain value |

### Common Aravis properties

```{note}
These properties are specific to the Aravis backend and must be set manually in the configuration file.
```

| Property | Type | Default | Description |
|--------|------|---------|-------------|
| `device_id` | string | — | Explicit Aravis device ID (overrides index) |
| `pixel_format` | string | `Mono8` | Requested pixel format |
| `timeout` | int | `2000000` | Frame timeout in microseconds |
| `n_buffers` | int | `10` | Number of streaming buffers |

### Pixel format

Supported values:

- `Mono8`
- `Mono12`
- `Mono16`
- `RGB8`
- `BGR8`

Internally, all frames are converted to **BGR (8-bit)** for consistency.

**Mono12 / Mono16 scaling behavior**:
- 12-bit and 16-bit images are dynamically scaled **per frame** to 8-bit
- Scaling is based on the maximum pixel value present in each frame
- This improves visibility but may cause frame-to-frame brightness variation

### Exposure and gain

- **Exposure** is specified in microseconds
- **Gain** is a unitless camera-specific value

```json
{
  "camera": {
    "exposure": 8000,
    "gain": 10.0
  }
}
```

Behavior:

- Exposure or gain values `<= 0` leave the camera in auto mode
- Positive values disable auto-exposure / auto-gain automatically
- Actual values are read back and may differ slightly due to camera constraints


### Frame rate (FPS)

```json
{
  "camera": {
    "fps": 60.0
  }
}
```

- FPS is only applied when a positive value is provided
- The backend attempts to set `AcquisitionFrameRate`
- The **actual FPS** reported by the camera is stored and may differ slightly
- Mismatches are logged but do not fail camera startup


### Resolution handling

Resolution is **only changed when explicitly requested**.
If no resolution is specified, the camera's default configuration is preserved.

Supported ways to request resolution:

```json
{
  "camera": {
    "width": 1920,
    "height": 1080
  }
}
```

Notes:
- The camera may clamp or adjust the requested resolution
- The backend records and exposes the **actual resolution** after opening
- A warning is logged if the requested and actual resolutions differ

### Auto-populated Aravis metadata

```{caution}
These fields may appear in saved configurations but are managed
automatically by the backend and GUI.
It is not recommended to set these manually.
```

- `device_physical_id`
- `device_vendor`
- `device_model`
- `device_serial_nbr`
- `device_protocol`
- `device_address`
- `device_name`
- `device_path`

## Streaming and performance tuning

### Buffer Count

Increase buffers for high-throughput or high-latency systems:

```json
{
  "camera": {
    "properties": {
      "aravis": {
        "n_buffers": 20
      }
    }
  }
}
```

### Timeout

Adjust frame timeout for slower cameras or congested networks:

```json
{
  "camera": {
    "properties": {
      "aravis": {
        "timeout": 5000000
      }
    }
  }
}
```

(5 seconds = 5,000,000 microseconds)

---

## Troubleshooting

### No cameras detected

1. Verify Aravis installation:
   ```bash
   arv-tool-0.8 -l
   ```
2. Check power, cabling, and network configuration
3. Ensure sufficient permissions for USB or network devices

### Timeout errors

- Increase the `timeout` value
- Increase `n_buffers`
- Check GigE bandwidth and packet size configuration

### Pixel format errors

- Inspect supported formats:
  ```bash
  arv-tool-0.8 -n <camera-name> features
  ```
- Try a simpler format such as `Mono8`

---

## Comparison with GenTL backend

| Feature | Aravis | GenTL |
|-------|--------|-------|
| Best Platform | Linux | Windows |
| Camera Support | GenICam / GigE | Vendor GenTL |
| Installation | System packages | Vendor CTI files |
| Auto-detection | Yes | Yes |
| Performance | Excellent | Excellent |

---

## Example configuration

```json
{
  "camera": {
    "backend": "aravis",
    "index": 0,
    "fps": 60.0,
    "exposure": 8000,
    "gain": 10.0,
    "properties": {
      "aravis": {
        "pixel_format": "Mono8",
        "n_buffers": 15,
        "timeout": 3000000
      }
    }
  }
}
```

---

## Resources

- [Aravis Project](https://github.com/AravisProject/aravis)
- [GenICam Standard](https://www.emva.org/standards-technology/genicam/)
- [Python GObject Documentation](https://pygobject.readthedocs.io/)
