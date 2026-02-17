# Basler backend

The Basler backend provides support for Basler cameras using the official **pylon SDK** through the **pypylon** Python bindings.

Download the official pylon SDK from Basler and install the `pypylon` Python package to use this backend.

```{note}
This backend requires the optional `pypylon` dependency. If `pypylon` is not installed, the backend will be unavailable.
```

---

## Features

- Native Basler camera support via **pypylon** (Pylon SDK bindings).
- Best-effort device discovery without opening cameras (enumerates `DeviceInfo` entries).
- Stable camera identity via **serial number** (`device_id`) with automatic index rebinding.
- Configurable exposure, gain, frame rate, and resolution.
- Frames are converted to **BGR (8-bit)** for consistency with other GUI backends.

---

## Installation

```{important}
For up to date installation instructions, please refer to the
[official pypylon documentation](https://github.com/basler/pypylon?tab=readme-ov-file#Installation)
and the [Basler pylon software installation guide](https://docs.baslerweb.com/camera-installation).
```

### 1) Install Basler pylon SDK

Basler recommends installing **pylon** first (strongly recommended even if you install `pypylon` via pip).

#### Linux

Basler provides pylon for Linux as **Debian packages** and **.tar.gz archives** (x86_64 and ARM variants).

- Download the matching pylon installer from Basler and follow the included `INSTALL` instructions.

#### Windows

- Install the Basler pylon Camera Software Suite (includes drivers and tooling).

#### macOS

- Install the Basler pylon package for macOS (Intel/ARM supported, depending on Basler release). Basler lists macOS as a supported system for pypylon usage.

### 2) Install `pypylon`

Install into the same Python environment as your GUI:

```bash
pip install pypylon
```

`pypylon` is the official Python wrapper for the Basler pylon Camera Software Suite

---

## Basic configuration

Select the Basler backend in the GUI or via configuration:

```json
{
  "camera": {
    "backend": "basler",
    "index": 0,
    "fps": 30.0,
    "exposure": 8000,
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
    "backend": "basler",
    "index": 0
  }
}
```

### By serial number (recommended for stability)

The backend supports a stable identity field `device_id` (serial number). When present, it is preferred over `index` and is also persisted automatically after a successful open.

```json
{
  "camera": {
    "backend": "basler",
    "index": 0,
    "properties": {
      "basler": {
        "device_id": "40312345"
      }
    }
  }
}
```

How selection works:

1. If `properties.basler.device_id` is set, the backend selects the device with a matching serial number.
2. Otherwise, the backend uses `index`.

---

## Full properties and configuration

Basler-specific options live under the `properties.basler` namespace.

### Related camera settings (shared across backends)

These values are configurable in the GUI and are shared for all backends.

- `width` (int): requested image width. Use `0` for “Auto / keep device default”.
- `height` (int): requested image height. Use `0` for “Auto / keep device default”.
- `fps` (float): requested acquisition frame rate. Use `0.0` to not set.
- `exposure` (float): exposure time in microseconds. Use `<= 0` to not set.
- `gain` (float): camera gain value. Use `<= 0` to not set.

### Basler namespace options

These settings live under the `properties.basler` entry.

- `device_id` (string): preferred stable identity (camera serial number).
- `fast_start` (bool, default: false): probe-mode hint.
  - When `true`, the backend will open the camera but **will not start grabbing** and will **not create the pixel format converter**.
  - This is intended for fast capability probing; it is **not suitable for normal capture**.
- `resolution` ([w, h]): optional override resolution pair.
  - Used only if `width`/`height` are not set.

### Auto-populated Basler metadata

After a successful open, the backend may populate the following read-only convenience fields in `properties.basler`:

- `device_name`: a friendly device name (if provided by the SDK).
- (Optionally) `device_path`: a full identifier string (depending on SDK/device).

These fields are managed automatically and are not required to configure the backend.

---

## Exposure and gain

Behavior:

- If `exposure > 0`, the backend attempts to disable `ExposureAuto` (if present) and sets `ExposureTime` in microseconds.
- If `gain > 0`, the backend attempts to disable `GainAuto` (if present) and sets `Gain`.
- If `exposure <= 0` or `gain <= 0`, the backend leaves the camera’s auto/manual mode unchanged.

Example:

```json
{
  "camera": {
    "backend": "basler",
    "exposure": 10000,
    "gain": 6.0
  }
}
```

---

## Frame rate (FPS)

- FPS is only applied when `fps > 0`.
- The backend attempts to enable `AcquisitionFrameRateEnable` when available, then sets `AcquisitionFrameRate`.
- The backend reads back the **actual FPS** (if available) and exposes it via telemetry.

---

## Resolution handling

Resolution is only changed when explicitly requested.

Priority order for requesting a resolution:

1. `width` + `height` (GUI fields)
2. `properties.basler.resolution` (namespaced override)

If no resolution is provided (or if width/height are `0`), the backend preserves the camera’s default configuration.

Increment and range constraints:

- Many Basler cameras restrict `Width`/`Height` to specific increments.
- The backend snaps requested values down to the nearest valid increment (best-effort) and clamps to min/max.
- A warning is logged if the requested and applied resolutions differ.

---

## Pixel format and color conversion

To provide a consistent frame format across backends, the Basler backend converts frames to:

- **BGR8 packed** (8-bit BGR)

Internally, it uses a pypylon `ImageFormatConverter` configured for `PixelType_BGR8packed`.

---

## Device discovery

The backend can enumerate devices without opening them and returns (best-effort):

- `index`: current device list index
- `label`: human-readable label (vendor/model + serial if available)
- `device_id`: serial number (stable identity)
- `path`: full name string (if available)

Note that availability and richness of fields depend on camera transport and SDK support.

---

## Troubleshooting

### Backend not available / import error

- Ensure `pypylon` is installed:
  ```bash
  python -c "import pypylon"
  ```
- Install it if missing:
  ```bash
  pip install pypylon
  ```


### No cameras detected

- Verify the Basler pylon runtime is installed and your camera is visible in Basler tooling.
- On Linux, ensure you installed pylon using the Basler-provided packages/archives and followed the included `INSTALL` guide.

### Resolution mismatch warnings

If you request a resolution that violates camera constraints (min/max or increment), the backend will snap/clamp to valid values and log a warning.

---

## Example configuration

```json
{
  "camera": {
    "backend": "basler",
    "index": 0,
    "fps": 60.0,
    "exposure": 8000,
    "gain": 10.0,
    "width": 1920,
    "height": 1080,
    "properties": {
      "basler": {
        "device_id": "40312345"
      }
    }
  }
}
```

---

## Resources

- Basler pypylon (PyPI): [Open on PyPI](https://pypi.org/project/pypylon/)
- Basler pylon Software Installation (Linux): [See Basler documentation](https://docs.baslerweb.com/camera-installation)
