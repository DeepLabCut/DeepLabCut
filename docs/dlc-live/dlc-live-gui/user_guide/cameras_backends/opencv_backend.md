# OpenCV Backend

The OpenCV backend provides camera support via `cv2.VideoCapture`.

```{important}
The OpenCV backend is intended for UVC/webcams and other devices supported by your system’s native multimedia stack.
Exposure and gain control are not standardized across OpenCV backends and are treated as unsupported in this GUI.
Other settings may not always behave as expected due to driver and backend limitations.
```

---

## Features

- Cross-platform capture using OpenCV (`cv2.VideoCapture`).
- Platform-optimized default backend selection:
  - Windows: prefer DirectShow (`CAP_DSHOW`), fall back to MSMF/ANY.
  - macOS: prefer AVFoundation (`CAP_AVFOUNDATION`), fall back to ANY.
  - Linux: prefer V4L2 (`CAP_V4L2`), fall back to ANY.
- Device discovery with stable identity (when enumeration is available):
  - Uses an external enumerator (`opencv2-enumerate-cameras`) via helper utilities.
  - Persists `device_id` / VID / PID / name into configuration when available.
- Best-effort format negotiation:
  - Resolution and FPS are applied with tolerance-based verification.
  - Mismatch handling is configurable (warn/strict/accept).
- Optional MJPG (Windows) and explicit FOURCC requests.

---

## Installation

### Python dependency

OpenCV must be available in the same Python environment as the GUI, but is installed
as part of the core DeepLabCut-live-GUI package, so no additional installation is needed for OpenCV support.

`cv2-enumerate-cameras` is also installed by default to provide camera enumeration support for this backend and make device selection more robust.

---

## Basic configuration

Select the OpenCV backend in the GUI or via configuration:

```json
{
  "camera": {
    "backend": "opencv",
    "index": 0,
    "fps": 30.0,
    "width": 1280,
    "height": 720
  }
}
```

Notes:

- If `width`/`height` are omitted or set to `0`, the backend keeps the camera’s default mode.
- OpenCV may ignore FPS and resolution requests depending on driver/backend.

---

## Camera selection configuration

### By index (default)

```json
{
  "camera": {
    "backend": "opencv",
    "index": 0
  }
}
```

### By stable identity (recommended when available)

When discovery is available, the backend can persist and later rebind a stable identity under `properties.opencv`:

- `device_id`: stable ID from camera enumeration
- `device_vid` / `device_pid`: USB VID/PID when known
- `device_name`: camera name

Example:

```json
{
  "camera": {
    "backend": "opencv",
    "index": 0,
    "properties": {
      "opencv": {
        "device_id": "usb:046d:0825:...",
        "device_name": "Logitech C270"
      }
    }
  }
}
```

Selection priority in `open()`:

1. `properties.opencv.device_id` (stable ID)
2. `properties.opencv.device_name` (substring match)
3. `properties.opencv.device_vid` + `device_pid`
4. `index` fallback

---

## Full properties and configuration

OpenCV-specific options live under `properties.opencv`.

### Related camera settings (shared across backends)

- `width` (int): requested image width; `0` means keep device default
- `height` (int): requested image height; `0` means keep device default
- `fps` (float): requested FPS; `0.0` means do not set

```{note}
`exposure` and `gain` fields may be present but are currently treated as unsupported in this backend due to lack of standardization across OpenCV drivers.
```

### OpenCV namespace options (`properties.opencv`)

Device selection:

- `device_id` (string): stable identity from enumeration
- `device_name` (string): substring to match a device name
- `device_vid` (int): USB vendor ID
- `device_pid` (int): USB product ID

Backend/open behavior:

- `api` (string | null): preferred OpenCV API backend override.
  - Common values: `DSHOW`, `MSMF`, `V4L2`, `AVFOUNDATION`, `ANY`.
- `fast_start` (bool, default: false):
  - Skips heavy negotiation; applies resolution in best-effort mode.
  - Useful for faster startup when probing devices, no effect on opening a known device.

Format negotiation policy:

- `resolution_policy` (string, default: `warn`): how to handle mismatch between requested and applied resolution.
  - `warn`: log a warning
  - `strict`: raise an error
  - `accept`: accept mismatch
- `persist_last_applied_resolution` (bool, default: false):
  - If enabled, stores `last_applied_resolution` in `properties.opencv` after successful negotiation.
- `enforce_aspect` (string, default: `strict`): aspect ratio policy for verification.
  - `strict`, `prefer`, `ignore`
- `aspect_tol` (float, default: 0.01): aspect tolerance (fraction)
- `area_tol` (float, default: 0.05): area tolerance (fraction)

Codec policy:

- `prefer_mjpg` (bool, default: false): attempt to enable MJPG on Windows.
- `fourcc` (string | null): explicit FOURCC request, overrides `prefer_mjpg`.
  - Examples: `MJPG`, `YUY2`, `NV12`, `H264`, `XRGB`, `BGR3`

---

## Resolution and FPS behavior

### Resolution

- If `width` and `height` are both `> 0`:
  - In normal mode, the backend uses a verified negotiation path and records the actual applied resolution.
  - In `fast_start` mode, it applies width/height via `CAP_PROP_FRAME_WIDTH/HEIGHT` best-effort.
- If `width` or `height` is `0`:
  - The backend does not attempt to set resolution.
  - It reads the current device defaults.

### FPS

- If `fps > 0`, the backend attempts to set `CAP_PROP_FPS` best-effort.
- Many drivers return `0.0` for FPS even when streaming successfully; this is normal for some OpenCV backends.

---

## Device discovery and rebind

### Discovery

If camera enumeration is available, `discover_devices()` returns a list of `DetectedCamera` with:

- `index`
- `label`
- `device_id` (stable ID)
- `vid`, `pid`
- `path` (if known)
- `backend_hint`

If enumeration is not available, `discover_devices()` returns `None` so the factory can fall back to probing.

### Rebinding

If `properties.opencv.device_id` (or VID/PID/name) exists, `rebind_settings()` attempts to map the saved identity to the current index and refresh stored fields.

---

## Troubleshooting

### Camera opens but frames are `None`

- This indicates transient grab/retrieve failures. Common causes:
  - Another application is using the camera.
  - The selected backend (DSHOW/MSMF/V4L2/AVFOUNDATION) is unstable for this device.

Try:

- Set an explicit backend API:
  ```json
  {
    "camera": {
      "backend": "opencv",
      "properties": {
        "opencv": { "api": "DSHOW" }
      }
    }
  }
  ```


### Slow open on Windows (MSMF)

If MSMF is selected and opening is slow, consider setting:

- `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`

This must be set **before importing `cv2`**.

### Resolution mismatch

If you request a resolution that the driver cannot apply, you may see warnings.

- Switch `resolution_policy` to `strict` to fail fast.
- Or switch it to `accept` to silence warnings.

### MJPG / codec issues

On Windows, MJPG can reduce USB bandwidth and improve FPS for some webcams.

- Enable MJPG attempt:
  ```json
  {
    "camera": {
      "backend": "opencv",
      "properties": {
        "opencv": { "prefer_mjpg": true }
      }
    }
  }
  ```

- Or force a specific FOURCC:
  ```json
  {
    "camera": {
      "backend": "opencv",
      "properties": {
        "opencv": { "fourcc": "MJPG" }
      }
    }
  }
  ```

---

## Example configuration

```json
{
  "camera": {
    "backend": "opencv",
    "index": 0,
    "fps": 60.0,
    "width": 1280,
    "height": 720,
    "properties": {
      "opencv": {
        "api": "DSHOW",
        "resolution_policy": "warn",
        "enforce_aspect": "strict",
        "aspect_tol": 0.01,
        "area_tol": 0.05,
        "prefer_mjpg": true,
        "timeout": 2.0
      }
    }
  }
}
```

---

## Notes and limitations

- Exposure and gain controls are marked unsupported in this backend (they are highly backend- and camera-specific in OpenCV).
- Some OpenCV backends do not report FPS reliably.
- For stable multi-camera setups, prefer saving and using `properties.opencv.device_id` (stable identity) when enumeration is available.
