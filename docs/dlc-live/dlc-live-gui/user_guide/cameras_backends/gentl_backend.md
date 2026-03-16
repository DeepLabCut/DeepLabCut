# GenTL backend

The GenTL backend provides support for **GenICam / GenTL** compatible cameras using the **Harvesters** Python library (a GenTL consumer).

```{note}
This backend requires the optional `harvesters` dependency and at least one vendor **GenTL Producer** (`.cti`) installed on the system.
```

```{important}
Support for GenTL in the GUI is currently experimental.
Please report issues on GitHub to help improve this backend.
```

---

## Features & design

- Image acquisition via **Harvesters** (GenTL consumer for GenICam-compliant devices).
- Loads **multiple GenTL Producers** (`.cti` files) to support mixed transports/vendors (USB3 Vision, GigE Vision, frame grabbers—depending on producer).
- **CTI persistence + diagnostics**:
  - `properties.gentl.cti_files`: all resolved CTI candidates (after resolution)
  - `properties.gentl.cti_files_loaded`: CTIs successfully loaded into Harvesters
  - `properties.gentl.cti_files_failed`: list of `{cti, error}` entries for producers that failed to load
  - `properties.gentl.cti_file`: convenience “first CTI” (for backward compatibility / display)
- **Stable identity** via `properties.gentl.device_id`:
  - `device_id = "serial:<SERIAL>"` when a serial number is available (**preferred**)
  - `device_id = "fp:<fingerprint...>"` as a best-effort fallback when serials are missing/ambiguous
- **Automated rebinding** (`rebind_settings`) maps stored `device_id` → the correct current index.
- Best-effort configuration through the device GenApi node map:
  - exposure, gain, frame rate, resolution, pixel format
- Pixel-format normalization to **BGR (8-bit)** for consistency:
  - Mono formats → BGR
  - `RGB8` → BGR
  - Non-8-bit frames → scaled down to 8-bit (per-frame scaling)

---

## Installation

### 1) Install Harvesters

Install Harvesters into the same Python environment as your GUI:

```bash
pip install harvesters
```

### 2) Install a GenTL Producer file (`.cti`)

A GenTL Producer is a vendor-provided library that exposes cameras to GenTL consumer applications.
It is typically distributed as part of the camera vendor SDK or framegrabber SDK.

```{note}
GenTL Producers are identified by files ending in `.cti`.
```

### 3) Make producers discoverable (environment variables)

Most GenTL consumers locate producers via the standard environment variables:

- `GENICAM_GENTL64_PATH` (64-bit producers)
- `GENICAM_GENTL32_PATH` (32-bit producers)

If you have multiple producers installed, separate entries with:

- `;` on Windows
- `:` on Linux/macOS (UNIX-like)

```{tip}
Many vendor installers set `GENICAM_GENTL64_PATH` automatically. If your camera is not discovered, explicitly set the variable (or provide `cti_file` / `cti_files` in configuration as described below).
```

---

## Basic configuration

Select the GenTL backend in the GUI or via configuration:

```json
{
  "camera": {
    "backend": "gentl",
    "index": 0,
    "fps": 30.0,
    "exposure": 8000,
    "gain": 5.0,
    "properties": {
      "gentl": {
        "pixel_format": "Mono8",
        "timeout": 2.0
      }
    }
  }
}
```

---

## CTI / producer configuration

### Default behavior: discover and load all producers

By default, the backend will **discover** and **try to load all available** GenTL Producers (`.cti`) it can find.

- If a producer fails to load, the backend continues and attempts to load the others.
- Startup fails only when:
  - **no CTI files can be found**, or
  - **no CTI files can be loaded**, or
  - **no devices are detected** after loading producers.

### CTI resolution precedence (advanced)

CTI locations are resolved in this order:

1. **Namespace explicit CTIs** (`properties.gentl`):
   - `properties.gentl.cti_files`
   - `properties.gentl.cti_file`

   Behavior depends on the persisted source marker `properties.gentl.cti_files_source`:

   - If `cti_files_source == "user"` (or missing/unknown):
     - Treated as a **user override**
     - **strict**: missing paths cause `open()` to raise

   - If `cti_files_source == "auto"`:
     - Treated as an **auto-discovered cache**
     - If cached paths are stale/missing, `open()` will **fall back to discovery** automatically

2. **Discovery** (auto):
   - environment: `GENICAM_GENTL64_PATH` / `GENICAM_GENTL32_PATH`
   - optional: `properties.gentl.cti_search_paths` (glob patterns)
   - optional: `properties.gentl.cti_dirs` (extra directories; non-recursive)
   - plus built-in Windows fallback patterns for some common installations

```{note}
You typically do **not** need to set `cti_files_source` yourself.
It is persisted by the backend so it can distinguish between a user-pinned CTI and an auto-discovered cache.
```

### Pin a known-good CTI (strict)

Use this when a specific vendor producer is known to work reliably (or when other installed CTIs are incompatible).

```json
{
  "camera": {
    "backend": "gentl",
    "properties": {
      "gentl": {
        "cti_file": "C:/Path/To/Your/Producer.cti"
      }
    }
  }
}
```

### Provide an explicit list of CTIs (strict)

```json
{
  "camera": {
    "backend": "gentl",
    "properties": {
      "gentl": {
        "cti_files": [
          "C:/Path/To/ProducerA.cti",
          "C:/Path/To/ProducerB.cti"
        ]
      }
    }
  }
}
```

### Provide CTI search patterns (auto)

```json
{
  "camera": {
    "backend": "gentl",
    "properties": {
      "gentl": {
        "cti_search_paths": [
          "C:/Program Files/YourVendor/**/bin/*.cti",
          "/opt/yourvendor/lib/gentlproducer/*.cti"
        ]
      }
    }
  }
}
```

### Provide extra CTI directories (auto)

```json
{
  "camera": {
    "backend": "gentl",
    "properties": {
      "gentl": {
        "cti_dirs": [
          "C:/Program Files/YourVendor/bin",
          "/opt/yourvendor/lib/gentlproducer"
        ]
      }
    }
  }
}
```

### CTI diagnostics persisted by `open()`

After `open()` (success or failure), the backend writes:

- `properties.gentl.cti_files`: all resolved candidates
- `properties.gentl.cti_files_loaded`: successfully loaded into Harvesters
- `properties.gentl.cti_files_failed`: `{cti, error}` for failures
- `properties.gentl.cti_file`: first loaded CTI (or first candidate)

These fields are intended for UI troubleshooting and do not normally need manual edits.

---

## Camera selection and stable identity

### By index (default)

```json
{
  "camera": {
    "backend": "gentl",
    "index": 0
  }
}
```

### By stable identity (recommended)

Prefer `properties.gentl.device_id`, which is persisted automatically after a successful `open()`.

- `serial:<SERIAL>` when a serial number is available
- `fp:<...>` fingerprint when serial numbers are missing

```json
{
  "camera": {
    "backend": "gentl",
    "properties": {
      "gentl": {
        "device_id": "serial:40312345"
      }
    }
  }
}
```

### Selection order in `open()`

The backend selects a device in this order:

1. Exact match of `device_id` against computed IDs for discovered devices
2. If `device_id` starts with `serial:`, match by exact serial number, then (if needed) substring
3. Legacy serial keys (`serial_number` / `serial`) if present (exact then substring)
4. Fallback to `index`

If a serial substring matches **multiple** cameras, an “ambiguous” error is raised.

```{tip}
The backend updates `settings.index` to the selected device’s current index to improve UI stability.
```

---

### Automated rebind (index changes, reconnects)

When the UI restarts (or devices re-enumerate), the backend can **rebind settings**:

- If `properties.gentl.device_id` exists, `rebind_settings()` tries to map it to the current device list.
- It prefers the persisted CTIs when available:
  - if `cti_files_source == "auto"` and cached CTIs are stale, it falls back to discovery automatically.
  - if CTIs were user-pinned and no longer exist, it does **not** attempt to override them silently.

Matching strategy:

1. Exact match on computed `device_id`
2. Fallback: treat stored value as a serial-like substring and match the first serial containing it

---

## Camera settings

These settings are shared across backends and configurable in the GUI:

- `width` (int): requested image width; **applied only if both width and height > 0**
- `height` (int): requested image height; **applied only if both width and height > 0**
- `fps` (float): requested frame rate; if unset/0, the backend does not set FPS
- `exposure` (float): exposure time; `<= 0` means do not set
- `gain` (float): gain value; `<= 0` means do not set

---

## Full properties and advanced configuration

GenTL backend options live under `properties.gentl` of the camera settings object.

### GenTL namespace options (`properties.gentl`)

Core / CTI resolution:

- `cti_file` (string): full path to a GenTL Producer `.cti` file
- `cti_files` (list[string]): explicit list of producer `.cti` files to load
- `cti_search_paths` (list[string] or string): glob patterns used to locate `.cti` files
- `cti_dirs` (list[string] or string): extra directories to scan for `*.cti` (non-recursive)
- `cti_files_source` (string): `"auto"` (cache) or `"user"` (strict override)
  - typically written by the backend; rarely set manually

Selection / identity:

- `device_id` (string): stable identifier (`serial:...` or `fp:...`)
- `serial_number` / `serial` (string): legacy selection helpers (still honored)

Acquisition:

- `pixel_format` (string, default: `Mono8`): requested `PixelFormat` symbolic
- `timeout` (float, default: `2.0`): acquisition timeout in seconds (`fetch(timeout=...)`)

Transforms:

- `rotate` (int, default: `0`): rotate output by 0/90/180/270
- `crop` ([top, bottom, left, right]): crop rectangle
  - if bottom/right are `<= 0`, they default to full frame extent

Probe / telemetry:

- `fast_start` (bool, default: false): probe-mode hint; when true, `open()` does not start acquisition
- `cti_files_loaded` (list[string]): populated automatically after open
- `cti_files_failed` (list[object]): populated automatically after open; each entry has `cti` and `error`

---

### Pixel format

- The backend attempts to set `node_map.PixelFormat` to the configured `pixel_format`
  if it appears in `PixelFormat.symbolics`.
- If the requested format is not available, it logs a warning and continues.

Frames are normalized to **BGR (8-bit)**:

- Mono images become BGR via grayscale-to-BGR conversion
- `RGB8` is converted to BGR
- Higher bit-depth images are scaled to 8-bit based on the frame’s max value (per frame)

---

### Exposure and gain

Best-effort behavior (depends on producer + camera GenApi implementation):

- If exposure is set (> 0):
  - attempts to disable `ExposureAuto` by setting it to `Off`
  - tries `ExposureTime` then `Exposure`
- If gain is set (> 0):
  - attempts to disable `GainAuto` by setting it to `Off`
  - tries `Gain`

If nodes are missing or read-only, the backend logs a warning and continues.

---

### Frame rate (FPS)

If `fps` is set to a non-zero value:

- attempts to enable frame rate control via:
  - `AcquisitionFrameRateEnable` or `AcquisitionFrameRateControlEnable`
- tries to set one of these nodes (first that works):
  - `AcquisitionFrameRate`
  - `ResultingFrameRate`
  - `AcquisitionFrameRateAbs`

The backend also tries to read back `ResultingFrameRate` for GUI telemetry (`actual_fps`).

---

### Resolution handling

Resolution is applied **only when explicitly requested** (either `width+height`, or legacy `properties.resolution`).

- Attempts to set `node_map.Width` and `node_map.Height`
- Clamps to node min/max and snaps down to the nearest valid increment (`inc`) when available
- Logs a warning if the applied values differ from the requested values

If no resolution is specified, the device’s current/default configuration is preserved.

---

### Streaming and probe mode

#### Normal capture

- On successful `open()`, acquisition is started with `self._acquirer.start()`.

#### Fast-start probe mode (`fast_start`)

If `properties.gentl.fast_start` is `true`:

- the backend configures the device and persists identity/metadata,
- but does **not** start streaming.

This is intended for capability probing and faster startup of probe workers.

---

## Troubleshooting

### No `.cti` found

Common causes:

- Vendor GenTL producer not installed
- `GENICAM_GENTL64_PATH` / `GENICAM_GENTL32_PATH` not set (or missing the producer directory)
- Wrong producer bitness (32-bit vs 64-bit)

Fix options:

- Set `camera.properties.gentl.cti_file` (pin a CTI), or
- set `GENICAM_GENTL64_PATH` / `GENICAM_GENTL32_PATH`, or
- set `camera.properties.gentl.cti_search_paths` / `cti_dirs`

### Producer load failures with multiple CTIs installed

Some installed producers may be incompatible or broken on a given system.

- Check `properties.gentl.cti_files_failed`
- Pin a known-good producer:
  - `properties.gentl.cti_file`, or
  - `properties.gentl.cti_files`

### Cached CTIs went stale (auto re-discovery)

If `properties.gentl.cti_files_source == "auto"`, stale cached CTI paths will trigger fallback to discovery automatically.
If you pinned CTIs as a user override and paths no longer exist, `open()` will fail (by design).

### Timeouts

- Increase `properties.gentl.timeout` (seconds)
- Reduce frame rate or resolution
- Check transport bandwidth (GigE: MTU/jumbo frames, NIC direct connection, etc.)

### Pixel format errors

- Inspect available formats via vendor tools or by checking `PixelFormat.symbolics`
- Try a simpler format such as `Mono8`

---

## Example configuration

```json
{
  "camera": {
    "backend": "gentl",
    "fps": 60.0,
    "exposure": 8000,
    "gain": 10.0,
    "width": 1920,
    "height": 1080,
    "properties": {
      "gentl": {
        "device_id": "serial:40312345",
        "pixel_format": "Mono8",
        "timeout": 3.0,
        "rotate": 0,
        "crop": [0, 0, 0, 0]
      }
    }
  }
}
```

---

## Resources

- [Harvesters installation documentation](https://harvesters.readthedocs.io/en/latest/INSTALL.html)
- [GenTL producer discovery variables](https://www.mvtec.com/products/interfaces/documentation/view/1307-standard-13-mvtecdoc-genicamtl) (`GENICAM_GENTL32_PATH` / `GENICAM_GENTL64_PATH`) and usage notes
- [GenTL producer path separators and CTI overview](https://softwareservices.flir.com/Spinnaker/latest/_gen_i_cam_gen_t_l.html) (example vendor documentation)
