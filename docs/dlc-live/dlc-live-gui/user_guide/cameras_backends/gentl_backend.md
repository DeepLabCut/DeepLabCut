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

## Features

- Image acquisition via **Harvesters** (GenTL consumer for GenICam-compliant devices).
- Works with vendor GenTL Producers (`.cti` files) for USB3 Vision / GigE Vision / frame grabbers (depending on producer).
- Device discovery (rich) and stable identity via:
  - `device_id = "serial:<SERIAL>"` when a serial number is available
  - `device_id = "fp:<fingerprint...>"` as a best-effort fallback when serials are missing/ambiguous
- Configurable exposure, gain, frame rate, and resolution through the device GenApi node map (best-effort).
- Pixel-format handling with conversion to **BGR (8-bit)** for consistency:
  - Mono formats are converted to BGR
  - `RGB8` is converted to BGR
  - Non-8-bit frames are scaled down to 8-bit

---

## Installation

### 1) Install Harvesters

Install Harvesters into the same Python environment as your GUI:

```bash
pip install harvesters
```

Harvesters is a Python library that performs image acquisition through GenTL Producers.

### 2) Install a GenTL Producer (`.cti`)

A GenTL Producer is a vendor-provided library that exposes cameras to GenTL consumer applications.
It is typically distributed as part of the camera vendor SDK or framegrabber SDK.

```{note}
GenTL Producers are identified by files ending in `.cti`.
```

After installing the vendor SDK, locate the `.cti` file (for example, your vendor may install it under a program files directory on Windows or under `/opt/...` on Linux).

### 3) Make producers discoverable (environment variables)

Most GenTL consumers (including many third-party tools) locate producers via the standard environment variables:

- `GENICAM_GENTL64_PATH` (64-bit producers)
- `GENICAM_GENTL32_PATH` (32-bit producers)

The `.cti` file must be located in a directory referenced by these variables, or the application must be configured with a full path to the `.cti` file.

If you have multiple producers installed, separate entries with:

- `;` on Windows
- `:` on Linux/macOS (UNIX-like)

```{tip}
Many vendor installers set GENICAM_GENTL64_PATH automatically. If your camera is not discovered, explicitly set the variable (or provide `cti_file` / `cti_files` in the backend configuration as described below).
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

### Default behavior: load all available producers

By default, the backend will **try to load all available** GenTL Producers (`.cti`) it can find.
This is intentional: many systems have multiple producers installed (different transports/vendors).

- If a producer fails to load, the backend continues and attempts to load the others.
- The backend persists load diagnostics to help troubleshooting:
  - `cti_files`: all resolved candidate CTIs
  - `cti_files_loaded`: CTIs successfully added to Harvesters
  - `cti_files_failed`: list of `{cti, error}` entries for producers that failed to load

Startup fails only when:

- **No CTI files** can be found, or
- **No CTI files** can be loaded successfully, or
- No devices are detected after loading producers.

### How CTIs are resolved (precedence)

The backend resolves CTI locations in this order:

1. `properties.gentl.cti_files` (explicit list of CTI paths)
2. `properties.gentl.cti_file` (explicit single CTI path)
3. `properties.cti_files` (legacy explicit list)
4. `properties.cti_file` (legacy explicit single path)
5. Discovery using:
   - `GENICAM_GENTL64_PATH` / `GENICAM_GENTL32_PATH`
   - `properties.gentl.cti_search_paths` (glob patterns)
   - `properties.gentl.cti_dirs` (extra directories to scan for `*.cti`)
   - built-in fallback patterns for some common Windows installations

```{note}
If you experience issues with a specific vendor producer, you can pin a known-good producer by setting `properties.gentl.cti_file` (or `cti_files`).
```

### Provide an explicit CTI file path

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

### Provide an explicit list of CTI producers

Use this when you want to control exactly which producers are loaded:

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

### Provide CTI search patterns

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

### Provide extra CTI directories

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

Notes:

- The backend includes built-in fallback patterns for some common Windows installations (e.g. The Imaging Source). If those do not apply to your setup, setting `cti_file` / `cti_files` explicitly is the most reliable option.

---

## Camera selection

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

The backend supports `properties.gentl.device_id`:

- `serial:<SERIAL>` when a serial number is available
- `fp:<...>` when serial numbers are missing; rebinding uses discovery to map the fingerprint back to a current index

```json
{
  "camera": {
    "backend": "gentl",
    "index": 0,
    "properties": {
      "gentl": {
        "device_id": "serial:40312345"
      }
    }
  }
}
```

Selection order in `open()`:

1. Exact match of `device_id` against discovered devices
2. If `device_id` starts with `serial:`, match by serial number
3. Legacy serial keys (`serial_number` / `serial`) if present
4. Fallback to `index`

If a serial substring matches multiple cameras, the backend raises an “ambiguous” error.

---

## Full properties and configuration

GenTL backend options live under `properties.gentl`.

### Related camera settings (shared across backends)

- `width` (int): requested image width; `0` means keep device default
- `height` (int): requested image height; `0` means keep device default
- `fps` (float): requested frame rate; `0.0` means do not set
- `exposure` (float): exposure time; `<= 0` means leave as-is / auto
- `gain` (float): gain value; `<= 0` means leave as-is / auto

### GenTL namespace options (`properties.gentl`)

Core:

- `cti_file` (string): full path to a GenTL Producer `.cti` file
- `cti_files` (list[string]): explicit list of producer `.cti` files to load
- `cti_search_paths` (list[string] or string): glob patterns used to locate `.cti` files
- `cti_dirs` (list[string] or string): extra directories to scan for `*.cti`

Selection / identity:

- `device_id` (string): stable identifier (`serial:...` or `fp:...`)
- `serial_number` / `serial` (string): legacy selection helpers

Acquisition:

- `pixel_format` (string, default: `Mono8`): requested `PixelFormat` symbolic
- `timeout` (float, default: `2.0`): acquisition timeout in seconds (used by `fetch(timeout=...)`)

Transforms:

- `rotate` (int, default: `0`): rotate output by 0/90/180/270
- `crop` ([top, bottom, left, right]): crop rectangle

Probe / telemetry:

- `fast_start` (bool, default: false): probe-mode hint; when true, `open()` does not start acquisition
- `cti_files_loaded` (list[string]): populated automatically after open
- `cti_files_failed` (list[object]): populated automatically after open; each entry has `cti` and `error`

---

## Pixel format

- The backend attempts to set `node_map.PixelFormat` to the configured `pixel_format` if it is present in `PixelFormat.symbolics`.
- If the requested format is not available, it logs a warning and continues.

Supported values depend on the camera and producer.

Internally, frames are normalized to **BGR (8-bit)**:

- Mono images become BGR via grayscale-to-BGR conversion
- `RGB8` is converted to BGR
- Higher bit-depth images are scaled to 8-bit based on the frame’s max value (per frame)

---

## Exposure and gain

Best-effort behavior (depends on the GenTL producer + camera GenApi implementation):

- If exposure is set (value > 0):
  - Attempts to disable `ExposureAuto` by setting it to `Off`
  - Tries `ExposureTime` (preferred) then `Exposure`
- If gain is set (value > 0):
  - Attempts to disable `GainAuto` by setting it to `Off`
  - Tries `Gain`

If the relevant nodes are missing or read-only, the backend logs a warning and continues.

---

## Frame rate (FPS)

If `fps > 0`:

- The backend attempts to enable frame-rate control via `AcquisitionFrameRateEnable` (or similar).
- Then it tries to set one of these nodes (first that works):
  - `AcquisitionFrameRate`
  - `ResultingFrameRate`
  - `AcquisitionFrameRateAbs`

The backend also tries to read back `ResultingFrameRate` as the “actual fps” for GUI telemetry.

---

## Resolution handling

Resolution is only applied when explicitly requested.

- If `width` and `height` are both > 0, the backend attempts to set `node_map.Width` and `node_map.Height`.
- It clamps to node min/max and snaps down to the nearest valid increment (`inc`) when available.
- A warning is logged if the applied values differ from the requested values.

If no resolution is specified, the camera’s current/default configuration is preserved.

---

## Streaming and probe mode

### Normal capture

- On successful `open()`, acquisition is started with `self._acquirer.start()`.

### Fast-start probe mode

If `properties.gentl.fast_start` is `true`:

- The backend configures the device and persists identity fields, but does **not** start streaming.
- This is intended for capability probing and faster startup of probe workers.

---

## Troubleshooting

### No .cti found / cameras not detected

Common causes:

- Vendor GenTL producer not installed
- Environment variable `GENICAM_GENTL64_PATH` not set (or not including the producer directory)
- Wrong producer bitness (32-bit vs 64-bit)

GenTL producers must be discoverable via `GENICAM_GENTL64_PATH` / `GENICAM_GENTL32_PATH`, or you must provide an explicit CTI file path.

### Producer load failures

If you have multiple producers installed, some may be incompatible or broken on your system.

- Check `properties.gentl.cti_files_failed` after a failed open
- Pin a known-good producer:
  - set `properties.gentl.cti_file`, or
  - set `properties.gentl.cti_files` to a curated list

### Timeouts

- Increase `properties.gentl.timeout` (seconds)
- Reduce frame rate or resolution
- Check transport bandwidth (GigE: MTU/jumbo frames, direct NIC connection, etc.)

### Pixel format errors

- Inspect available formats via your vendor tools or by checking `PixelFormat.symbolics`.
- Try a simpler format such as `Mono8`.

---

## Example configuration

```json
{
  "camera": {
    "backend": "gentl",
    "index": 0,
    "fps": 60.0,
    "exposure": 8000,
    "gain": 10.0,
    "width": 1920,
    "height": 1080,
    "properties": {
      "gentl": {
        "cti_file": "C:/Path/To/Your/Producer.cti",
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
