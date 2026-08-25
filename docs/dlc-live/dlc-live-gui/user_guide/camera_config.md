---
deeplabcut:
  last_metadata_updated: '2026-08-19'
  last_verified: '2026-08-19'
  ignore: false
---

(file:dlclivegui-camera-configuration)=

# Camera configuration

The **Configure Cameras** dialog is used to add cameras to the application, adjust capture settings, and verify each camera with a live preview.

```{important}
The application currently allows **up to four** enabled cameras.

For backend installation, supported camera families, and backend-specific limitations, see:

{ref}`file:dlclivegui-camera-support`.
```

## Open the camera configuration dialog

From the main window, select **Configure Cameras…** in the **Camera** section of the Controls panel.

The dialog contains:

- **Active cameras**: Cameras included in the current application configuration
- **Available cameras**: Devices discovered for the selected backend
- **Camera settings and preview**: Settings, detected values, trigger controls, and a live preview for the selected active camera

```{important}
The main window live preview must be stopped before opening the camera configuration dialog.
```

## Example configuration

1. Select the desired camera **Backend**.
   - A backend must be properly installed to appear in the list
   - See {ref}`file:dlclivegui-camera-support` for backend installation
1. Wait for discovery to complete, or select **Refresh** to scan again
1. Select a device under **Available cameras**
1. Select **Add Camera** or double-click the device
1. Select the camera under **Active cameras**
1. Adjust capture, output, crop, or trigger settings
1. Select **Apply Settings**
1. Start the configuration preview and verify the image and reported values
1. Repeat for any additional cameras
1. Select **OK** to apply the complete camera configuration to the main window

## Discover cameras

### Select a backend

Changing the backend starts a new discovery scan.
The available devices shown in the dialog belong to the currently selected backend.

```{note}
Camera capabilities **vary by backend, camera model and driver.**

The GUI enables, disables, or marks controls as best-effort according to the selected backend's reported capabilities.

There are many camera functionalities **currently unsupported by the GUI**, though they may be available on the camera hardware or SDK.

If you need a specific camera feature on a specific backend, **please let us know** so we can improve our support selectively.
```

### Refresh the device list

Select **Refresh** to scan the selected backend again.

### Device identity

When supported by the backend, the application stores a stable device identity such as a serial number or device ID.
This helps reconnect the configuration to the same physical camera if device enumeration order changes.

If stable identity is unavailable, the backend may fall back to the device index.

```{warning}
A device index is not necessarily permanent. Connecting, disconnecting, or reordering devices can change enumeration order. Verify the selected physical camera after hardware changes.
```

## Manage active cameras

### Add a camera

Select a detected device under **Available cameras**, then select **Add Camera**.

```{tip}
You can also double-click a detected device to add it.
```

### Enable or disable a camera

Use the **Enabled** checkbox for the selected active camera.

### Remove a camera

Select a camera under **Active cameras**, then select **Remove Camera**.

Removing a camera deletes it from the working camera configuration.
It can be added again later.

### Reorder cameras

Use the move-up and move-down controls to change **active camera order**.

```{important}
Camera order can affect presentation order in the GUI, including the tiled multi-camera preview.
```

### Active camera labels

The active-camera list summarizes important state using labels:

- `✓` indicates an enabled camera.
- `○` indicates a disabled camera.
- `[external]`, `[follower]`, or `[master]` indicates an active trigger role.
- `[DLC]` identifies the camera selected for pose inference when applicable.
- `[Mono]` indicates that mono output is preserved.

## Camera identity and detected values

The settings area displays read-only identity and probe information for the selected camera, including values such as:

- Camera name
- Device ID
- Device index
- Backend
- Detected resolution
- Detected frame rate
- Camera pixel or output format

The application probes the selected camera to retrieve available runtime information.
Detected values describe what the backend reports and may differ from the values requested in the editable controls.

```{note}
A blank or unavailable detected value does not always mean the camera cannot be used.
Some backends and drivers do not reliably expose all runtime properties.
```

### Requested versus detected values

Editable fields represent the requested configuration.
Detected labels represent values reported by the opened camera.

For example:

- A requested frame rate may be adjusted to a nearby supported value.
- A requested resolution may be constrained by camera increments or supported modes.
- A frame rate of `0` represents **Auto**, meaning the backend does not force a specific value.

Always verify the detected values and live preview after applying settings.

## Capture settings

### Resolution

Use **Width** and **Height** to request the capture resolution.

A value of `0` represents **Auto** or the device default.
Camera backends may clamp or adjust requested dimensions according to supported ranges and increments.

```{important}
Recording requires a constant frame size. Do not externally change camera resolution during an active recording.
```

### Frame rate

Use **FPS** to request the camera frame rate.

- A positive value requests a specific frame rate.
- `0` leaves frame-rate selection to the camera or backend.

When the backend reports an actual frame rate, the configuration preview adjusts its update cadence accordingly.
If the reported value differs from the request, the GUI may display the device-supported value.

### Exposure

Use **Exposure** to request camera exposure. A value of `0` leaves exposure automatic or unchanged, depending on the backend.

Exposure units and supported ranges depend on the backend and camera model. For supported Basler and GenTL cameras, exposure is commonly represented in microseconds.

```{important}
Long exposure times can limit achievable frame rate.
When a requested frame rate cannot be reached, verify that exposure duration is short enough for the intended frame interval.
```

### Gain

Use **Gain** to request camera gain. A value of `0` leaves gain automatic or unchanged, depending on the backend.
Higher gain can brighten an image but may also increase image noise. Verify the result in the configuration preview.

### Rotation

Use **Rotation** to rotate frames for display and downstream processing.
Rotation is applied by the application and can be updated without reopening the camera preview.

### Crop coordinates

Use `x0`, `y0`, `x1`, and `y1` to define a rectangular crop:

- `x0`, `y0`: Top-left corner
- `x1`, `y1`: Bottom-right corner

A valid crop requires `x1 > x0` and `y1 > y0`.

Coordinates are clamped to the current frame dimensions during preview.

```{tip}
Click and drag left/right on a coordinate input value to adjust pixel-by-pixel.
Hold **Ctrl** or **Shift** for faster/slower adjustments respectively.
```

## Preserve Mono output

Enable **Preserve mono frames** when a supported monochrome camera should produce two-dimensional grayscale frames instead of expanding them to three-channel color.

```{important}
Conversion to color can slow down the capture pipeline when using several high-FPS/resolution cameras.

- Always **enable** mono preservation when you have **grayscale cameras and the backend supports it**
- Always **disable** mono preservation when you have **color cameras**
```

The probe may recommend mono preservation when it detects a monochrome camera and the backend supports mono output.

(sec:dlclivegui-trigger-settings)=

## Trigger settings

For camera backends that expose triggering support, select a configured camera and choose **Trigger Settings…**.

```{tip} When to use triggered cameras
Triggering cameras can be useful if you need precise camera synchronization or need to coordinate with other devices.

In principle, it makes a camera follow external signals rather than its own internal clock,
ensuring much tighter synchronization between multiple cameras or other devices.
```

Trigger configuration is currently available for the **Basler** and **GenTL** backends.

*The available fields remain **camera and driver-dependent**.*

- **Off / Free-run**: Disable triggering and acquire frames continuously.
- **External trigger**: Wait for hardware pulses on the selected input source.
- **Master output**: Keep the camera free-running and configure an output signal for another camera or device.
- **Follower**: Configure the camera as a synchronized input that follows an external trigger source. Similar to external trigger conceptually.

The active-camera list shows the configured role, such as `[external]`, `[follower]`, or `[master]`.

For **External trigger** and **Follower**, configure:

- **Trigger selector**: Usually `FrameStart` for area-scan cameras.
- **Trigger source**: Select `auto` or enter a camera-supported source such as `Line1` or `Line2`.
- **Activation**: Choose the signal condition, such as `RisingEdge` or `FallingEdge`.
- **Read timeout**: Maximum frame wait in seconds. The backend may use shorter individual waits to keep preview shutdown responsive.

```{note}
The read timeout is a maximum wait for a single frame.

If the camera does not receive a valid trigger signal within this time, the backend ends the current wait and reports a timeout error.
It does not disable the trigger configuration or permanently stop the camera.
```

For **Master output**, configure:

- **Output line**: The camera output line, such as `Line2`.
- **Output source**: The signal routed to that line, such as `ExposureActive`.
- **GenTL strobe options**: Compatible GenTL cameras may additionally expose strobe polarity, operation, duration, and delay. A value of **Default** leaves duration or delay unset.

```{important}
The trigger dialog provides backend-specific suggestions, not a guarantee that the selected camera supports every displayed value.

Enable **Strict mode** when missing or unsupported required trigger features should prevent the camera from opening.
With strict mode disabled, the backend applies supported settings best-effort and may disable an unsupported trigger configuration.
```

```{tip}
Start the camera preview after changing trigger settings to verify the configuration. An externally triggered camera may wait or time out until it receives a valid pulse.
```

## Apply, reset, and preview settings

### Apply Settings

Select **Apply Settings** to validate and store the current camera settings.
Pending edits are also applied automatically before actions such as switching cameras, adding a camera, starting preview, or closing the dialog with **OK**.
If validation fails, remain on the current camera and correct the reported setting.

### Automated preview restart

Camera-side changes require the backend to reopen. The configuration preview restarts for changes such as:

- Width or height
- Frame rate
- Exposure
- Gain
- Preserve-mono mode
- Trigger settings

```{note}
Rotation and crop settings are applied by the preview path and do not require reopening the camera.
Note that the **main GUI preview** may **use the SDK native rotation and crop** instead if the backend supports it.
```

### Reset Settings

Select **Reset Settings** to clear requested capture settings.

```{warning}
Reset updates the working camera configuration.
```

### Configuration preview

Select **Start Preview** to open the selected active camera.
The preview displays status messages while loading and reports available information such as requested versus actual resolution, device identity, pixel format, and backend output format.

Use **Stop Preview** when finished.
Select **Cancel Loading** if camera startup is taking too long or seems to hang.

```{important}
An externally triggered camera may show no frames until valid trigger pulses arrive.
This can be expected behavior rather than a camera failure.

Check your trigger settings, wiring and signal source.
```

## Save or discard the configuration

### OK

Select **OK** to apply any pending edits and return the working camera configuration to the main window

If cameras have been added but none are enabled, the dialog asks you to enable at least one camera or remove all cameras.

### Cancel

Select **Cancel** or close the dialog to discard unaccepted changes and stop active preview, discovery, and probe work.

## Multi-camera recommendations

For a reliable multi-camera setup:

1. Add and preview each camera individually
1. Confirm stable device identity where supported
1. Use consistent resolution and frame-rate targets when appropriate for the experiment
1. Keep exposure short enough for the intended acquisition rate
1. Verify trigger wiring and roles before starting the main preview
1. Enable only cameras required for the session
1. Save the completed application configuration for reproducibility

```{note}
The camera selected for pose inference is configured separately in the main window.
All enabled cameras can still participate in preview and recording.
```

## Troubleshooting

### No cameras detected

- Confirm the camera is powered and connected.
- Confirm the correct backend is selected.
- Install and configure the required vendor SDK or transport layer as well as driver files.
  - If a GenTL device appears in OpenCV but not in the GenTL backend, check that the vendor CTI files are installed and accessible.
- Select **Refresh** after changing the connection or driver setup.
- Close other applications that may have exclusive access to the camera.

See {ref}`file:dlclivegui-camera-support` for backend-specific installation and troubleshooting guidance.

### Camera opens but no frames arrive

- For free-running cameras, verify exposure, frame rate, and camera acquisition mode.
- For triggered cameras, verify that valid trigger pulses reach the selected input.
- Check the preview status for timeout or backend errors.
- Reset the camera settings and test with device defaults.

### Requested values are not applied exactly

The camera or backend may constrain values to supported ranges, increments, or capture modes.
Compare the requested settings with detected values and preview status.
OpenCV controls are particularly dependent on the operating-system driver and camera implementation.

### Mono camera recording is dropping frames at high FPS

Enable **Preserve mono frames** if the backend and camera support native mono output. Otherwise, the backend may convert the frame to a three-channel format for compatibility.
Also try the faster encoding presets in the main window recording settings.

### Trigger settings are ignored or disabled

- Confirm the backend has trigger support in the SDK **and** the GUI
  - Please open an issue if you need your camera to use triggering but the GUI does not expose it
- Verify that the selected camera exposes the requested trigger nodes and values
- Start with strict mode disabled to test best-effort behavior
- Enable strict mode when unsupported settings should fail visibly
- Consult the camera manufacturer's documentation for line assignments and electrical requirements

## Related pages

- {ref}`file:dlclivegui-camera-support`: Supported backends, installation, and backend-specific limitations
- {ref}`file:dlclivegui-timestamp-format`: Software and hardware timestamps recorded with video
- {ref}`sec:dlclivegui-recording-paths-info`: Recording output paths and naming
- {ref}`deeplabcut-live`: DeepLabCut Live inference documentation
