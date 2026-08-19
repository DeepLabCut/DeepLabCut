---
deeplabcut:
  last_metadata_updated: '2026-03-17'
  ignore: false
---

(file:dlclivegui-timestamp-format)=

# Video timestamp format

When recording videos, the application automatically saves frame timestamps to a JSON file alongside each video file.
The file includes an app-side software timestamp for every written frame and, when supported by the camera backend, optional hardware-generated timestamp metadata.

## File naming

```{note}
If you would like more information on the output path structure and settings,
please refer to the {ref}`sec:dlclivegui-recording-paths-info` section.
```

For a video file named `recording_2025-10-23_143052.mp4`, the timestamp file will be:

```txt
recording_2025-10-23_143052.mp4_timestamps.json
```

## Important notes

- Software timestamp records are added only after a frame is successfully written by the encoder.
  - It is therefore coupled **to the recording rather than inference or GUI display**.
  - **It should not be interpreted as a model-side inference timestamp or a camera acquisition timestamp.**
- Frames dropped before writing are not included in `frame_timestamps`.
- Hardware timestamp availability may differ by backend, camera model, and driver.
- The encoded video is written with a fixed input frame rate configured when recording starts.
  - The timestamps reflect acquisition and enqueue timing and may not perfectly match encoded frame pacing, especially if frames are dropped or capture timing varies.

## JSON structure

Timestamp files currently use the following format:

```json
{
  "video_file": "recording_2025-10-23_143052.mp4",
  "num_frames": 2,
  "timestamp_sources": {
    "software_timestamp": {
      "source": "host_time.time",
      "backend": "host",
      "kind": "software_wall_clock",
      "timebase": "Unix epoch",
      "unit": "seconds",
      "description": "Host-side software timestamp captured during acquisition."
    },
    "hardware_timestamp": {
      "source": "grab_result.GetTimeStamp",
      "backend": "basler",
      "default_reported": "seconds",
      "raw_unit": "ticks",
      "tick_frequency_hz": 1000000000.0,
      "timebase": "Basler camera timestamp counter",
      "kind": "camera_clock",
      "extra": {
        "tick_frequency_source": "GevTimestampTickFrequency"
      }
    }
  },
  "frame_timestamps": [
    {
      "frame_index": 0,
      "software_timestamp": 1729693852.123456,
      "hardware_timestamp": {
        "seconds": 0.123456789,
        "raw_value": 123456789
      },
      "hardware_timestamp_default": 0.123456789
    },
    {
      "frame_index": 1,
      "software_timestamp": 1729693852.156789,
      "hardware_timestamp": {
        "seconds": 0.156789123,
        "raw_value": 156789123
      },
      "hardware_timestamp_default": 0.156789123
    }
  ],
  "start_time": 1729693852.123456,
  "end_time": 1729693852.156789,
  "duration_seconds": 0.033333
}
```

If hardware timestamps are unavailable, `timestamp_sources.hardware_timestamp` is `null` and each frame record contains only `frame_index` and `software_timestamp`.

### Top-level fields

- **video_file**: Name of the associated video file.
- **num_frames**: Number of frames successfully written to the video and represented in `frame_timestamps`.
- **timestamp_sources**: Describes the software timestamp source and any available hardware timestamp source.
- **frame_timestamps**: Ordered per-frame timestamp records.
- **start_time**: Software timestamp of the first written frame.
- **end_time**: Software timestamp of the last written frame.
- **duration_seconds**: Difference between the last and first software timestamps.

### Per-frame fields

- **frame_index**: Zero-based index of the written frame.
- **software_timestamp**: Host-side Unix timestamp in seconds.
- **hardware_timestamp**: Optional backend-provided timestamp values. Defined values may include:
  - **seconds**: Hardware timestamp converted to seconds in the device timebase.
  - **wall_clock_time**: Unix timestamp, only when the backend can confidently provide a wall-clock value.
  - **raw_value**: Original device-specific timestamp value, such as camera clock ticks.
- **hardware_timestamp_default**: Optional convenience value selected by the backend's `default_reported` field.

### Timestamp source fields

The `timestamp_sources.hardware_timestamp` object describes how hardware values should be interpreted:

- **source**: Backend API or feature that produced the timestamp.
- **backend**: Camera backend name.
- **default_reported**: Per-frame hardware value recommended as the default, such as `seconds` or `raw_value`.
- **raw_unit**: Unit of the original hardware value, such as `ticks`.
- **tick_frequency_hz**: Frequency used to convert ticks to seconds, when available.
- **timebase**: Description of the hardware clock.
- **kind**: Timestamp category, such as `camera_clock`.
- **extra**: Additional backend-specific source information.

## Software and hardware timestamps

### Software timestamp

Every written frame has a `software_timestamp`. It is normally supplied by the camera acquisition path.
If no timestamp is provided to the recorder, the recorder falls back to `time.time()` when the frame is enqueued.

Software timestamps use Unix epoch seconds and can therefore be compared directly with other host-side wall-clock data.

### Hardware timestamp

Hardware timestamps are optional and **backend-dependent**.
They supplement the software timestamp rather than replace it.

- The Basler backend provides hardware timestamp metadata on supported cameras using `grab_result.GetTimeStamp()`.
  - The raw value is stored in ticks.
  - When a valid tick frequency is available, the JSON also contains a value converted to seconds.

```{important}
A hardware timestamp expressed in seconds is not necessarily Unix time. For example, a Basler `camera_clock` value is relative to the camera's own timestamp counter unless the backend explicitly reports a wall-clock timebase.

Do not directly subtract a camera-clock timestamp from a Unix software timestamp.
```

```{note}
Hardware timestamp support is best-effort. Cameras and backends that do not expose usable hardware timestamps still record software timestamps normally.
```

## Usage

The timestamp JSON can help support synchronization with:

- DLC pose estimation results
- External sensors or triggers
- Other data streams recorded during the same session

Use `software_timestamp` for host wall-clock comparisons.
Use hardware timestamps when device-level timing is required, while **accounting for the hardware clock's timebase and synchronization method.**

### Loading timestamps in Python

```python
import json
from datetime import datetime

with open("recording_2025-10-23_143052.mp4_timestamps.json", "r") as f:
    data = json.load(f)

print(f"Schema: {data['schema_version']}")
print(f"Video: {data['video_file']}")
print(f"Total frames: {data['num_frames']}")
print(f"Duration: {data['duration_seconds']:.6f} seconds")

# Convert the first software timestamp to local readable time
first_record = data["frame_timestamps"][0]
start_dt = datetime.fromtimestamp(first_record["software_timestamp"])
print(f"Recording started: {start_dt.isoformat()}")

# Calculate average frame rate from software timestamps.
if data["duration_seconds"] > 0 and data["num_frames"] > 1:
    avg_fps = (data["num_frames"] - 1) / data["duration_seconds"]
else:
    avg_fps = 0.0
print(f"Average FPS: {avg_fps:.2f}")

# Access software and optional hardware timestamps per frame.
for record in data["frame_timestamps"]:
    frame_index = record["frame_index"]
    software_timestamp = record["software_timestamp"]
    hardware_timestamp = record.get("hardware_timestamp")

    print(
        f"Frame {frame_index}: "
        f"software={software_timestamp}, "
        f"hardware={hardware_timestamp}"
    )
```

## Additional notes

- The `frame_timestamps` array length should match `num_frames` and the number of frames written to the video.
- `start_time`, `end_time`, and `duration_seconds` are calculated from software timestamps.
- Hardware timestamp source metadata is written once in `timestamp_sources`; changing timestamp sources during one recording is not expected.
