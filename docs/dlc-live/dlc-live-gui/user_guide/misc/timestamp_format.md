(file:dlclivegui-tinmestamp-format)=
# Video timestamp format

When recording videos, the application automatically saves frame timestamps to a JSON file alongside the video file.

## File naming

```{note}
If you would like more information on the ouput path structure and settings,
please refer to the {ref}`file:dlclivegui-recording-paths-info` section.
```

For a video file named `recording_2025-10-23_143052.mp4`, the timestamp file will be:


```
recording_2025-10-23_143052.mp4_timestamps.json
```

## JSON structure

```json
{
  "video_file": "recording_2025-10-23_143052.mp4",
  "num_frames": 1500,
  "timestamps": [
    1729693852.123456,
    1729693852.156789,
    1729693852.190123
  ],
  "start_time": 1729693852.123456,
  "end_time": 1729693902.123456,
  "duration_seconds": 50.0
}
```

### Fields

- **video_file**: Name of the associated video file
- **num_frames**: Total number of frames recorded
- **timestamps**: Array of Unix timestamps (seconds since epoch with microsecond precision) for each frame
- **start_time**: Timestamp of the first frame
- **end_time**: Timestamp of the last frame
- **duration_seconds**: Total recording duration in seconds

### Usage

The timestamps correspond to the time each frame was captured by the camera (from `FrameData.timestamp`) **when that timestamp is provided by the caller**.
If no timestamp is provided, the recorder falls back to `time.time()` at enqueue time.

This allows precise synchronization with:

- DLC pose estimation results
- External sensors or triggers
- Other data streams recorded during the same session

### Loading timestamps in Python

```python
import json
from datetime import datetime

# Load timestamps
with open('recording_2025-10-23_143052.mp4_timestamps.json', 'r') as f:
    data = json.load(f)

print(f"Video: {data['video_file']}")
print(f"Total frames: {data['num_frames']}")
print(f"Duration: {data['duration_seconds']:.2f} seconds")

# Convert first timestamp to human-readable format
start_dt = datetime.fromtimestamp(data['start_time'])
print(f"Recording started: {start_dt.isoformat()}")

# Calculate average frame rate (based on timestamps)
avg_fps = data['num_frames'] / data['duration_seconds'] if data['duration_seconds'] else 0.0
print(f"Average FPS: {avg_fps:.2f}")

# Access individual frame timestamps
for frame_idx, timestamp in enumerate(data['timestamps']):
    print(f"Frame {frame_idx}: {timestamp}")
```

## Notes

### About timestamps

- Timestamps use `time.time()` (Unix epoch seconds) when no explicit timestamp is supplied to the recorder.
- Frame timestamps are captured when frames are enqueued for writing (before encoding), and when provided by the caller can represent camera capture time.
- If frames are dropped due to queue overflow, those frames will not have timestamps in the array.
- The timestamp array length should match the number of frames in the video file.

The encoded video is written with a fixed input frame rate configured when recording starts.

The timestamps reflect capture/enqueue timing and may not perfectly match the encoded frame pacing, especially if frames are dropped or capture timing varies.

### About frame size mismatches


```{warning}
Frame size must remain constant for a recording session. If the recorder is configured with an expected `frame_size` and a frame with a different size is written, the recorder enters an error state to prevent encoder corruption:

- The mismatched frame is rejected (`write(...)` returns `False`)
- Subsequent `write(...)` calls will raise an exception indicating encoding failed
- Stop the recorder and start a new recording after fixing the frame size
```


```{note}
Frames are converted automatically for encoding:

- Non-`uint8` frames are scaled/clipped into the `uint8` range.
- Grayscale frames (`H x W`) are expanded to 3 channels (`H x W x 3`).
- Frames are made contiguous in memory before being passed to the encoder.
```
