# Video Frame Timestamp Format

When recording video, the application automatically saves frame timestamps to a JSON file alongside the video file.

## File Naming

For a video file named `recording_2025-10-23_143052.mp4`, the timestamp file will be:
```
recording_2025-10-23_143052.mp4_timestamps.json
```

## JSON Structure

```json
{
  "video_file": "recording_2025-10-23_143052.mp4",
  "num_frames": 1500,
  "timestamps": [
    1729693852.123456,
    1729693852.156789,
    1729693852.190123,
    ...
  ],
  "start_time": 1729693852.123456,
  "end_time": 1729693902.123456,
  "duration_seconds": 50.0
}
```

## Fields

- **video_file**: Name of the associated video file
- **num_frames**: Total number of frames recorded
- **timestamps**: Array of Unix timestamps (seconds since epoch with microsecond precision) for each frame
- **start_time**: Timestamp of the first frame
- **end_time**: Timestamp of the last frame
- **duration_seconds**: Total recording duration in seconds

## Usage

The timestamps correspond to the exact time each frame was captured by the camera (from `FrameData.timestamp`). This allows precise synchronization with:

- DLC pose estimation results
- External sensors or triggers
- Other data streams recorded during the same session

## Example: Loading Timestamps in Python

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

# Calculate average frame rate
avg_fps = data['num_frames'] / data['duration_seconds']
print(f"Average FPS: {avg_fps:.2f}")

# Access individual frame timestamps
for frame_idx, timestamp in enumerate(data['timestamps']):
    print(f"Frame {frame_idx}: {timestamp}")
```

## Notes

- Timestamps use `time.time()` which returns Unix epoch time with high precision
- Frame timestamps are captured when frames arrive from the camera, before any processing
- If frames are dropped due to queue overflow, those frames will not have timestamps in the array
- The timestamp array length should match the number of frames in the video file
