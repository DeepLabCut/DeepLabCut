# Input/output manipulations with DeepLabCut

## Analyzing very large videos in chunks
Analyzing hour-long videos may take a while, but the task can be
conveniently broken down into the analysis of smaller video clips:

```python
import deeplabcut
import os
from deeplabcut.utils.auxfun_videos import VideoWriter

_, ext = os.path.splitext(video_path)
vid = VideoWriter(video_path)
clips = vid.split(n_splits=10)
deeplabcut.analyze_videos(config_path, clips, ext)
```
