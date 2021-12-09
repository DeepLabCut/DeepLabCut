# Multi-animal pose estimation with DeepLabCut: A 5-minute tutorial

**Import deeplabcut**
```python
import deeplabcut
```

**Create a project**
```python
project_name = "cutemice"
experimenter = "teamdlc"
video_path = "path_to_a_video_file"
config_path = deeplabcut.create_new_project(
    project_name,
    experimenter,
    [video_path],
    multianimal=True,
    copy_videos=True,
)
```
> **_NOTE:_**  Make sure to specify the absolute path to the video file.
> It is quickly obtained on Windows with <kbd>⇧ Shift</kbd>+<kbd>Right click</kbd> and `Copy as path`,
> and on Mac with <kbd>⌥ Option</kbd>+<kbd>Right click</kbd> and `Copy as Pathname`.
> Ubuntu users only need to copy the file and its path gets added to the clipboard.

> Next, you can set a variable for the config_path: 'Full path of the project configuration file*'

**Edit the config.ymal file to set up your project**

**Extract video frames to annotate**
```python
deeplabcut.extract_frames(
    config_path,
    mode="automatic",
    algo="kmeans",
    userfeedback=False,
)
```

**Annotate Frames**
```python
deeplabcut.label_frames(config_path)
```


**Visually check annotated frames**
```python
deeplabcut.check_labels(
    config_path,
    draw_skeleton=False,
)
```

**Create the training dataset**
```python
deeplabcut.create_multianimaltraining_dataset(
    config_path,
    num_shuffles=1,
    net_type="dlcrnet_ms5",
)
```

**Train the network**
```python
deeplabcut.train_network(
    config_path,
    saveiters=10000,
    maxiters=50000,
    allow_growth=True,
)
```

**Evaluate the network**
```python
deeplabcut.evaluate_network(
    config_path,
    plotting=True,
)
```

**Analyze a video (extracts detections and association costs)**
```python
deeplabcut.analyze_videos(
    config_path,
    [video],
)
```


**Spatial and (locally) temporal grouping: Track body part assemblies frame-by-frame**
```python
deeplabcut.convert_detections2tracklets(
    config_path,
    [video],
    track_method="ellipse",
)
```


**Reconstruct full animal trajectories (tracks from tracklets)**
```python
deeplabcut.stitch_tracklets(
    config_path,
    [video],
    track_method="ellipse",
    min_length=5,
)
```


**Create a pretty video output**
```python
deeplabcut.create_labeled_video(
    config_path,
    [video],
    color_by="individual",
    keypoints_only=False,
    trailpoints=10,
    draw_skeleton=False,
    track_method="ellipse",
)
```
