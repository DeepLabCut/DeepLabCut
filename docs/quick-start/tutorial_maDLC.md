---
deeplabcut:
  last_content_updated: '2025-02-28'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: viable
  recommendation: keep
---

(file:multi-animal-quick-start)=

# Multi-animal at a glance

This page summarizes the main DeepLabCut functions used in a standard multi-animal
2D pose-estimation workflow.

## GUI workflow

DeepLabCut provides a full graphical user interface. To launch it, run:

```bash
python -m deeplabcut
```

Then follow the tabs in the Project Manager GUI.

## Python workflow

The same workflow can also be run from Python. Start by importing DeepLabCut:

```python
import deeplabcut
```

### 1. Create a project

```python
project_name = "cutemice"
experimenter = "teamdlc"
video_paths = ["/absolute/path/to/video_file.mp4"]

config_path = deeplabcut.create_new_project(
    project_name,
    experimenter,
    video_paths,
    multianimal=True,
    copy_videos=True,
)
```

```{note}
Use absolute paths to your video file(s).

On Windows, you can quickly copy a file path with <kbd>Shift</kbd> +
<kbd>Right click</kbd> and **Copy as path**. On macOS, use
<kbd>Option</kbd> + <kbd>Right click</kbd> and **Copy as Pathname**.
On Ubuntu, copying the file usually also copies its path to the clipboard.
```

The returned `config_path` is the full path to the project `config.yaml` file. This
variable is used throughout the rest of the workflow.

### 2. Configure the project

Open the generated `config.yaml` file and edit it for your experiment.

At this stage, define the animal identities and keypoints you want to track. For
multi-animal projects, the most important fields are typically:

- `individuals`
- `multianimalbodyparts`
- `uniquebodyparts`
- `identity`
- `numframes2pick`

```{important}
Do not include spaces in the names of individuals, body parts, multi-animal body
parts, or unique body parts.
```

```{tip}
You can also adjust the number of frames to extract in the next step by editing
`numframes2pick` in `config.yaml`.
```

### 3. Extract video frames to annotate

```python
deeplabcut.extract_frames(
    config_path,
    mode="automatic",
    algo="kmeans",
    userfeedback=False,
)
```

```{tip}
For robust training data, it is usually better to extract a few informative frames
from many videos than many similar frames from a single video.
```

### 4. Annotate frames

```python
deeplabcut.label_frames(config_path)
```

### 5. Check annotated frames

```python
deeplabcut.check_labels(
    config_path,
    draw_skeleton=False,
)
```

Use the generated labeled images to visually confirm that the annotations were saved
correctly.

### 6. Create the training dataset

```python
deeplabcut.create_multianimaltraining_dataset(
    config_path,
    num_shuffles=1,
    net_type="dlcrnet_ms5",
)
```

### 7. Train the network

Choose the example corresponding to the engine you are using.

#### PyTorch engine

```python
deeplabcut.train_network(
    config_path,
    device="cuda",
    save_epochs=5,
    epochs=200,
)
```

#### TensorFlow engine

```python
deeplabcut.train_network(
    config_path,
    saveiters=10000,
    maxiters=50000,
    allow_growth=True,
)
```

### 8. Evaluate the network

```python
deeplabcut.evaluate_network(
    config_path,
    plotting=True,
)
```

Inspect the evaluation results before moving on to video analysis. For multi-animal
projects, pay particular attention to detection quality and tracking readiness.

### 9. Analyze videos

```python
deeplabcut.analyze_videos(
    config_path,
    video_paths,
    auto_track=True,
)
```

```{note}
With `auto_track=True`, DeepLabCut automatically performs the tracking steps needed to
produce the final `.h5` output file. Use the manual tracking steps below only if you
need to inspect detections or tune tracking parameters for your dataset.
```

### 10. Convert detections to tracklets

This step performs spatial and local temporal grouping, assembling body parts into
tracklets frame by frame.

```python
deeplabcut.convert_detections2tracklets(
    config_path,
    video_paths,
    track_method="ellipse",
)
```

### 11. Stitch tracklets into trajectories

This step reconstructs full animal trajectories from the tracklets.

```python
deeplabcut.stitch_tracklets(
    config_path,
    video_paths,
    track_method="ellipse",
    min_length=5,
)
```

### 12. Create labeled videos

```python
deeplabcut.create_labeled_video(
    config_path,
    video_paths,
    color_by="individual",
    keypoints_only=False,
    trailpoints=10,
    draw_skeleton=False,
    track_method="ellipse",
)
```

This creates a video with the predicted keypoints overlaid. For multi-animal projects,
`color_by="individual"` is useful for visually checking identity assignment and
trajectory consistency.
