---
deeplabcut:
  last_content_updated: '2025-06-30'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: viable
  recommendation: archive
  notes: This is a bit stuck between minimal guide and quick start, as the lack of explanations makes it more into a catalogue of commands (which is an API docs responsibility), and a proper quick start guide that gives users a proper sense of the workflow. This should either be expanded greatly or simply archived. For simplicity, I recommend archiving.
---

(file:single-animal-quick-start)=

# Single-animal at a glance

This page summarizes the main DeepLabCut functions used in a standard single-animal
2D pose-estimation workflow, from project creation to analyzed videos.

## Start Python

Open a terminal and start an interactive Python session, for example with IPython:

```bash
ipython
```

Then import DeepLabCut:

```python
import deeplabcut
```

## Workflow

### 1. Create a project

```python
project_name = "project_name"
experimenter = "experimenter"
video_paths = [
    "/absolute/path/to/video_1.mp4",
    "/absolute/path/to/video_2.mp4",
]

config_path = deeplabcut.create_new_project(
    project_name,
    experimenter,
    video_paths,
    copy_videos=True,
)
```

```{note}
Use absolute paths to your video files. The returned `config_path` is the full path to
the project `config.yaml` file and is used throughout the rest of the workflow.
```

### 2. Configure the project

Open the generated `config.yaml` file and edit it for your experiment.

At this stage, define the body parts or points of interest that you want to track. You
can also adjust project settings such as the number of frames to extract, visualization
settings, and training options.

```{important}
Do not include spaces in body-part names.
```

### 3. Extract frames

```python
deeplabcut.extract_frames(config_path)
```

### 4. Label frames

```python
deeplabcut.label_frames(config_path)
```

### 5. Check labels

```python
deeplabcut.check_labels(config_path)
```

```{tip}
This step is optional, but strongly recommended. Use the generated labeled images to
visually confirm that the annotations were saved correctly before training.
```

### 6. Create the training dataset

```python
deeplabcut.create_training_dataset(config_path)
```

### 7. Train the network

```python
deeplabcut.train_network(config_path)
```

### 8. Evaluate the trained network

```python
deeplabcut.evaluate_network(config_path)
```

Inspect the evaluation results before moving on to video analysis. If pose-estimation
quality is not sufficient, improve the labels, add more training data, or train for
longer before continuing.

### 9. Analyze videos

```python
deeplabcut.analyze_videos(
    config_path,
    video_paths,
)
```

### 10. Filter predictions

```python
deeplabcut.filterpredictions(
    config_path,
    video_paths,
)
```

```{note}
Filtering is optional. It can help smooth pose predictions before plotting
trajectories or creating labeled videos.
```

### 11. Plot trajectories

```python
deeplabcut.plot_trajectories(
    config_path,
    video_paths,
    filtered=True,
)
```

### 12. Create labeled videos

```python
deeplabcut.create_labeled_video(
    config_path,
    video_paths,
    filtered=True,
)
```

This creates videos with the predicted keypoints overlaid, which is useful for a quick
visual inspection of tracking quality.
