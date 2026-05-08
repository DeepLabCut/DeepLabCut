---
deeplabcut:
  last_metadata_updated: '2026-05-08'
  last_verified: '2026-05-08'
  verified_for: 3.0.0rc14
  ignore: false
---

# Automated annotation with point tracking

```{seealso}
For basic usage of the annotation plugin, see {ref}`file:napari-dlc-basic-usage` for the recommended workflow.
```

> We use third-party, open-source models for point tracking, and we thank the original authors and developers.
> Please see {ref}`sec:napari-tracking-models-attribution` at the end of this page for information about the tracking models used in the plugin and their citation information.

## Overview

The **Tracking Controls** widget is designed to help automate DeepLabCut annotation workflows:

1. Manually annotate a small set of keypoints on a *reference frame*.
1. Use a point tracking model to propagate those keypoints forward and/or backward in time.
1. Inspect, refine, delete, and merge tracked results before exporting them back to DeepLabCut.

## Requirements

```{important}
Before using tracking, you must:

- Load a **video** as an `Image` layer with time as the first dimension.
  - The easiest is to drag-and-drop one of the `labeled-data` folders from your DLC project
- Ensure you have a **Points** layer containing DeepLabCut-style keypoints.
    - If annotating from scratch, drag-and-drop the `config.yaml` file from your DLC project to create a new Points layer with the correct metadata.
    - If loading an folder which already contains a `CollectedData_*.h5` file, the plugin will automatically create a Points layer with the existing annotations.
- Annotate at least one frame with valid keypoints.
```

If you do not have PyTorch installed or if you are using the plugin without the DeepLabCut package installed, install with:

```bash
pip install napari-deeplabcut[tracking]
```

## User interface

```{figure} ../../../images/napari/tracking/controls.png
---
name: tracking-controls
caption: Tracking Controls widget with annotated keypoints and tracking results.
---
Tracking Controls widget with annotated keypoints and tracking results.
```

### Showing the widget

Use:

> Plugins -> napari-deeplabcut -> Tracking controls

### 1. Model selection

- **Tracker**: Selects the tracking backend from `AVAILABLE_TRACKERS`.
- **Info button**: Hover to see tracker-specific details.

### 2. Layer selection

| Control       | Description                                              |
| ------------- | -------------------------------------------------------- |
| **Keypoints** | Points layer containing manually annotated DLC keypoints |
| **Video**     | Image layer containing the video to track                |

The widget automatically updates when layers are added, removed, or reordered.

### 3. Reference frame selection

- The **Current** spinbox always reflects the viewer's current time index.
- This frame is used as the **query frame** for tracking.
  - This means that the model will generate tracking predictions based on the keypoints present on this frame, and use them as seeds to track forward and/or backward in time.

```{note}
Only keypoints *visible on the reference frame* are used to initialize tracking.
```

### 4. Frame Range Controls

Tracking range can be specified **relative** or **absolute** to the reference frame.

#### Backward (left)

- Slider: relative negative offset
- `<< Abs`: absolute frame index
- `<< Rel`: relative frame offset

#### Forward (right)

- Slider: relative positive offset
- `Abs >>`: absolute frame index
- `Rel >>`: relative frame offset

### 5. Tracking Actions

| Button | Action                        |
| ------ | ----------------------------- |
| ◀      | Track backward                |
| ◀◀     | Track backward to first frame |
| ▶      | Track forward                 |
| ▶▶     | Track forward to last frame   |
| ⟳      | Track both directions         |
| ■      | Stop tracking                 |

```{note}
Tracking runs in a background worker thread. You may edit layers while it is running, and results will appear as a new layer once tracking is complete.
```

## Keyboard shortcuts

Most tracking functions have keyboard shortcuts for easier usage.

```{tip}
You can see shortcuts and their status using:
> Help -> Show napari-dlc shortcuts
This is only available if the Keypoint controls widget has been opened.
```

## Tracking results

```{tip}
**Hiding layers, and being able to distinguish which results originate from which layer, is a very important notion for effectively using the plugin.**
Layers can be toggled (visible/invisible) with `V` by default or by clicking the eye icon next to the layer name in the layer list.
Grid mode (toggled with `Ctrl+G` by default) can also help visually separate different layers and their results.
```

Each tracking run creates a **new Points layer**:

- Named automatically (`[Tracking vXX] Ref. layer name - tT - Tracker name`)
  - `XX` refers to the iteration number (if multiple tracking runs are performed from the same reference frame)
  - `T` refers to the reference frame index used to generate the tracking result
- Visually different from manual annotations:
  - Cross symbol
  - Slight transparency
  - Green border

```{note}
The original annotation layer is never modified by tracking.
This has to be done manually by merging, see below.
```

```{important}
If you run into accessibility issues with the default visualization style, please [open an issue](https://github.com/DeepLabCut/napari-deeplabcut/issues), we would be happy to expand settings and provide more customization options if requested.
```

## Refinement & saving tools

### Deleting tracked Points in future frames

**Oftentimes, tracking results will be satisfactory for a certain number of frames, then start to drift or produce errors.**
This is inherent to the tools, and as such we provide a simple way to delete incorrect tracking results in future frames while preserving the original annotations on the reference frame.

1. Select a tracking-result Points layer.

- This is always disabled for the original annotation layer.

2. Select one or more points on the **current frame**.
1. Click **Delete selected points in future frames**.

Only *exact identity matches* in future frames are removed.

```{important}
Points on the current frame are preserved so you can correct them and re-run tracking.
```

### Merge (save) tracked Points

The **Merge tracked points** workflow allows you to:

- Combine multiple tracking passes
- Resolve overlaps or conflicts
- Produce a clean final annotation layer

This is especially useful when tracking was run from multiple reference frames.
There are several merge options available to help you achieve the desired result:

- **Fill missing only**: Existing keypoints are always preserved. Missing keypoints in frames are filled with tracked results.
  - Intended for merging final tracking results into the original annotation layer.
- **Overwrite existing target points**: Tracked keypoints always overwrite existing ones, regardless of presence.
  - Intended for replacing poor tracking results with a new, updated tracking pass.

```{danger}
There is **currently no undo option**. Any **deletion or merging action you perform is irreversible**, so we recommend keeping track of your layers and using visibility toggles to compare before/after merge results.
```

## Workflow example

### Loading & annotating from scratch

1. Create a DeepLabCut project and add the videos to label.
1. Extract frames in the videos.

- Currently implemented trackers prefer continuous video frames. We recommend avoiding large gaps in frame indices, which can make tracking more difficult.

3. Go to the `labeled-data` folder, drag-and-drop a folder with extracted frames into napari.

- This will create an Image layer with the frames

4. Drag-and-drop the `config.yaml` file from your DLC project into napari.

- This will create an empty Points layer with the correct DLC metadata, ready for annotation.

5. Annotate keypoints on a reference frame.

See {ref}`sec:tracking-workflow-guides`.

### Loading and annotating from existing DLC annotations

1. Go to the `labeled-data` folder, drag-and-drop a folder with extracted frames into napari.

- This will create an Image layer with the frames.
- The existing annotation from the `CollectedData_*.h5` file will be loaded as a Points layer.

2. Inspect existing annotations, select a reference frame, and refine keypoints if needed.

See {ref}`sec:tracking-workflow-guides`.

(sec:tracking-workflow-guides)=

### Tracking

1. Open the Tracking Controls widget (`Plugins -> napari-deeplabcut -> Tracking controls`).
1. Go to the desired reference frame (with annotated keypoints visible).
1. Select the forward/backward tracking range using the sliders, OR track to beginning/end of the video using the fast-forward buttons.
1. Inspect the tracking results
   - You can use "Show trajectories" in the Keypoint Controls layer to visualize the trajectories of tracked points across frames, which can help identify where tracking starts to drift.
   - The plot is filtered by selected keypoints, so you can select a subset of points to inspect their trajectories more closely.
1. On the frame where tracking starts to drift:
1. Select the problematic point(s) and click "Delete selected points in future frames" to remove incorrect tracking results while preserving the tracked point(s) on the current frame.
1. Refine the keypoint(s) on the current frame to correct their position.
1. Re-run tracking from that frame to propagate the correction forward/backward in time.
1. Merge the new tracking result back into the previous tracking layer (e.g. using "Overwrite existing target points")
1. Repeat until satisfied with the tracking result, then merge into the original annotation layer using "Fill missing only" to preserve your original annotations and only add tracked keypoints in frames where you don't have manual annotations.
1. **Remember to save the original Points annotation layer.** This is the only step that writes back to the DLC project folder directly and integrates with the `h5` file.

```{note}
The "Show trails" feature is currently not implemented for tracking result layers, please [open an issue](https://github.com/DeepLabCut/napari-deeplabcut/issues) if this is something you would like to see in the future!
```

## Troubleshooting

### No keypoints found on reference frame

Ensure:

- You are on the intended frame
- The correct Points layer is selected
- Points exist exactly on that frame index

### Tracking buttons do nothing

Check that:

- A video layer is selected
- A keypoint layer is selected
- Tracking is not already running

(sec:napari-tracking-models-attribution)=

## Models information and citation info

### CoTracker3

CoTracker is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.

[Link to GitHub repository](https://github.com/facebookresearch/co-tracker)
[Citation information](https://github.com/facebookresearch/co-tracker#citing-cotracker)

```{admonition} Emprical observations
---
class: tip
---
This information is based on our own testing and experience with the model.
Please share any feedback or insights you have with us!

- **Strengths:** fast on GPU, can output 10-100 frames of satisfactory tracking results, depending on difficulty
- **Limitations:** strong preference for continuous video frames, struggles with large gaps in frame indices (e.g. automated DLC frame extraction via clustering, or uniform extraction with large step size)
```

## Limitations and future directions

- We currently only provide CoTracker3 as a model. It is however easy to add new models to the plugin via the registry; feel free to ask if you would like to contribute a model or see a specific model added!
- Saving tracking layers as CSV is supported, but they will not be loaded correctly as tracking results in the plugin. We currently recommend using the "Merge tracked points" workflow to save results back into the original annotation layer, which is then saved to the DLC project folder and can be loaded in future sessions.
- If there is demand, we may add support for saving/loading tracking layers as separate files in the DLC project folder.
- Manual curation is still essential for good tracking results, and the tracking models do not fully replace the need for manual annotation.
- Ideally mixing manual annotations from challenging/distincts frames with tracking results from easier frames would yield the best results.
- Be mindful of training set imbalance: if you flood your training set with easy frames that are well tracked, and only have few hand-picked frames with rare or difficult poses, your model may not learn to generalize well to those challenging poses.

## Getting help and providing feedback

- [GitHub issues](https://github.com/DeepLabCut/napari-deeplabcut/issues): for bug reports, feature requests, or general questions. We welcome your feedback and contributions!
- [Discussion forum](https://forum.image.sc/tag/deeplabcut): for general discussion, questions, and sharing your work with the community. We also provide troubleshooting help and guidance here, but may open an issue for actual bugs or feature requests and request more information there.
