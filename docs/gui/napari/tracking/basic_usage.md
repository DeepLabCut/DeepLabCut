---
deeplabcut:
  last_metadata_updated: '2026-05-08'
  last_verified: '2026-05-08'
  verified_for: 3.0.0rc14
  ignore: false
  last_content_updated: '2026-05-08'
---

# Automated annotation with point tracking

```{seealso}
For basic usage of the annotation plugin, see {ref}`file:napari-dlc-basic-usage` for the recommended workflow.
```

```{note}
The plugin relies on third-party open-source tracking models.
Please see {ref}`sec:napari-tracking-models-attribution` at the end of this page for information about the tracking models used in the plugin and their citation information.
```

## Overview

The **Tracking Controls** widget is designed to help automate DeepLabCut annotation workflows:

1. Manually annotate a small set of keypoints on a *reference frame*.
1. Use a point tracking model to propagate those keypoints forward and/or backward in time.
1. Inspect, refine, delete, and merge tracked results before exporting them back to DeepLabCut.

> **Tracking is intended to accelerate annotation, and cannot replace manual review.**

## Requirements

```{tip}
We recommend **having a GPU available for tracking**, as it can be computationally intensive and slow on CPU.
Expect longer processing times on CPU, especially for longer videos or larger tracking ranges.
```

### In napari

```{important}
Before using tracking, you must:

- Load a **video** as an `Image` layer with time as the first dimension.
  - For DLC-integrated workflows, the easiest starting point is often to drag-and-drop one of the `labeled-data` folders from your DLC project.
  - Tracking is most useful on temporally continuous image sequences or videos.
- Ensure you have a **Points** layer containing DeepLabCut-style keypoints.
  - If annotating from scratch, drag-and-drop the `config.yaml` file from your DLC project to create a new Points layer with the correct metadata.
  - If loading a folder which already contains a `CollectedData_*.h5` file, the plugin will automatically create a Points layer with the existing annotations.
- Annotate at least one frame with valid keypoints.

See the workflow guides below for more details on how to prepare your data and annotations for tracking.
```

### In your Python environment

**Skip this if you have already installed PyTorch or DeepLabCut**

If you do not have PyTorch installed, or if you are using the plugin without the DeepLabCut package installed, install with:

```bash
pip install napari-deeplabcut[tracking]
```

## User interface

<!-- Do we want an image that highlights which parts is which section below? -->

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

| Control         | Description                                             |
| --------------- | ------------------------------------------------------- |
| **Tracker**     | Selects the tracking backend from `AVAILABLE_TRACKERS`. |
| **Info button** | Hover to see tracker-specific details.                  |

<!-- More of a placeholder for future models, can be commented if preferred -->

```{note}
Available models may depend on your installation and optional dependencies.
```

### 2. Layer selection

| Control       | Description                                              |
| ------------- | -------------------------------------------------------- |
| **Keypoints** | Points layer containing manually annotated DLC keypoints |
| **Video**     | Image layer containing the video to track                |

The widget automatically updates based on layer changes.

### 3. Reference frame selection

- The **Current** spinbox always reflects the viewer's current time index.
- This frame is used as the **query frame** for tracking.
  - The model generates tracking predictions from the keypoints present on this frame and uses them as seeds to track forward and/or backward in time.

```{note}
Only keypoints present on the selected reference frame are used to initialize tracking.
```

### 4. Frame range controls

Tracking range can be specified **relative** or **absolute** to the reference frame.

#### Backward (left)

- Slider: relative negative offset
- `<< Abs`: absolute frame index
- `<< Rel`: relative frame offset

#### Forward (right)

- Slider: relative positive offset
- `Abs >>`: absolute frame index
- `Rel >>`: relative frame offset

Changing the current frame updates the valid forward/backward range automatically.

### 5. Tracking actions

| Button | Action                        |
| ------ | ----------------------------- |
| ◀      | Track backward                |
| ◀◀     | Track backward to first frame |
| ▶      | Track forward                 |
| ▶▶     | Track forward to last frame   |
| ⟳      | Track both directions         |
| ■      | Stop tracking                 |

```{note}
Tracking runs in a background worker thread. You can continue navigating the viewer and editing layers while it runs; results will appear as a new layer once tracking is complete.
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
**Being able to tell which results originate from which layer is very important for effectively using the plugin.**
- Layers can be toggled (visible/invisible) with `V` by default or by clicking the eye icon next to the layer name in the layer list.
- Grid mode (toggled with `Ctrl+G` by default) can also help visually separate different layers and their results.
```

Each tracking run creates a **new Points layer**:

- Named automatically (`[Tracking v<XX>] Ref. layer name - t<T> - Tracker name`)
  - `XX` refers to the iteration number (if multiple tracking runs are performed from the same reference layer and model)
  - `T` refers to the reference frame index used to generate the tracking result
- Visually distinct from manual annotations:
  - Cross symbol
  - Slight transparency
  - Green border

```{note}
The original annotation layer is never modified by tracking.
To incorporate tracking results into your annotation data, use the merge workflow described below.
```

```{important}
If you run into accessibility issues with the default visualization style, please [open an issue](https://github.com/DeepLabCut/napari-deeplabcut/issues). We would be happy to expand settings and provide more customization options if requested.
```

## Refinement and saving tools

```{danger}
There is **currently no undo option**. Any **deletion or merging action you perform is irreversible**, so we recommend keeping track of your layers and using visibility toggles to compare before and after merge results.
```

### Deleting tracked points in future frames

**Tracking results are often satisfactory for a certain number of frames, then start to drift or produce errors.**
Because of this sometimes unavoidable drift, we provide a way to delete future tracked points while keeping the current frame intact.

1. Select a tracking-result Points layer.
   - This action is always disabled for the original annotation layer.
1. Select one or more points on the **current frame**.
1. Click **Delete selected points in future frames**.

Only *exact identity matches* in future frames are removed.

```{important}
Points on the current frame are preserved so you can correct them and re-run tracking.
```

### Merging tracked points

The **Merge tracked points** workflow allows you to:

- Combine multiple tracking passes
- Decide how to handle overlaps or conflicts
- Produce a clean final annotation layer

This is especially useful when tracking was run from multiple reference frames.
There are several merge options available to help you achieve the desired result:

- **Fill missing only**: Existing keypoints are always preserved. Missing keypoints in frames are filled with tracked results.
  - Intended for merging final tracking results into the original annotation layer.
- **Overwrite existing target points**: Tracked keypoints overwrite existing ones in the target layer.
  - Intended for replacing poor tracking results with a new, updated tracking pass.

```{important}
Tracking-result layers are intermediate working layers.
To save results back into the DeepLabCut project, first merge tracked points into a standard DLC annotation layer, then save that final annotation layer.
```

## Workflow example

### Loading and annotating from scratch

1. Create a DeepLabCut project and add the videos to label.
1. Extract frames from the videos.
   - Currently implemented trackers prefer continuous video frames. We recommend avoiding large gaps in frame indices, which can make tracking more difficult.
1. Go to the `labeled-data` folder, then drag-and-drop a folder with extracted frames into napari.
   - This creates an Image layer with the frames.
1. Drag-and-drop the `config.yaml` file from your DLC project into napari.
   - This creates an empty Points layer with the correct DLC metadata, ready for annotation.
1. Annotate keypoints on a reference frame.

> See {ref}`sec:tracking-workflow-guides`.

### Loading and annotating from existing DLC annotations

1. Go to the `labeled-data` folder, then drag-and-drop a folder with extracted frames into napari.
   - This creates an Image layer with the frames.
   - Existing annotations from the `CollectedData_*.h5` file are loaded as a Points layer.
1. Inspect existing annotations, select a reference frame, and refine keypoints if needed.

> See {ref}`sec:tracking-workflow-guides`.

(sec:tracking-workflow-guides)=

### Tracking

1. Open the Tracking Controls widget (`Plugins -> napari-deeplabcut -> Tracking controls`).
1. Go to the desired reference frame, with annotated keypoints visible.
1. Select the forward/backward tracking range using the sliders, or track to the beginning/end of the video using the seek buttons.
1. Inspect the tracking results.
   - You can use **Show trajectories** in the Keypoint Controls dock widget to visualize the trajectories of tracked points across frames, which can help identify where tracking starts to drift.
   - The plot is filtered by selected keypoints, so you can select a subset of points to inspect their trajectories more closely.
1. If there are problematic points:
   1. On the frame where tracking starts to drift, select the problematic point(s) and click **Delete selected points in future frames** to remove incorrect tracking results while preserving the tracked point(s) on the current frame.
   1. Refine the keypoint(s) on the current frame to correct their position.
   1. Re-run tracking from that frame to propagate the correction forward or backward in time.
1. Merge the new tracking result back into the previous tracking layer when appropriate (for example, using **Overwrite existing target points**).
1. Repeat until satisfied with the tracking result, then merge into the original annotation layer using **Fill missing only** to preserve your original annotations and only add tracked keypoints in frames where you do not yet have manual annotations.
1. **Save the final DLC annotation layer** (usually the original annotation layer after merging).
   - Tracking-result layers are intermediate working layers and are not written back directly as DLC project annotations.
   - Saving the final merged annotation layer is the step that writes back to the DLC project folder and updates the `CollectedData_*.h5` workflow.

```{note}
The **Show trails** feature is currently not available for tracking-result layers. Please [open an issue](https://github.com/DeepLabCut/napari-deeplabcut/issues) if this is something you would like to see in the future.
```

## Troubleshooting

### No keypoints found on reference frame

Ensure:

- You are on the intended frame.
- The correct Points layer is selected.
- Points exist exactly on that frame index.

### Tracking buttons do nothing

Check that:

- A video layer is selected.
- A keypoint layer is selected.
- Tracking is not already running.

(sec:napari-tracking-models-attribution)=

## Models information and citation info

### CoTracker3

> CoTracker is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of OpticalFlow.

- [Link to GitHub repository](https://github.com/facebookresearch/co-tracker)
- [Citation information](https://github.com/facebookresearch/co-tracker#citing-cotracker)

```{admonition} Empirical observations
---
class: tip
---
This information is based on our own testing and experience with the model.
Please share any feedback or insights you have with us!

- **Strengths:** fast on GPU, can output 10-100 frames of satisfactory tracking results, depending on difficulty.
- **Limitations:** strong preference for continuous video frames; struggles with large gaps in frame indices (for example, automated DLC frame extraction via clustering, or uniform extraction with a large step size).
```

## Limitations and future directions

### Important considerations

- As correcting labels can be time-consuming, annotating by hand may sometimes be faster than running tracking and heavily correcting its results.
  - The benefits are mostly for long, continuous videos with many frames to annotate, where tracking can save time by propagating annotations across many frames at once.
  - In very high-variability or very challenging videos, annotating by hand may still be more efficient than running tracking and correcting its results, especially if you only have a few frames to annotate.
- Manual curation is still essential for good tracking results, and the tracking models do not fully replace the need for manual annotation.
- In practice, a mix of hand-labeled hard frames and tracked easy frames should often works best.
- Be mindful of training set imbalance: if you flood your training set with easy frames that are well tracked, and only have a few hand-picked frames with rare or difficult poses, your model may not learn to generalize well to those challenging poses.

#### Future features

- We currently only provide CoTracker3 as a model. It is, however, relatively easy to add new models to the plugin via the registry; feel free to ask if you would like to contribute a model or see a specific model added.
- Generic napari saves or exports of tracking-result layers are not part of the recommended DeepLabCut workflow. Tracking-result layers are intermediate working layers; to preserve results in a DLC project-compatible way, merge them into a standard annotation layer and save that layer.
- If there is demand, we may add support for saving and loading tracking layers as separate files in the DLC project folder.
- If you have ideas for specific refinement tools, shortcuts, or other features that would be useful to add to the plugin, please share them with us.

## Getting help and providing feedback

- [GitHub issues](https://github.com/DeepLabCut/napari-deeplabcut/issues): for bug reports, feature requests, or general questions. We welcome your feedback and contributions.
- [Discussion forum](https://forum.image.sc/tag/deeplabcut): for general discussion, questions, and sharing your work with the community. We also provide troubleshooting help and guidance here, but may open an issue for actual bugs or feature requests and request more information there.
