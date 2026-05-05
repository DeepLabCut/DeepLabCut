---
deeplabcut:
  last_content_updated: '2026-04-09'
  last_metadata_updated: '2026-04-09'
  ignore: false
  last_verified: '2026-04-09'
  verified_for: 3.0.0rc14
---

(file:napari-dlc-advanced-features)=

# napari-DLC - Advanced features

napari-DLC provides several additional features to enhance the annotation experience.

This section covers some of these features in more detail.
For more basic features and workflows, see the {ref}`basic usage section <file:napari-dlc-basic-usage>`.

## Layer status panel

### Current folder

The current folder associated with the active Points layer is displayed at the top of the dock widget.
This is the folder where annotations will be saved when using **File -> Save Selected Layer(s)** (or `Ctrl+S`).

### Labeling progress

When a labeled data folder is loaded, the widget shows a percentage of labeled frames, based on the theoretical maximum number of keypoints (i.e. number of body parts x number of individuals x number of frames) that could be labeled.

```{note}
This can be a useful reference to track labeling progress.<br>
Since visibility cannot be accounted for, it should be considered an estimate of relative labeling progress rather than an absolute measure of completeness. (as not all videos would need 100% labeling, i.e. every body part on every individual in every frame).
```

### Point size slider

The dock widget includes a slider to adjust the size of all keypoints in the viewer; the selected dot size will be saved in `config.yaml` for convenience, meaning DLC will reuse it for future sessions.

## Copy-paste annotations

To copy-paste keypoints from one frame to another:

- Select the keypoints you want to copy using the selection tool (shortcut `3`)
- Press `Ctrl+C` to copy the selected keypoints
- Navigate to the target frame and press `Ctrl+V` to paste the keypoints

## Color scheme display features

The plugin shows a list of body parts and their corresponding colors in the dock widget. You can toggle the visibility of this color scheme using the **Show color scheme** button.

```{tip}
The display only shows keypoints that are currently visible in the viewer.<br>
To show all bodyparts in the color scheme from the config, use the checkbox at the top of the color scheme list.
```

### Quick body part/individual selection

Clicking on a body part in the color scheme will select all keypoints of that body part in the viewer (including across individuals if applicable).

This can be useful for quickly selecting and editing all keypoints of a specific body part.

In individual coloring mode, the color scheme also shows the individuals list, and clicking on an individual will select all keypoints belonging to that individual.

### Jump to body part in viewer

If showing all body parts in the color scheme from the config, clicking on a keypoint in the list that is not currently visible in the viewer will jump to the first instance of that body part in the viewer and select it, if applicable.
This helps quickly find a specific body part in the viewer.

## Trajectory plot

The **Show trajectories** button opens a trajectory plot in a separate dock widget. This plot shows the trajectories of all **selected keypoints** over time, and will color-code them according to the active color scheme (bodyparts or individuals).

To show the trajectory of a specific keypoint, simply select that keypoint in the viewer (using the selection tool or by clicking on the corresponding body part in the color scheme).

Additional controls in the trajectory plot dock widget allow you to zoom and pan the plot, as well as adjust the time window shown.
