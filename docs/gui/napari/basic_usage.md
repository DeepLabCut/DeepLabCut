---
deeplabcut:
  last_content_updated: '2026-04-09'
  last_metadata_updated: '2026-04-09'
  ignore: false
  last_verified: '2026-04-09'
  verified_for: 3.0.0rc14
---

(file:napari-dlc-basic-usage)=

# Basic usage

`napari-deeplabcut` is a napari plugin for keypoint annotation and label refinement. It can be used either as part of the DeepLabCut GUI or as a standalone annotation tool.

```{tip}
**New:** check out the new {ref}`semi-automated annotation workflow <file:napari-dlc-tracking-basic-usage>` to speed up your work in the plugin.
```

## Labeling stages

- {ref}`Extracting frames from a video <sec:napari-dlc-video-workflow>`
- {ref}`Labeling from scratch <sec:napari-dlc-labeling-from-scratch>`
- {ref}`Resuming labeling <sec:napari-dlc-resuming-labeling>`
- {ref}`Refining machine labels <sec:napari-dlc-refining-machine-labels>`

Also see {ref}`file:napari-dlc-troubleshooting` for troubleshooting tips.

## Before you start

If you installed `DeepLabCut[gui]`, `napari-deeplabcut` is already included.

You may use `napari-deeplabcut` in two ways, described below.

### Usage in the DeepLabCut GUI

When labeling frames, checking labels, or manually extracting frames from videos, the napari plugin will open automatically.

### Usage as a standalone plugin

You can also install it as a standalone plugin:

```bash
pip install napari-deeplabcut
```

Start napari from a terminal:

```bash
napari
```

Then open the plugin from:

**Plugins -> napari-deeplabcut: Keypoint controls**

## Supported inputs

The plugin reader can open the following inputs:

- DeepLabCut `config.yaml`
- Image folders (supports `.png`, `.jpg`, extracted frames from DLC, as well as folders of mixed formats)
- Videos (`.mp4`, `.avi`, `.mov`)
- `.h5` annotation files

You can load files either by:

- using the main DeepLabCut GUI "Label Frames" button

*OR*

- dragging and dropping them onto the napari viewer, or
- using the **File** menu

```{tip}
If you drag and drop a compatible labeled-data folder, the widget opens automatically.
```

## Using napari

```{important}
To familiarize yourself with napari, we recommend checking out the [official napari documentation and tutorials](https://napari.org/stable/usage.html).
```

(sec:napari-dlc-basic-workflow)=

## Labeling

Once the **Points** layer is active, you can place and edit keypoints in the viewer.

Select the correct **Points** layer in the layer list. Use the **+** tool to add keypoints, the selection tool to edit existing keypoints, and the pan/zoom tool to navigate the viewer.

```{caution}
There is no built-in backup or undo functionality in current plugin versions.
Do not iterate on the only copy of your project files.
We recommend backing up your project files as you would any other experimental data.
```

### Widget options

- **Keypoint selection**: The dropdown shows which bodypart will be added when placing a new keypoint in the Points layer. It can be changed manually, and will be updated according to the active labeling mode (see below).
- **View shortcuts**: opens a reference of napari-deeplabcut shortcuts and their context (i.e. when they are active).
- **Show tutorial**: opens the napari-DLC tutorial panels.

#### Labeling mode

- **Sequential**: when a keypoint is placed, the next keypoint in the config list is automatically selected. This is useful for labeling frames in order. Adding an already present keypoint in the frame does nothing.
- **Quick**: As sequential, but adding an already present keypoint in the frame will move it to the new location.
- **Loop**: The currently selected bodypart is retained and the viewer advances to the next frame. This is useful for labeling a specific body part across many frames in a row. If the end of the video is reached, the viewer will loop back to the beginning.

The dock widget also provides additional controls, including:

- **Warn on overwrite**: enable or disable overwrite confirmation
- **Show trails**: display keypoint trails over time
- **Show trajectories**: open a trajectory plot in a separate dock widget
- **Show color scheme**: display the active color mapping
- **Video tools**: extract frames and store crop coordinates when a video is loaded

### Useful shortcuts

- napari native:
  - `2` / `3`: switch between labeling and selection mode
  - `4`: pan and zoom mode
  - `Ctrl+R`: reset the viewer to the default zoom and position

```{tip}
Use the **View shortcuts** button in the dock widget for a quick reference of napari-deeplabcut shortcuts and when they are active.
```

### More quality-of-life features

See the {ref}`Advanced features <file:napari-dlc-advanced-features>` for useful features such as copy-pasting annotations, quick bodypart selection, and more.

## Saving annotations

To save annotations, select the **Points** layer you want to save and use:

**File -> Save Selected Layer(s)...**

or press:

```text
Ctrl+S
```

```{note}
If you open a folder that is outside a DeepLabCut project and then save a Points layer, you will be prompted to provide the corresponding `config.yaml`. After saving, you can move the labeled-data folder into your project for downstream DeepLabCut workflows.
```

Annotations are saved into the dataset folder as:

```text
CollectedData_<ScorerName>.h5
```

These are the ground truth annotations that DeepLabCut will use for training and evaluation.
A companion CSV file is also written:

```text
CollectedData_<ScorerName>.csv
```

```{important}
DeepLabCut uses the `.h5` file as the authoritative annotation file. CSVs and machine labels will not be taken into account for training.
```

### Save behavior and notes

- Make sure the correct **Points** layer is selected before saving.
- If several Points layers are selected at the same time, the plugin will not save them in order to avoid ambiguity.
- If saving would overwrite existing annotations, the plugin will ask for confirmation.
  - Removing a keypoint that exists in the file counts as a **deletion**, and is listed separately in that confirmation.
  - This confirmation can be disabled by unchecking **Warn on overwrite** in the dock widget. **Deletions are then no longer reported either.**
- **Removing a bodypart from `config.yaml` and then loading that config drops its annotations.** It is removed from the current layer, and its existing annotations are dropped from `CollectedData...` on the next save. This will be adjusted in the future, please use this behavior with caution. We recommend keeping legacy bodyparts and filtering on subsequent steps; DeepLabCut will **NOT** use undeclared bodyparts for training data generation.

```{note}
Several plugin functions expect `config.yaml` to be located two folders above the saved `CollectedData...` file, matching the standard DeepLabCut project structure.<br>
Keeping data inside the project directory is recommended for best compatibility. Fallbacks asking for the config file location are provided when this structure is not respected, but some features may be disabled or limited in that case.
```

## Labeling workflows

(sec:napari-dlc-labeling-from-scratch)=

### Labeling from scratch

Use this when the image folder does **not** yet contain a `CollectedData_<ScorerName>.h5` file.

**From the DeepLabCut GUI**

Click the **Label Frames** button to start labeling from scratch.

**In napari directly**

1. Open a folder of extracted images
1. Open the corresponding DeepLabCut `config.yaml`, this creates an empty **Points** layer named `CollectedData_<ScorerName>`
1. Select the created **Points** layer
1. Label keypoints
1. Save with `Ctrl+S`

After saving, the folder will contain:

```text
CollectedData_<ScorerName>.h5
CollectedData_<ScorerName>.csv
```

(sec:napari-dlc-resuming-labeling)=

### Resuming labeling

Use this when the folder already contains a `CollectedData_<ScorerName>.h5` file.

1. Open or drag and drop the folder in napari.
1. Select the loaded **Points** layer.
1. Continue labeling and save with `Ctrl+S`.

Existing annotations and keypoint metadata will be loaded automatically from the H5 file.
In this case, loading `config.yaml` manually is usually **not needed** unless:

- The project's bodyparts have changed
- The color scheme in `config.yaml` has changed since you last opened the folder

See {ref}`sec:napari-dlc-update-keypoints-from-config`

(sec:napari-dlc-refining-machine-labels)=

### Refining machine labels

Use this when the folder contains a machine predictions file such as:

```text
machinelabels-iter<...>.h5
```

Open the folder in napari.

```{tip}
When machine labels are present, an extra layer is added and you will see keypoints from ALL current layers.
GT labels are shown with a **disk marker**, whereas machine labels use a **cross marker**.
**Try hiding each point layer individually to better distinguish between GT and machine labels.**
Before editing, **make sure to select the correct layer to edit** in the viewer list (e.g. the `machinelabels...` layer if you want to refine machine predictions).
```

If both a `CollectedData...` file and a `machinelabels...` file are present:

1. Select the `machinelabels` layer
1. Optionally press `E` to show edge coloring (red edges indicate confidence below the threshold defined in `config.yaml`)
1. Hide other Points layers to make editing easier if needed
1. Edit keypoints in the `machinelabels` layer to refine machine predictions
1. Save the selected `machinelabels` layer

The refined annotations will be merged into `CollectedData...`.

If only `machinelabels...` is present, saving refinements will still create a new `CollectedData...` target.

```{important}
Saving a `machinelabels...` layer does **not** overwrite the machine labels file itself.
Refinements are written into the appropriate `CollectedData...` file.<br>
Make sure overwrite confirmation is enabled if you want to avoid accidentally overwriting existing `CollectedData...` annotations.
```

(sec:napari-dlc-video-workflow)=

## Video workflow (crop and frame extraction)

Videos can also be opened directly in napari.

```{tip}
This works best by using the main DLC GUI and following steps there for manual frame extraction, which will automatically open the video in napari.
The workflow is otherwise the same when opening a video directly in napari.
```

When a video is loaded, the plugin provides a small video action panel that can be used to:

- Extract the current frame into the dataset
- Optionally export existing machine labels for that frame (load the corresponding h5 file first)
- Define and save crop coordinates to the DeepLabCut `config.yaml`

Keypoints from video-based workflows can be edited and saved in the same way as image-folder workflows.

## Working with multiple folders

We do not currently support working on **more than one dataset folder at a time**.

When using the main DeepLabCut GUI, the recommended way to switch folders is to close napari and open the next folder from the GUI.

If a new folder is opened while another one is already open, the plugin will prevent new frames from being loaded, attempt to load annotations using the current folder context, and show a warning.

After finishing one folder, simply:

1. Save the relevant **Points** layer
1. Remove the current layers from the viewer using the layer list (left panel)
1. Open the next folder (e.g. by dragging and dropping it onto the viewer)

This helps keep saving behavior unambiguous.

(sec:napari-dlc-update-keypoints-from-config)=

## Updating the keypoints definitions and color scheme while labeling

To update in place the current keypoint list, you can drag-and-drop a new `config.yaml` with the updated keypoints into the viewer.

If the config contains keypoints the current layer does not have, you will be prompted to choose how to handle the new keypoints:

| Choice               | Effect                                                      |
| -------------------- | ----------------------------------------------------------- |
| **Apply to current** | Adds the updated keypoints to the layer you were working on |
| **Keep both**        | Leaves your layer untouched and keeps the config layer      |
| **Cancel**           | Discards the temporary layer                                |

```{note}
If the config adds no new keypoints, there is no prompt and it is applied directly. This is the usual case when only the color scheme changed: the colors defined in the new config are applied to the current Points layer.
```

## Demo

A short demo video is available here:

[Link to video](https://youtu.be/hsA9IB5r73E)

```{note}
The interface shown in this video may differ from the current version, but the general annotation workflow remains applicable.
```
