---
deeplabcut:
  last_content_updated: '2026-04-09'
  last_metadata_updated: '2026-04-09'
  ignore: false
  last_verified: '2026-04-09'
  verified_for: 3.0.0rc14
---
(file:napari-dlc-basic-usage)=
# napari-DLC - Basic usage

`napari-deeplabcut` is a napari plugin for keypoint annotation and label refinement. It can be used either as part of the DeepLabCut GUI or as a standalone annotation tool.

## Before you start

If you installed `DeepLabCut[gui]`, `napari-deeplabcut` is already included.

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

- dragging and dropping them onto the napari viewer, or
- using the **File** menu

```{tip}
If you drag and drop a compatible labeled-data folder, the widget opens automatically.
```

## Recommended basic labeling workflow

The simplest way to **start labeling** is:

1. Open an image-only folder
2. Open the corresponding `config.yaml` from your DeepLabCut project

**OR**

1. Open a folder inside a DeepLabCut project's `labeled-data` directory with a `CollectedData_<ScorerName>.h5` file already present

This creates:

- an **Image** layer containing the images (or video frames)
- a **Points** layer initialized with the keypoints defined in the project config

You can then start annotating directly in the **Points** layer.
To do so, make sure the correct **Points** layer is selected in the layer list (left panel of the viewer). Click on the **+** icon to start adding keypoints; the selection tool to edit existing keypoints; and the pan/zoom tool to navigate the viewer.

```{note}
If you open a folder that is outside a DeepLabCut project and then save a Points layer, you will be prompted to provide the corresponding `config.yaml`. After saving, you can move the labeled-data folder into your project for downstream DeepLabCut workflows.
```

## Labeling

Once the **Points** layer is active, you can place and edit keypoints in the viewer.

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

## Saving annotations

To save annotations, select the **Points** layer you want to save and use:

**File -> Save Selected Layer(s)...**

or press:

```text
Ctrl+S
```

Annotations are saved into the dataset folder as:

```text
CollectedData_<ScorerName>.h5
```

A companion CSV file is also written:

```text
CollectedData_<ScorerName>.csv
```

```{important}
DeepLabCut uses the `.h5` file as the authoritative annotation file.
```

### Save behavior and notes

- Make sure the correct **Points** layer is selected before saving.
- If several Points layers are selected at the same time, the plugin will not save them in order to avoid ambiguity.
- If saving would overwrite existing annotations, the plugin will ask for confirmation.
- This confirmation can be disabled by unchecking **Warn on overwrite** in the dock widget.

```{note}
Several plugin functions expect `config.yaml` to be located two folders above the saved `CollectedData...` file, matching the standard DeepLabCut project structure.<br>
Keeping data inside the project directory is recommended for best compatibility. Fallbacks asking for the config file location are provided when this structure is not respected, but some features may be disabled or limited in that case.
```

### Useful shortcuts

- napari native:
  - `2` / `3`: switch between labeling and selection mode
  - `4`: pan and zoom mode
  - `Ctrl+R`: reset the viewer to the default zoom and position
- napari-deeplabcut specific:
  - `M`: cycle through annotation modes
  - `E`: toggle edge coloring
  - `F`: toggle between individual and body-part coloring modes
  - `V`: toggle visibility of the selected layer
  - `Backspace`: delete selected point(s)
  - `Ctrl+C` / `Ctrl+V`: copy and paste selected points

```{tip}
Use the **View shortcuts** button in the dock widget for a quick reference of napari-deeplabcut shortcuts and their context (i.e. when they are active).
```

### More quality-of-life features

See the [advanced features section](file:napari-dlc-advanced-features) for useful features such as copy-pasting annotations, quick body part selection, and more.

## Labeling workflows

### Labeling from scratch

Use this when the image folder does **not** yet contain a `CollectedData_<ScorerName>.h5` file.

1. Open a folder of extracted images
2. Open the corresponding DeepLabCut `config.yaml`
3. Select the created **Points** layer
4. Label keypoints
5. Save with `Ctrl+S`

After saving, the folder will contain:

```text
CollectedData_<ScorerName>.h5
CollectedData_<ScorerName>.csv
```

### Resuming labeling

Use this when the folder already contains a `CollectedData_<ScorerName>.h5` file.

- Open (or drag and drop) the folder in napari.

Existing annotations and keypoint metadata will be loaded automatically from the H5 file.
In this case, loading `config.yaml` is usually **not needed** unless :

- The project's body parts have changed or
- You want to refresh the configured color scheme

### Refining machine labels

Use this when the folder contains a machine predictions file such as:

```text
machinelabels-iter<...>.h5
```

Open the folder in napari.

If both a `CollectedData...` file and a `machinelabels...` file are present:

1. Edit the `machinelabels` layer
2. Optionally press `E` to show edge coloring (red edges indicate confidence below the threshold defined in `config.yaml`)
3. Save the selected `machinelabels` layer

The refined annotations will be merged into `CollectedData...`.

If only `machinelabels...` is present, saving refinements will still create a new `CollectedData...` target.

```{important}
Saving a `machinelabels...` layer does **not** overwrite the machine labels file itself. Refinements are written into the appropriate `CollectedData...` file.<br>
Make sure overwrite confirmation is enabled if you want to avoid accidentally overwriting existing `CollectedData...` annotations.
```

## Video workflow (crop and frame extraction)

Videos can also be opened directly in napari.

When a video is loaded, the plugin provides a small video action panel that can be used to:

- Extract the current frame into the dataset
- Optionally export existing machine labels for that frame (load the corresponding h5 file first)
- Define and save crop coordinates to the DeepLabCut `config.yaml`

Keypoints from video-based workflows can be edited and saved in the same way as image-folder workflows.

## Working with multiple folders

We do not currently support working on **more than one dataset folder at a time**.

After finishing one folder, simply:

1. Save the relevant **Points** layer
2. Remove the current layers from the viewer
3. Open the next folder

This helps keep saving behavior unambiguous.



## Demo

A short demo video is available here:

<https://youtu.be/hsA9IB5r73E>
