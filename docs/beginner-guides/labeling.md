---
deeplabcut:
  last_content_updated: '2025-06-30'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: viable
  recommendation: update
  notes: Updated to link directly to the napari plugin docs. Making the link specific to the workflow section of the napari docs could help.
---

(file:labeling-gui)=

# Labeling GUI

## Selecting frames to label

In DeepLabCut, choosing the right frames for labeling is a key step.

```{important}
Always aim to select the **most diverse data** you can for your model to be trained on. This implies picking a variety of good lighting, bad lighting, partial occlusions, and different poses.
If relevant, label data across several experimental sessions, animals, and conditions.
**Labeling 10 frames from several different videos is typically more effective than labeling 100 frames from a single video.**
```

To help you select "different" frames, DeepLabCut provides two main options:

1. **Automated frame extraction** DeepLabCut can extract frames automatically for you.

   - **Uniform:** Samples at regular time intervals. Does not guarantee diversity, but is simple and fast.
   - **K-means clustering:** This one runs a k-means algorithm and picks images from different clusters. This is typically more robust in extracting a variety of actions and poses. Note, as it is a clustering tool, it will miss rare events, so after running this step, consider using the manual GUI to get some rare frames! You can do both within DLC.

1. **Manual frame extraction** Pick frames yourself using the GUI. This is the most time-consuming, but allows you to have full control over the frames you pick, and can be useful to get rare events that automated tools might miss. You can also use this after running automated frame extraction to get some of those "rare" frames.

### Example workflow

1. Click on **`automatic`** in the frame selection area.
1. Choose **`k-means`** as a good default option for frame extraction, and set the number of frames you want to extract.
1. Hit the **`Extract Frames`** button.

By default, DeepLabCut will grab 20 frames from each of your videos and put them into sub-folders, per video, under **labeled-data** in your project.
With this, you are all set to start labeling!

## Frame labeling workflow

Alright, you've got your extracted frames ready. Now comes the labeling!

### Launching the labeling GUI

- **Click on `Label Frames`:** This takes you straight to where your frames are, sorted in the **labeled-data** folder, each video in its own sub-folder.
- **Open a Folder:** Click on the first unlabeled folder to start, and then click **`Open`**.

### napari-deeplabcut

Please refer to the {ref}`file:napari-dlc-basic-usage` section for a detailed walkthrough of how to use the napari-DLC plugin for labeling your frames.

### Completing the labeling

Work through all the frames in the first folder and save them.

```{tip}
After saving, you can close napari and click **`Label Frames`* again to open the next folder
**OR**
Remove all layers in napari and drag-and-drop the next folder in the same napari session to keep going without needing to close and reopen napari.
```

## Checking labels

After you've labeled all your frames, you may want to review their accuracy before moving on to training your model. This is a crucial step, as the quality of your labels will directly impact the performance of your model.

- **Return to the Main Window:** Once you're done with labeling, head back to DeepLabCut's main window, and click on **`Check Labels`**.
- **Review the Labeled Folders:** The system will have created new folders for each labeled set inside your labeled-data folder. These folders contain your original frames overlaid with the keypoints you've added.

![Checking Labels in DeepLabCut](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779615252-6BNW661XB2ULH85RTAD3/evaluation-example.png?format=500w)

Take the time to go through each folder. Accurate labels are key.
If there are mistakes, the model might learn incorrectly and mislabel your videos later on.
A clean foundation is essential for accurate analysis.

## Next steps

Head on to {ref}`file:training-evaluation-gui` to learn about training and evaluating your neural network with the labeled data you created!
