---
deeplabcut:
  last_metadata_updated: '2026-09-28'
  last_verified: '2026-09-28'
  verified_for: 3.0.2
  ignore: false
  last_content_updated: '2026-09-28'
---

(file:napari-dlc-troubleshooting)=

# Troubleshooting the napari plugin

## Useful tips

When the Keypoint or Tracking controls have been opened at least once, you may use **`Help -> Generate napari-dlc logs`** to collect diagnostic information for troubleshooting.
Use **`Copy to clipboard`** to copy the generated logs for issues reporting on GitHub.

**This is one of the most helpful ways to provide detailed information when reporting issues on GitHub.**
We may also ask that you share some configuration files or additional context to help us diagnose the issue.

## Messages when opening a folder

Please find several of the common messages you may encounter when opening a folder below.

### "does not match the frames now in ..."

**Your annotations are unchanged and still save to their own folder. The layer is locked
for editing.**

A keypoints layer from another dataset folder is still open, or the frames were renamed or
re-extracted. Keypoints are tied to a position in the frame order, not to a filename, so
labelling the layer now would store them against the frames it was loaded with rather than
the ones on screen. The layer is locked until it matches the folder on screen again.

To label the folder you opened:

1. Save the locked layer if it has unsaved changes
1. Clear all layers, including the images
1. Open the folder again
1. If it has no annotations yet, drag in the project's `config.yaml` to get a keypoints
   layer carrying the project's bodyparts

To go back to the previous folder instead, clear all layers and reopen it. The lock lifts
on its own once the layer's frames match the folder on screen.

```{note}
Saving a locked layer still works, and writes to its own folder. Only editing is blocked.
```

### "Annotated frames lost their path"

**Your annotation file is not modified. The layer is locked for editing.**

A labeled frame was renamed or deleted. Keypoints are tied to a position in the frame
order, not to a filename, and said position now belongs to a different image.
Keypoints after the missing frame may display one frame off.

Unlike the message above, reopening the folder does not lift this lock: the frame is still
missing, so the plugin refuses again. The annotation file and the folder have to agree
first.

Restore the frame in the folder, or remove its row:

1. Delete the row for the missing frame in `CollectedData_<ScorerName>.csv`
1. Run `deeplabcut.convertcsv2h5("/path/to/config.yaml")`
1. Reopen the folder

```{note}
`convertcsv2h5` prompts per folder, and only visits folders listed in `video_sets`.
```

### "These annotations are already open as ..."

**Keep working in the layer you already had.**

The folder was opened twice. Both layers would save to the same file, so
the second copy is closed automatically.

(sec:napari-dlc-repair-wrong-folder)=

## Annotations written into the wrong folder

**This affects files written before v0.4.0 of the plugin. Current versions refuse the write that causes it.**

A keypoints layer left open while a different extracted frames folder was opened could follow that folder and, on the next save, write annotations to the wrong folder. The result is a `CollectedData_<ScorerName>` file holding rows for frames belonging to another folder.

No message is shown for this, since it happened in an earlier session. To check a file, open the `.csv` next to it: the first column lists one frame per row, and every row should name the folder the file sits in.

To repair it:

1. Copy any row naming another folder into that folder's `CollectedData_<ScorerName>.csv`, if the annotations are not already there
1. Delete those rows from the file you are repairing
1. Run `deeplabcut.convertcsv2h5("/path/to/config.yaml")`
1. Reopen the folder

```{warning}
Deleting the rows discards those annotations unless they exist in the folder they belong to. Check before deleting.
```
