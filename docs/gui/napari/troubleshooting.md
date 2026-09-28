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

**Your annotations are unchanged and still save to their own folder.**

A keypoints layer from another dataset folder is still open, or the frames were renamed or
re-extracted.

- Switching video or project: save, remove the layers, open the next folder
- Same folder, re-extracted frames: informational, nothing to do

### "Annotated frames lost their path"

**Your annotation file is not modified. Do not move or replace the keypoints.**

A labeled frame was renamed or deleted. Keypoints are tied to a position in the frame
order, not to a filename, and said position now belongs to a different image.
Keypoints after the missing frame may display one frame off.

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
