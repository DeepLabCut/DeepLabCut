# Some data processing recipes!

## Stitching tracklets in batch

This ultimate step of the maDLC pipeline is not batched as the number of tracks to
reconstruct may vary from a video to another. If that number is constant across videos though,
stitching tracklets in batch could be done as follows:

```python
from deeplabcut import stitch_tracklets
from deeplabcut.utils import grab_files_in_folder

root_folder = 'path_to_the_folder_containing_your_ellipse_pickles'
for pickle_file in grab_files_in_folder(root_folder, ext='_el.pickle', relative=False):
   stitch_tracklets(config_path, pickle_file)
```


## Flagging frames with abnormal bodypart distances

Beyond `deeplabcut.check_labels`, you may want to automatically detect
labeled frames where the distance between two body parts exceeds a
given threshold. For example, the frames where the head–tail length
is greater than 100 pixels could be found as follows:

```python
import numpy as np
import pandas as pd

max_dist = 100
df = pd.read_hdf('path_to_your_labeled_data_file')
bpt1 = df.xs('head', level='bodyparts', axis=1).to_numpy()
bpt2 = df.xs('tail', level='bodyparts', axis=1).to_numpy()
# We calculate the vectors from a point to the other
# and group them per frame and per animal.
diff = (bpt1 - bpt2).reshape((len(df), -1, 2))
dist = np.linalg.norm(diff, axis=2)
mask = np.any(dist >= max_dist, axis=1)
flagged_frames = df.iloc[mask].index
```
