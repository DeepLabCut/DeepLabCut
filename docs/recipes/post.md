# Some data processing recipes!

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
try:
    diff = (bpt1 - bpt2).reshape((len(df), -1, 2))
except ValueError:
    diff = (bpt1 - bpt2).reshape((len(df), -1, 3))
dist = np.linalg.norm(diff, axis=2)
mask = np.any(dist >= max_dist, axis=1)
flagged_frames = df.iloc[mask].index
```
