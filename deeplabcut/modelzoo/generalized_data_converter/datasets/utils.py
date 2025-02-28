#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from functools import lru_cache

import numpy as np
from PIL import Image


def calc_bboxes_from_keypoints(data, slack=0, offset=0, clip=False):
    data = np.asarray(data)
    if data.shape[-1] < 3:
        raise ValueError("Data should be of shape (n_animals, n_bodyparts, 3)")

    if data.ndim != 3:
        data = np.expand_dims(data, axis=0)
    bboxes = np.full((data.shape[0], 5), np.nan)
    bboxes[:, :2] = np.nanmin(data[..., :2], axis=1) - slack  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(data[..., :2], axis=1) + slack  # X2, Y2
    bboxes[:, -1] = np.nanmean(data[..., 2])  # Average confidence
    bboxes[:, [0, 2]] += offset
    if clip:
        coord = bboxes[:, :4]
        coord[coord < 0] = 0
    return bboxes


@lru_cache(maxsize=None)
def read_image_shape_fast(path):
    # Blazing fast and does not load the image into memory
    with Image.open(path) as img:
        width, height = img.size
        return len(img.getbands()), height, width
