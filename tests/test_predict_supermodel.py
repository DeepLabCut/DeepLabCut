#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import numpy as np
import pytest
from deeplabcut.pose_estimation_tensorflow.modelzoo.api import superanimal_inference


def test_get_multi_scale_frames():
    fake_img = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    ar = fake_img.shape[1] / fake_img.shape[0]
    heights = list(range(100, 1000, 100))
    frames, shapes = superanimal_inference.get_multi_scale_frames(
        fake_img,
        heights,
    )
    assert len(frames) == len(shapes) == len(heights)
    assert all(shape[0] == h for shape, h in zip(shapes, heights))
    assert all(round(shape[0] * ar) == shape[1] for shape in shapes)


@pytest.mark.parametrize("scale", [0.7, 1.5, 2])
def test_project_pred_to_original_size(scale):
    old_shape = 400, 600, 3
    new_shape = old_shape[0] // scale, old_shape[1] // scale, 3
    xs = [10, 25, 50, 100]
    conf = [[1] for _ in range(len(xs))]
    coords = [[np.array([[x, x]]) for x in xs]]
    preds = {
        "coordinates": coords,
        "confidence": conf,
    }
    preds_orig = superanimal_inference._project_pred_to_original_size(
        preds,
        old_shape,
        new_shape,
    )
    coords_orig = preds_orig["coordinates"][0]
    assert len(coords_orig) == len(xs)
    assert all([round(x * scale) == round(xy[0]) for xy, x in zip(coords_orig, xs)])
