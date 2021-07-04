import numpy as np
import os
from conftest import TEST_DATA_DIR
from deeplabcut.generate_training_dataset import (
    read_image_shape_fast,
    SplitTrials,
)
from skimage import io, color


def test_read_image_shape_fast(tmp_path):
    path_rgb_image = os.path.join(TEST_DATA_DIR, "image.png")
    img = io.imread(path_rgb_image)
    shape = img.shape
    assert read_image_shape_fast(path_rgb_image) == (shape[2], shape[0], shape[1])
    path_gray_image = str(tmp_path / "gray.png")
    io.imsave(path_gray_image, color.rgb2gray(img).astype(np.uint8))
    assert read_image_shape_fast(path_gray_image) == (1, shape[0], shape[1])


def test_split_trials():
    n_rows = 123
    train_fractions = np.arange(50, 96) / 100
    for frac in train_fractions:
        train_inds, test_inds = SplitTrials(
            range(n_rows), frac, enforce_train_fraction=True,
        )
        assert (len(train_inds) / (len(train_inds) + len(test_inds))) == frac
        train_inds = train_inds[train_inds != -1]
        test_inds = test_inds[test_inds != -1]
        assert (len(train_inds) + len(test_inds)) == n_rows
