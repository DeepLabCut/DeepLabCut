import numpy as np
import os
import pandas as pd
from conftest import TEST_DATA_DIR
from deeplabcut.generate_training_dataset import (
    read_image_shape_fast,
    SplitTrials,
    format_training_data,
    format_multianimal_training_data,
    trainingsetmanipulation,
    multiple_individuals_trainingsetmanipulation,
)

from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.utils.conversioncode import guarantee_multiindex_rows
from skimage import color, io


def test_read_image_shape_fast(tmp_path):
    path_rgb_image = os.path.join(TEST_DATA_DIR, "image.png")
    img = imread(path_rgb_image, mode="skimage")
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


def test_format_training_data(monkeypatch):
    fake_shape = 3, 480, 640
    monkeypatch.setattr(
        trainingsetmanipulation, "read_image_shape_fast", lambda _: fake_shape,
    )
    df = pd.read_hdf(os.path.join(TEST_DATA_DIR, "trimouse_calib.h5")).xs(
        "mus1", level="individuals", axis=1
    )
    guarantee_multiindex_rows(df)
    train_inds = list(range(10))
    _, data = format_training_data(df, train_inds, 12, "")
    assert len(data) == len(train_inds)
    # Check data comprise path, shape, and xy coordinates
    assert all(len(d) == 3 for d in data)
    assert all(
        (d[0].size == 3 and d[0].dtype.char == "U" and d[0][0, -1].endswith(".png"))
        for d in data
    )
    assert all(np.all(d[1] == np.array(fake_shape)[None]) for d in data)
    assert all(
        (d[2][0, 0].shape[1] == 3 and d[2][0, 0].dtype == np.int64) for d in data
    )


def test_format_multianimal_training_data(monkeypatch):
    fake_shape = 3, 480, 640
    monkeypatch.setattr(
        multiple_individuals_trainingsetmanipulation,
        "read_image_shape_fast",
        lambda _: fake_shape,
    )
    df = pd.read_hdf(os.path.join(TEST_DATA_DIR, "trimouse_calib.h5"))
    guarantee_multiindex_rows(df)
    train_inds = list(range(10))
    n_decimals = 1
    data = format_multianimal_training_data(df, train_inds, "", n_decimals)
    assert len(data) == len(train_inds)
    assert all(isinstance(d, dict) for d in data)
    assert all(len(d["image"]) == 3 for d in data)
    assert all(np.all(d["size"] == np.array(fake_shape)) for d in data)
    assert all(
        (xy.shape[1] == 3 and np.isfinite(xy).all())
        for d in data
        for xy in d["joints"].values()
    )
