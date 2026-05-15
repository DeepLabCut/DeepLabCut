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
import os

import numpy as np
import pandas as pd
import pytest
from conftest import TEST_DATA_DIR
from skimage import color, io

from deeplabcut.generate_training_dataset import (
    SplitTrials,
    format_multianimal_training_data,
    format_training_data,
    multiple_individuals_trainingsetmanipulation,
    parse_video_filenames,
    read_image_shape_fast,
    trainingsetmanipulation,
)
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.utils.conversioncode import guarantee_multiindex_rows


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
            range(n_rows),
            frac,
            enforce_train_fraction=True,
        )
        assert (len(train_inds) / (len(train_inds) + len(test_inds))) == frac
        train_inds = train_inds[train_inds != -1]
        test_inds = test_inds[test_inds != -1]
        assert (len(train_inds) + len(test_inds)) == n_rows


def test_format_training_data(monkeypatch):
    fake_shape = 3, 480, 640
    monkeypatch.setattr(
        trainingsetmanipulation,
        "read_image_shape_fast",
        lambda _: fake_shape,
    )
    df = pd.read_hdf(os.path.join(TEST_DATA_DIR, "trimouse_calib.h5")).xs("mus1", level="individuals", axis=1)
    guarantee_multiindex_rows(df)
    train_inds = list(range(10))
    _, data = format_training_data(df, train_inds, 12, "")
    assert len(data) == len(train_inds)
    # Check data comprise path, shape, and xy coordinates
    assert all(len(d) == 3 for d in data)
    assert all((d[0].size == 3 and d[0].dtype.char == "U" and d[0][0, -1].endswith(".png")) for d in data)
    assert all(np.all(d[1] == np.array(fake_shape)[None]) for d in data)
    assert all((d[2][0, 0].shape[1] == 3 and d[2][0, 0].dtype == np.int64) for d in data)


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
    assert all((xy.shape[1] == 3 and np.isfinite(xy).all()) for d in data for xy in d["joints"].values())


@pytest.mark.parametrize(
    "videos, expected_filenames",
    [
        ([], []),
        (["/data/my-video.mov"], ["my-video"]),
        (["/data/my-video.mp4", "/data2/my-video.mov"], ["my-video"]),
        (["/data/my-video.mov", "/data/video2.mov"], ["my-video", "video2"]),
        (["/a/v1.mov", "/a/v2.mp4", "/b/v1.mov"], ["v1", "v2"]),
        (["v1.mov", "v2.mov", "v1.mov"], ["v1", "v2"]),
        (["/a/v1.mp4", "/a/v2.mov", "/b/v2.mov"], ["v1", "v2"]),
        (["/a/v1.mp4", "/a/v2.mov", "/b/v2.mov", "/b/v3.mp4"], ["v1", "v2", "v3"]),
    ],
)
def test_parse_video_filenames(videos: list[str], expected_filenames: list[str]):
    filenames = parse_video_filenames(videos)
    assert filenames == expected_filenames


def test_format_training_data_ignores_likelihood_columns(monkeypatch):
    fake_shape = 3, 480, 640
    monkeypatch.setattr(
        trainingsetmanipulation,
        "read_image_shape_fast",
        lambda _: fake_shape,
    )

    # Base single-animal dataframe (x/y only)
    df = pd.read_hdf(os.path.join(TEST_DATA_DIR, "trimouse_calib.h5")).xs(
        "mus1",
        level="individuals",
        axis=1,
    )
    guarantee_multiindex_rows(df)

    # Add a likelihood column so the layout becomes:
    # x, y, likelihood, x, y, likelihood, ...
    new_cols = []
    new_arrays = []

    coord_level = df.columns.names.index("coords")

    for col in df.columns:
        new_cols.append(col)
        new_arrays.append(df[col].to_numpy())

        if col[coord_level] == "y":
            lik_col = list(col)
            lik_col[coord_level] = "likelihood"
            new_cols.append(tuple(lik_col))
            new_arrays.append(np.ones(len(df), dtype=float))

    df_with_likelihood = pd.DataFrame(
        np.column_stack(new_arrays),
        index=df.index,
        columns=pd.MultiIndex.from_tuples(new_cols, names=df.columns.names),
    )

    train_inds = list(range(10))

    baseline_train_data, baseline_matlab_data = format_training_data(df, train_inds, 12, "")
    train_data, matlab_data = format_training_data(df_with_likelihood, train_inds, 12, "")

    # The presence of likelihood columns should not change the formatted result
    assert len(train_data) == len(baseline_train_data)
    assert len(matlab_data) == len(baseline_matlab_data)

    for got, expected in zip(train_data, baseline_train_data, strict=False):
        assert got["image"] == expected["image"]
        assert got["size"] == expected["size"]
        assert np.array_equal(got["joints"], expected["joints"])

    for got, expected in zip(matlab_data, baseline_matlab_data, strict=False):
        assert np.array_equal(got["image"], expected["image"])
        assert np.array_equal(got["size"], expected["size"])
        assert np.array_equal(got["joints"][0, 0], expected["joints"][0, 0])


def test_merge_annotateddatasets_drops_likelihood_columns(tmp_path):
    scorer = "testscorer"
    video_name = "video1"
    bodyparts = ["nose", "tail"]

    project_path = tmp_path
    labeled_data_dir = project_path / "labeled-data" / video_name
    labeled_data_dir.mkdir(parents=True)

    trainingsetfolder_full = project_path / "training-datasets" / "iteration-0"
    trainingsetfolder_full.mkdir(parents=True)

    # Build a single-animal annotation dataframe with x/y/likelihood columns
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )

    index = pd.MultiIndex.from_tuples(
        [("labeled-data", video_name, "img0001.png")],
    )

    data = np.array([[10.0, 20.0, 0.9, 30.0, 40.0, 0.8]])
    df = pd.DataFrame(data, index=index, columns=columns)

    input_h5 = labeled_data_dir / f"CollectedData_{scorer}.h5"
    df.to_hdf(input_h5, key="df_with_missing", mode="w")

    cfg = {
        "project_path": str(project_path),
        "video_sets": {str(project_path / "videos" / f"{video_name}.mp4"): {}},
        "scorer": scorer,
        "bodyparts": bodyparts,
        "multianimalproject": False,
    }

    merged = trainingsetmanipulation.merge_annotateddatasets(
        cfg,
        trainingsetfolder_full,
    )

    # Returned dataframe should not contain likelihood anymore
    coord_level = "coords" if "coords" in merged.columns.names else merged.columns.names[-1]
    assert "likelihood" not in merged.columns.get_level_values(coord_level)

    # Saved merged h5 should also not contain likelihood
    output_h5 = trainingsetfolder_full / f"CollectedData_{scorer}.h5"
    saved = pd.read_hdf(output_h5)

    coord_level = "coords" if "coords" in saved.columns.names else saved.columns.names[-1]
    assert "likelihood" not in saved.columns.get_level_values(coord_level)

    # Sanity check: x/y are preserved
    assert set(saved.columns.get_level_values(coord_level)) == {"x", "y"}
    output_csv = trainingsetfolder_full / f"CollectedData_{scorer}.csv"
    saved_csv = pd.read_csv(output_csv, header=[0, 1, 2], index_col=[0, 1, 2])

    coord_level = "coords" if "coords" in saved_csv.columns.names else saved_csv.columns.names[-1]
    assert "likelihood" not in saved_csv.columns.get_level_values(coord_level)
