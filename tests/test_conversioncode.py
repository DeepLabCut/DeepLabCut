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

from deeplabcut.core.config import ProjectConfig
from deeplabcut.utils import conversioncode


def test_guarantee_multiindex_rows(test_data_dir):
    df_unix = pd.read_hdf(os.path.join(test_data_dir, "trimouse_calib.h5"))
    df_posix = df_unix.copy()
    df_posix.index = df_posix.index.str.replace("/", "\\")
    nrows = len(df_unix)
    for df in (df_unix, df_posix):
        conversioncode.guarantee_multiindex_rows(df)
        assert isinstance(df.index, pd.MultiIndex)
        assert len(df) == nrows
        assert df.index.nlevels == 3
        assert all(df.index.get_level_values(0) == "labeled-data")
        assert all(img.endswith(".png") for img in df.index.get_level_values(2))


@pytest.mark.parametrize(
    "video",
    [
        "/data/videos/mouse1.mp4",
        r"C:\data\videos\mouse1.mp4",
    ],
)
def test_adapt_labeled_data_to_new_project_adds_new_bodyparts(tmp_path, video):
    config_path = tmp_path / "config.yaml"
    ProjectConfig(
        Task="task",
        scorer="me",
        project_path=tmp_path,
        multianimalproject=True,
        bodyparts="MULTI!",
        individuals=["mus1", "mus2"],
        multianimalbodyparts=["nose", "tail", "ear"],
        video_sets={video: {"crop": "0, 100, 0, 100"}},
    ).to_yaml(config_path)

    # Synthetic labels annotated before "ear" was added to the project
    folder = tmp_path / "labeled-data" / "mouse1"
    folder.mkdir(parents=True)
    columns = pd.MultiIndex.from_product(
        [["me"], ["mus1", "mus2"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples([("labeled-data", "mouse1", f"img{i}.png") for i in range(3)])
    df = pd.DataFrame(np.arange(24, dtype=float).reshape(3, 8), index=index, columns=columns)
    df.to_csv(folder / "CollectedData_me.csv")

    conversioncode.adapt_labeled_data_to_new_project(config_path)

    adapted = pd.read_hdf(folder / "CollectedData_me.h5")
    for individual in ("mus1", "mus2"):
        bodyparts = adapted.xs(individual, level="individuals", axis=1).columns.get_level_values("bodyparts")
        assert list(bodyparts.unique()) == ["nose", "tail", "ear"]
        assert adapted.xs((individual, "ear"), level=("individuals", "bodyparts"), axis=1).isna().all().all()
    pd.testing.assert_frame_equal(adapted.loc[:, df.columns], df, check_names=False)
