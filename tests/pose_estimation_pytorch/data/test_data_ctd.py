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
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deeplabcut.pose_estimation_pytorch.data.ctd import CondFromFile


CONDITIONS = [
    np.zeros((4, 3, 3)).tolist(),
    np.ones((4, 3, 3)).tolist(),
    2 * np.ones((4, 3, 3)).tolist(),
    3 * np.ones((4, 3, 3)).tolist(),
]


@pytest.mark.parametrize("path_prefix", ["/a/b"])
@pytest.mark.parametrize(
    "data",
    [
        [("/a/b/c/d.png", "/a/b/c/d.png", CONDITIONS[1])],
        [("/a/b/c/d.png", "c/d.png", CONDITIONS[1])],
        [
            ("/a/b/c.png", "c.png", CONDITIONS[1]),
            ("/a/b/c/d.png", "c/d.png", CONDITIONS[2]),
            ("/a/b/c/e.png", "/a/b/c/e.png", CONDITIONS[3]),
        ],
    ],
)
def test_ctd_load_json_containing_rel_paths(
    tmp_path_factory,
    path_prefix: str | Path,
    data: tuple[list[str], list[str], list],
) -> None:
    print("Starting test")
    # convert the image paths to Windows format
    if platform.system() == "Windows":
        print("Converting to windows filesystem")

        print("Path Prefix:", path_prefix)
        if isinstance(path_prefix, Path):
            print(f"  As string: {str(path_prefix)}")
            path_prefix = Path(_to_windows_path(str(path_prefix)))
        else:
            path_prefix = _to_windows_path(path_prefix)
        print(f"  Converted {path_prefix}")

        data = [
            (_to_windows_path(img), _to_windows_path(key), cond)
            for img, key, cond in data
        ]
        print(f"Images: {[d[0] for d in data]}")
        print(f"Condition keys: {[d[1] for d in data]}")
        print("---")

    images = [img for img, _, _ in data]
    conditions = {key: cond for _, key, cond in data}

    tmp_folder = Path(tmp_path_factory.mktemp("tmp-project"))
    conditions_filepath = tmp_folder / "conditions.json"
    with open(conditions_filepath, "w") as f:
        json.dump(conditions, f)

    conditions = CondFromFile.load_conditions_json(
        conditions_filepath,
        images,
        path_prefix=path_prefix,
    )
    for img_path, _, condition in data:
        assert img_path in conditions
        np.testing.assert_allclose(condition, conditions[img_path])


@pytest.mark.parametrize("path_prefix", ["/p"])
@pytest.mark.parametrize("num_conditions", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("num_bodyparts", [1, 2, 3, 5, 10])
@pytest.mark.parametrize(
    "data",
    [
        [("/p/data/video0/img0.png", ("data", "video0", "img0.png"))],
        [("/p/data/video0/img0.png", "data/video0/img0.png")],
        [
            ("/p/b/c/d0.png", ("b", "c", "d0.png")),
            ("/p/b/c/d1.png", ("b", "c", "d1.png")),
            ("/p/b/c/d2.png", ("b", "c", "d2.png")),
        ],
        [
            ("/p/b/c/d0.png", "b/c/d0.png"),
            ("/p/b/c/d1.png", "b/c/d1.png"),
            ("/p/b/c/d2.png", "b/c/d2.png"),
        ],
    ],
)
def test_ctd_load_hdf_containing_rel_paths(
    tmp_path_factory,
    path_prefix: str | Path,
    num_conditions: int,
    num_bodyparts: int,
    data: tuple[list[str], list[str]],
) -> None:
    print("\nStarting test")

    # convert the image paths to Windows format
    if platform.system() == "Windows":
        print("Converting to windows filesystem")

        print("Path Prefix:", path_prefix)
        if isinstance(path_prefix, Path):
            print(f"  As string: {str(path_prefix)}")
            path_prefix = Path(_to_windows_path(str(path_prefix)))
        else:
            path_prefix = _to_windows_path(path_prefix)
        print(f"  Converted {path_prefix}")

        data = [(_to_windows_path(img), idx) for img, idx in data]
        print(f"Images: {[d[0] for d in data]}")
        print("---")

    num_images = len(data)
    images = [img for img, _ in data]
    index = [idx for _, idx in data]
    if isinstance(index[0], tuple):
        index = pd.MultiIndex.from_tuples(index)

    # generate random pose data
    size = (num_images, num_conditions, num_bodyparts, 3)
    rng = np.random.default_rng(0)
    pose = rng.integers(low=0, high=1024, size=size).astype(float)
    pose[:, :, :, 2] = rng.random(size=(num_images, num_conditions, num_bodyparts))

    # set some missing data
    is_nans = rng.random(size=size) > 0.8
    pose[is_nans] = np.nan

    # create what the output data will look like
    keypoint_mask = np.any(is_nans, axis=3)
    output_pose = pose.copy()
    output_pose[keypoint_mask] = 0.0
    idv_mask = ~np.all(keypoint_mask, axis=2)

    output_pose = [
        p[p_mask] if np.any(p_mask) else np.zeros((0, num_bodyparts, 3))
        for p, p_mask in zip(output_pose, idv_mask)
    ]

    # generate columns for the dataframe
    columns = pd.MultiIndex.from_product(
        [
            ["scorer"],
            [f"idv{i}" for i in range(num_conditions)],
            [f"bpt{i}" for i in range(num_bodyparts)],
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(data=pose.reshape(num_images, -1), index=index, columns=columns)

    print(df.head())

    tmp_folder = Path(tmp_path_factory.mktemp("tmp-project"))
    conditions_filepath = tmp_folder / "conditions.h5"
    df.to_hdf(conditions_filepath, key="df_with_missing")

    conditions = CondFromFile.load_conditions_h5(
        conditions_filepath, images, path_prefix=path_prefix
    )
    for idx, (img_path, img_index) in enumerate(data):
        assert img_path in conditions
        np.testing.assert_allclose(output_pose[idx], conditions[img_path])


def _to_windows_path(s: str) -> str:
    # Convert absolute paths to paths on C:
    if s.startswith("/"):
        return str(Path("C:\\", *s[1:].split("/")))

    return s
