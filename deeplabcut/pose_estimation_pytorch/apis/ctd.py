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
"""Methods to help with conditional top-down models"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_conditions_h5(
    images: list[str],
    filepath: str | Path,
    path_prefix: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Loads conditions for a model from a pandas DataFrame stored in an HDF file

    The DataFrame must be in the same format as DeepLabCut Predictions:

        ```
        scorer                                   model-name  ...
        individuals                                    idv0  ...                   idvM
        bodyparts                                      bpt0  ...                   bptN
        coords                           x     y likelihood  ...    x      y likelihood
        ---------------------------------------------------------------------------------
        (labeled-data, v0, img0.png)  87.0  62.0       0.73  ...  83.2  99.1     0.8326
        ```

    Args:
        images: A list of image paths to load conditions for
        filepath: Path to the JSON file containing conditions.
        path_prefix: Optional prefix to prepend to image paths when looking up
            conditions. This is useful when the paths in the conditions file are
            relative but the provided image paths are absolute, or vice versa.

    Returns:
        A dictionary mapping image paths to condition arrays. Each array has shape
        (num_conditions, num_bodyparts, 3).
    """
    if path_prefix is not None:
        path_prefix = Path(path_prefix)

    df = pd.read_hdf(filepath)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{filepath} is not a dataframe.")

    num_bodyparts = len(df.columns.get_level_values("bodyparts").unique())
    num_conditions = 1
    if "individuals" in df.columns.names:
        num_conditions = len(df.columns.get_level_values("individuals").unique())

    image_set = set(images)
    conditions = {}
    for filename, row in df.iterrows():
        if isinstance(filename, tuple):
            filename = str(Path(*filename))

        if path_prefix is not None and filename not in image_set:
            filename = str(path_prefix / filename)

        if filename in image_set:
            pose = row.to_numpy().reshape((num_conditions, num_bodyparts, 3))

            # Remove NaNs and set likelihood to 0 for missing keypoints
            missing_keypoints = np.any(np.isnan(pose) | (pose < 0), axis=2)
            pose[missing_keypoints] = 0

            # Only keep conditions with at least one visible keypoint
            visible_conditions = np.any(~missing_keypoints, axis=1)
            if np.sum(visible_conditions) > 0:
                pose = pose[visible_conditions]
            else:
                pose = np.zeros((0, num_bodyparts, 3))

            conditions[filename] = pose

    missing = image_set.difference(set(conditions.keys()))
    if len(missing) > 0:
        print(
            f"Warning: did not find conditions for {len(missing)} of the {len(images)} "
            f"images. Missing conditions:"
        )
        for img_path in missing:
            print(f"  - {img_path}")

    return conditions


def load_conditions_json(
    images: list[str],
    filepath: str | Path,
    path_prefix: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Loads conditions for a model from a JSON file.

    The JSON file must contain data in the format:
        ```
        {
            "img000.png": [  # conditions for image 0
                [  # condition 0 pose
                    [x, y, score],  # keypoint 0
                    [x, y, score],  # keypoint 1
                    ...
                    [x, y, score],  # keypoint N
                ],
                [ ... ], # condition 1
                ...
                [ ... ] # condition M
            ],
            "img001.png": [...]  # conditions for image 1
        }
        ```

    Args:
        images: A list of image paths to load conditions for
        filepath: Path to the JSON file containing conditions.
        path_prefix: Optional prefix to prepend to image paths when looking up
            conditions. This is useful when the paths in the conditions file are
            relative but the provided image paths are absolute, or vice versa.

    Returns:
        A dictionary mapping image paths to condition arrays. Each array has shape
        (num_conditions, num_bodyparts, 3).
    """
    with open(filepath, "r") as f:
        conditions = json.load(f)

    if not isinstance(conditions, dict):
        raise ValueError(
            f"Conditions are expected to be of type dict, got {type(conditions)}. They "
            "should be in the format 'labeled-data/video-0/img0000.png' -> "
            "list[list[list[float]]], where the list represents an array of shape "
            "(num_conditions, num_bodyparts, 3)."
        )

    path_with_prefix_to_key = {}
    if path_prefix is not None:
        path_with_prefix_to_key = {
            str(Path(path_prefix) / k): k for k in conditions.keys()
        }

    parsed = {}
    missing = []
    for img_path in images:
        if img_path in conditions:
            pose = np.asarray(conditions[img_path])
        elif img_path in path_with_prefix_to_key:
            pose = np.asarray(conditions[path_with_prefix_to_key[img_path]])
        else:
            pose = np.zeros((0, 0, 3))
            missing.append(img_path)

        if len(pose) == 0:
            pose = np.zeros((0, 0, 3))

        parsed[img_path] = pose

    if len(missing) > 0:
        print(
            f"Warning: did not find conditions for {len(missing)} of the {len(images)} "
            f"images. Missing conditions:"
        )
        for img_path in missing:
            print(f"  - {img_path}")

    return parsed


def load_conditions(
    images: list[str],
    filepath: str | Path,
    path_prefix: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Loads conditions for a model from a file

    Args:
        images: A list of image paths to load conditions for
        filepath: Path to the file containing conditions. Must be either a JSON (with a
            ".json" suffix) or HDF5 file (with a ".h5" suffix).
        path_prefix: Optional prefix to prepend to image paths when looking up
            conditions. This is useful when the paths in the conditions file are
            relative but the provided image paths are absolute, or vice versa.

    Returns:
        A dictionary mapping image paths to condition arrays. Each array has shape
        (num_conditions, num_bodyparts, 3).
    """
    suffix = Path(filepath).suffix.lower()
    if suffix == ".h5":
        return load_conditions_h5(images, filepath, path_prefix)
    elif suffix == ".json":
        return load_conditions_json(images, filepath, path_prefix)

    raise ValueError(
        f"Unknown file suffix {suffix}. Can only read conditions from HDF5 or JSON "
        f"files. Received {filepath}."
    )
