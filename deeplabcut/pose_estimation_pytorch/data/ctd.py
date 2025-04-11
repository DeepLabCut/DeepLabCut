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
from __future__ import annotations

import json
import pickle
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
from deeplabcut.pose_estimation_pytorch.data.snapshots import Snapshot
from deeplabcut.pose_estimation_pytorch.task import Task


class CondProvider(ABC):
    """A class providing conditions for a CTD model."""

    @classmethod
    def get_loader_and_snapshot(
        cls,
        config: str | Path,
        shuffle: int,
        trainset_index: int = 0,
        modelprefix: str = "",
        snapshot: str | None = None,
        snapshot_index: int | None = None,
    ) -> tuple[DLCLoader, Snapshot]:
        """Creates a DLCLoader for the BU shuffle and the path to conditions snapshot.

        One of `snapshot` or `snapshot_index` must be provided.

        Args:
            config: Path to the DeepLabCut project config, or the project config itself
            trainset_index: The index of the TrainingsetFraction for which to load data
            shuffle: The index of the shuffle for which to load data.
            modelprefix: The modelprefix for the shuffle.
            snapshot: The name of the snapshot to use.
            snapshot_index: The index of the snapshot to use. If `snapshot` is
                provided, the `snapshot_index` is not used.

        Returns:
            loader: The DLCLoader for the BU shuffle.
            snapshot: The BU Snapshot to use for conditions.

        Raises:
            ValueError: If the given shuffle is not for a BU model.
        """
        loader = DLCLoader(
            config,
            trainset_index=trainset_index,
            shuffle=shuffle,
            modelprefix=modelprefix,
        )
        if loader.pose_task != Task.BOTTOM_UP:
            raise ValueError(
                "Conditions can only be loaded from shuffles for bottom-up models, but "
                f"shuffle {shuffle} has task {loader.pose_task} (config={config}, "
                f"trainset_index={trainset_index}, modelprefix={modelprefix})."
            )

        if snapshot is not None:
            snapshot_path = loader.model_folder / snapshot
            if not snapshot_path.exists():
                raise ValueError(f"Snapshot file {snapshot_path} does not exist.")
            bu_snapshot = Snapshot.from_path(snapshot_path)

        else:
            if snapshot_index is None:
                snapshot_index = -1

            snapshots = loader.snapshots()
            if len(snapshots) == 0:
                raise ValueError(
                    f"No snapshots found for shuffle={shuffle} in {loader.model_folder}"
                )

            if snapshot_index > len(snapshots):
                snapshot_str = "\n".join(
                    [f"  {i}: {s.path.name}" for i, s in enumerate(snapshots)]
                )
                raise ValueError(
                    f"Snapshot index {snapshot_index} is out of range. Existing "
                    f"snapshots: {snapshot_str}"
                )

            bu_snapshot = snapshots[snapshot_index]

        return loader, bu_snapshot


class CondFromFile(CondProvider):
    """A class providing conditions for a CTD model from a file

    Args:
        filepath: The path to the file containing the conditions for the CTD model.
            These conditions must be pose predictions made by a BU model on the data
        images: Only load the conditions for the given image keys.
        kwargs: A `CondFromFile` instance can also be created from a DeepLabCut
            shuffle by passing kwargs and setting `filepath=None`. See examples for more
            information.
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        **kwargs,
    ) -> None:
        if filepath is None:
            # Load the conditions filepath from the Shuffle
            bu_loader, bu_snapshot = self.get_loader_and_snapshot(**kwargs)
            bu_scorer = bu_loader.scorer(bu_snapshot)
            filepath = bu_loader.evaluation_folder / f"{bu_scorer}.h5"
            if not filepath.exists():
                raise ValueError(
                    f"Conditions file {filepath} does not exist. Please make sure "
                    f"snapshot {bu_snapshot.path.name} for {kwargs['shuffle']} "
                    f"was evaluated (which is when the predictions file is created)."
                )

        if not filepath.exists():
            raise ValueError(
                "Conditions file {conditions_filepath} does not exist. Please check "
                f"the given path."
            )

        self.filepath = filepath

    def load_conditions(
        self,
        images: list[str] | None = None,
        path_prefix: str | None = None,
    ) -> dict[str, np.ndarray] | list[np.ndarray]:
        """Loads conditions for a model from a file.

        When loading conditions for individual images, the `images` must be provided
        (indicating which images to load conditions for). A dict is returned containing
        the conditions for each requested image.

        When loading conditions for a video, the `images` parameter must be set to None.
        A list is returned containing the conditions for each frame.

        Args:
            images: A list of image paths to load conditions for.
            path_prefix: Optional prefix to prepend to image paths when looking up
                conditions. This is useful when the paths in the conditions file are
                relative but the provided image paths are absolute, or vice versa.

        Returns:
            If "images" is given: a dictionary mapping image paths to condition arrays.
                Each array has shape (num_conditions, num_bodyparts, 3).
            If "images" is None: a list containing the conditions for each frame.
        """
        suffix = Path(self.filepath).suffix.lower()
        if suffix == ".h5":
            return self.load_conditions_h5(self.filepath, images, path_prefix)
        elif suffix == ".json":
            return self.load_conditions_json(self.filepath, images, path_prefix)
        elif suffix == ".pickle":
            return self.load_conditions_pickle(self.filepath)

        raise ValueError(
            f"Unknown file suffix {suffix}. Can only read conditions from HDF5 or JSON "
            f"files. Received {self.filepath}."
        )

    @staticmethod
    def load_conditions_h5(
        filepath: str | Path,
        images: list[str] | None = None,
        path_prefix: str | Path | None = None,
    ) -> dict[str, np.ndarray] | list[np.ndarray]:
        """Loads conditions for a model from a pandas DataFrame stored in an HDF file

        When loading conditions for individual images, the `images` must be provided
        (indicating which images to load conditions for). A dict is returned containing
        the conditions for each requested image.

        When loading conditions for a video, the `images` parameter must be set to None.
        A list is returned containing the conditions for each frame.

        The DataFrame must be in the same format as DeepLabCut Predictions. For
        predictions on images (e.g. on a training/test set), the DataFrame should be in
        the format:

            ```
            scorer                                model-name  ...
            individuals                                 idv0  ...                   idvM
            bodyparts                                   bpt0  ...                   bptN
            coords                        x     y likelihood  ...    x      y likelihood
            ----------------------------------------------------------------------------
            (labeled-data, v0, 0.png)  87.0  62.0       0.73  ...  83.2  99.1     0.8326
            ```

        While for conditions for videos, the DataFrame should be in the format:

            ```
            scorer                      model-name  ...
            individuals                       idv0  ...                   idvM
            bodyparts                         bpt0  ...                   bptN
            coords              x     y likelihood  ...    x      y likelihood
            ----------------------------------------------------------------------------
            frame0000.png    87.0  62.0       0.73  ...  83.2  99.1     0.8326
            ```

        Args:
            images: A list of image paths to load conditions for
            filepath: Path to the JSON file containing conditions.
            path_prefix: Optional prefix to prepend to image paths when looking up
                conditions. This is useful when the paths in the conditions file are
                relative but the provided image paths are absolute, or vice versa.

        Returns:
             If "images" is given: a dictionary mapping image paths to condition arrays.
                Each array has shape (num_conditions, num_bodyparts, 3).
            If "images" is None: a list containing the conditions for each frame.
        """
        def _parse_row(df_row) -> np.ndarray:
            # Row to numpy and reshape
            pose = df_row.to_numpy().reshape((num_conditions, num_bodyparts, 3))

            # Remove missing data
            missing_keypoints = np.any(np.isnan(pose) | (pose < 0), axis=2)
            pose[missing_keypoints] = 0

            # Only keep conditions with at least one visible keypoint
            visible_conditions = np.any(~missing_keypoints, axis=1)
            if np.sum(visible_conditions) > 0:
                pose = pose[visible_conditions]
            else:
                pose = np.zeros((0, num_bodyparts, 3))

            return pose

        if path_prefix is not None:
            path_prefix = Path(path_prefix)

        df = pd.read_hdf(filepath)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{filepath} is not a dataframe.")

        num_bodyparts = len(df.columns.get_level_values("bodyparts").unique())
        num_conditions = 1
        if "individuals" in df.columns.names:
            num_conditions = len(df.columns.get_level_values("individuals").unique())

        # Parse as list and return
        if images is None:
            parsed = []
            for _, cond in df.iterrows():
                parsed.append(_parse_row(cond))

            return parsed

        image_set = set(images)
        conditions = {}
        for filename, row in df.iterrows():
            if isinstance(filename, tuple):
                filename = str(Path(*filename))

            if path_prefix is not None and filename not in image_set:
                filename = str(path_prefix / filename)

            if filename in image_set:
                conditions[filename] = _parse_row(row)

        missing = image_set.difference(set(conditions.keys()))
        if len(missing) > 0:
            print(
                f"Warning: did not find conditions for {len(missing)} of the {len(images)} "
                f"images. Missing conditions:"
            )
            for img_path in missing:
                print(f"  - {img_path}")

        return conditions

    @staticmethod
    def load_conditions_json(
        filepath: str | Path,
        images: list[str] | None = None,
        path_prefix: str | Path | None = None,
    ) -> dict[str, np.ndarray] | list[np.ndarray]:
        """Loads conditions for a model from a JSON file.

        When loading conditions for individual images, the `images` must be provided
        (indicating which images to load conditions for). A dict is returned containing
        the conditions for each requested image. The JSON data structure should be:

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

        When loading conditions for a video, the `images` parameter must be set to None.
        A list is returned containing the conditions for each frame. The JSON data
        structure should be:

            ```
            [
                [  # conditions for frame 0
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
                [ ... ],   # conditions for frame 1
                ...
                [ ... ] # conditions for frame N
            ]
            ```

        Args:
            images: A list of image paths to load conditions for.
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

        # Parse list and return
        if images is None:
            if not isinstance(conditions, list):
                raise ValueError(
                    f"Conditions are expected to be of type list when `images=None`, "
                    f"got {type(conditions)}."
                )

            parsed = []
            for cond in conditions:
                if len(cond) == 0:
                    parsed.append(np.zeros((0, 0, 3)))
                else:
                    parsed.append(np.asarray(cond))
            return parsed

        if not isinstance(conditions, dict):
            raise ValueError(
                f"Conditions are expected to be of type dict, got {type(conditions)}. "
                "They should be in the format 'labeled-data/video-0/img0000.png' -> "
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
                f"Warning: did not find conditions for {len(missing)} of the "
                f"{len(images)} images. Missing conditions:"
            )
            for img_path in missing:
                print(f"  - {img_path}")

        return parsed

    @staticmethod
    def load_conditions_pickle(filepath: str | Path) -> list[np.ndarray]:
        """Loads conditions from a `*_assemblies.pickle` file containing predictions

        Args:
            filepath: Path to the Pickle file containing conditions.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        frames = [f for f in data.keys() if isinstance(f, int)]
        n_frames = max(*frames)

        parsed = []
        for i in range(n_frames):
            assemblies = data.get(i)
            if assemblies is None or len(assemblies) == 0:
                pose = np.zeros((0, 0, 3))
            else:
                pose = np.stack(assemblies, axis=0)[:, :, :3]

                mask = np.any(np.all(pose > 0, axis=-1), axis=-1)
                if np.sum(mask) == 0:
                    pose = np.zeros((0, 0, 3))
                else:
                    pose = pose[mask]

            parsed.append(pose)
        return parsed


class CondFromModel(CondProvider):
    """A class providing conditions for a CTD model from a BU model.

    Attributes:
        config_path: (Path)
            The path to the `pytorch_config.yaml` for the BU model to use as conditions.
        snapshot_path: (Path)
            The path to the BU snapshot to use to generate conditions for the CTD model.
        scorer: str
            The scorer name for the BU model. This can be used to look for files
            containing conditions instead of recomputing them.

    Args:
        config_path: (Path)
            The path to the `pytorch_config.yaml` for the BU model to use as conditions.
        snapshot_path: (Path)
            The path to the BU snapshot to use to generate conditions for the CTD model.
        **kwargs: A `CondFromModel` instance can also be created from a DeepLabCut
            shuffle. See examples for more information.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        snapshot_path: str | Path | None = None,
        scorer: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is not None and snapshot_path is not None:
            config_path = Path(config_path)
            snapshot_path = Path(config_path)
        elif "config" in kwargs and "shuffle" in kwargs:
            bu_loader, snapshot = self.get_loader_and_snapshot(**kwargs)
            config_path = bu_loader.model_config_path
            snapshot_path = snapshot.path
            if scorer is None:
                scorer = bu_loader.scorer(snapshot)

        self.config_path = config_path
        self.snapshot_path = snapshot_path
        self.scorer = scorer
