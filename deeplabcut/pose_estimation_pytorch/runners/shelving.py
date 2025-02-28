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
"""Modules used to read/write shelve data during video analysis in DeepLabCut 3.0"""
import pickle
import shelve
from abc import ABC
from pathlib import Path

import numpy as np


class ShelfManager(ABC):
    """Class to manage shelf data"""

    def __init__(self, filepath: str | Path, flag: str = "r") -> None:
        self.filepath = Path(filepath)
        self.flag = flag

        self._db: shelve.Shelf | None = None
        self._open: bool = False

    def open(self) -> None:
        """Opens the shelf"""
        self._db = shelve.open(
            str(self.filepath),
            flag=self.flag,
            protocol=pickle.DEFAULT_PROTOCOL,
        )
        self._open = True

    def close(self) -> None:
        """Closes the shelf"""
        if not self._open:
            return

        try:
            self._db.close()
        except AttributeError:
            pass

        self._open = False

    def keys(self) -> list[str]:
        if not self._open:
            raise ValueError(f"You must call open() before reading keys!")

        return [k for k in self._db]


class ShelfReader(ShelfManager):
    """Reads data from a shelf"""

    def __getitem__(self, item: str) -> dict:
        """Reads an item from the shelf.

        Args:
            item: The key of the item to read.

        Returns:
            The item.
        """
        if not self._open:
            raise ValueError(f"You must call open() before reading data!")

        return self._db[item]


class ShelfWriter(ShelfManager):
    """Writes data to a shelf on-the-fly during video analysis.

    Args:
        pose_cfg: The test pose config for the model.
        filepath: The path where the data should be saved.
        num_frames: The number of frames in the video. Used to set the number of
            leading 0s in the keys of the dictionary. Default is 5 if the number of
            frames is not given.

    Attributes:
        filepath: The path to the shelf.
    """

    def __init__(
        self, pose_cfg: dict, filepath: str | Path, num_frames: int | None = None
    ):
        super().__init__(filepath, flag="c")
        self._pose_cfg = pose_cfg
        self._num_frames = num_frames
        self._frame_index = 0

        self._str_width = 5
        if num_frames is not None:
            self._str_width = int(np.ceil(np.log10(num_frames)))

    def add_prediction(
        self,
        bodyparts: np.ndarray,
        unique_bodyparts: np.ndarray | None = None,
        identity_scores: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Adds the prediction for a frame to the shelf

        Args:
            bodyparts: The predicted bodyparts.
            unique_bodyparts: The predicted unique bodyparts, if there are any.
            identity_scores: The predicted identities, if there are any.
        """
        if not self._open:
            raise ValueError(f"You must call open() before adding data!")

        key = "frame" + str(self._frame_index).zfill(self._str_width)

        # convert bodyparts to shape (num_bpts, num_assemblies, 3)
        bodyparts = bodyparts.transpose((1, 0, 2))
        coordinates = [bpt[:, :2] for bpt in bodyparts]
        scores = [bpt[:, 2:3] for bpt in bodyparts]

        # full pickle has bodyparts and unique bodyparts in same array
        unique_bodyparts = kwargs.get("unique_bodyparts", None)
        if unique_bodyparts is not None:
            unique_bpts = unique_bodyparts.transpose((1, 0, 2))
            coordinates += [bpt[:, :2] for bpt in unique_bpts]
            scores += [bpt[:, 2:] for bpt in unique_bpts]

        output = dict(coordinates=(coordinates,), confidence=scores, costs=None)

        identity_scores = kwargs.get("identity_scores", None)
        if identity_scores is not None:
            # Reshape id scores from (num_assemblies, num_bpts, num_individuals)
            # to the original DLC full pickle format: (num_bpts, num_assem, num_ind)
            id_scores = identity_scores.transpose((1, 0, 2))
            output["identity"] = [bpt_id_scores for bpt_id_scores in id_scores]

            if unique_bodyparts is not None:
                # needed for create_video_with_all_detections to display unique bpts
                num_unique = unique_bodyparts.shape[1]
                num_assem, num_ind = id_scores.shape[1:]
                output["identity"] += [
                    -1 * np.ones((num_assem, num_ind)) for i in range(num_unique)
                ]

        self._db[key] = output
        self._frame_index += 1

    def close(self) -> None:
        """Opens the shelf"""
        if self._open and self._frame_index > 0:
            self._db["metadata"]["nframes"] = self._frame_index

        super().close()

    def open(self) -> None:
        """Opens the shelf"""
        super().open()
        self._frame_index = 0

        all_joints = self._pose_cfg["all_joints"]
        paf_graph = self._pose_cfg.get("partaffinityfield_graph", [])

        self._db["metadata"] = {
            "nms radius": self._pose_cfg.get("nmsradius"),
            "minimal confidence": self._pose_cfg.get("minconfidence"),
            "sigma": self._pose_cfg.get("sigma", 1),
            "PAFgraph": paf_graph,
            "PAFinds": self._pose_cfg.get("paf_best", np.arange(len(paf_graph))),
            "all_joints": [[i] for i in range(len(all_joints))],
            "all_joints_names": [
                self._pose_cfg["all_joints_names"][i] for i in range(len(all_joints))
            ],
            "nframes": self._num_frames,
            "key_str_width": self._str_width,
        }


class FeatureShelfWriter(ShelfWriter):
    """Writes bodypart features to a shelf on-the-fly for ReID model training.

    Args:
        pose_cfg: The test pose config for the model.
        filepath: The path where the data should be saved.
        num_frames: The number of frames in the video. Used to set the number of
            leading 0s in the keys of the dictionary. Default is 5 if the number of
            frames is not given.

    Attributes:
        filepath: The path to the shelf.
    """

    def __init__(
        self, pose_cfg: dict, filepath: str | Path, num_frames: int | None = None
    ):
        super().__init__(pose_cfg, filepath, num_frames)

    def add_prediction(
        self,
        bodyparts: np.ndarray,
        features: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Adds the prediction for a frame to the shelf

        Args:
            bodyparts: The predicted bodyparts.
            features: The features for the bodyparts.
        """
        if not self._open:
            raise ValueError(f"You must call open() before adding data!")

        key = "frame" + str(self._frame_index).zfill(self._str_width)

        # bodyparts to shape (num_assemblies, num_bpts, xy)
        coordinates = bodyparts[:, :, :2]
        if features is None:
            raise ValueError(
                "Backbone features must be given to the FeatureShelfWriter"
            )

        self._db[key] = dict(coordinates=coordinates, features=features)
        self._frame_index += 1
