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
"""Module used to shelve data during video analysis in DeepLabCut 3.0"""
import pickle
import shelve
from pathlib import Path

import numpy as np


class ShelfManager:
    """Manages shelve predictions

    Args:
        pose_cfg: The test pose config for the model.
        filepath: The path where the data should be saved.
        num_frames: The number of frames in the video. Used to set the number of
            leading 0s in the keys of the dictionary. Default is 5 if the number of
            frames is not given.

    Attributes:
        filepath: The path to the shelf.
        str_width: The number of leading 0s in frame index in the keys.
    """

    def __init__(
        self,
        pose_cfg: dict,
        filepath: str | Path,
        num_frames: int | None = None,
    ):
        self.filepath = filepath
        self._pose_cfg = pose_cfg
        self._num_frames = num_frames
        self.str_width = 5
        if num_frames is not None:
            self.str_width = int(np.ceil(np.log10(num_frames)))

        self._db = None
        self._open = False
        self._frame_index = 0

    def add_prediction(
        self,
        bodyparts: np.ndarray,
        unique_bodyparts: np.ndarray | None = None,
        identity_scores: np.ndarray | None = None,
    ) -> None:
        """Adds the prediction for a frame to the shelf

        Args:
            bodyparts: The predicted bodyparts.
            unique_bodyparts: The predicted unique bodyparts, if there are any.
            identity_scores: The predicted identities, if there are any.
        """
        if not self._open:
            raise ValueError(f"You must call open() before adding data!")

        key = "frame" + str(self._frame_index).zfill(self.str_width)

        # convert bodyparts to shape (num_bpts, num_assemblies, 3)
        bodyparts = bodyparts.transpose((1, 0, 2))
        coordinates = [bpt[:, :2] for bpt in bodyparts]
        scores = [bpt[:, 2:3] for bpt in bodyparts]

        # full pickle has bodyparts and unique bodyparts in same array
        if unique_bodyparts is not None:
            unique_bpts = unique_bodyparts.transpose((1, 0, 2))
            coordinates += [bpt[:, :2] for bpt in unique_bpts]
            scores += [bpt[:, 2:] for bpt in unique_bpts]

        output = dict(coordinates=(coordinates,), confidence=scores, costs=None)
        if identity_scores is not None:
            # Reshape id scores from (num_assemblies, num_bpts, num_individuals)
            # to the original DLC full pickle format: (num_bpts, num_assem, num_ind)
            id_scores = identity_scores.transpose((1, 0, 2))
            output["identity"] = [bpt_id_scores for bpt_id_scores in id_scores]

        self._db[key] = output
        self._frame_index += 1

    def close(self) -> None:
        """Closes the shelf"""
        if not self._open:
            return

        try:
            self._db.close()
        except AttributeError:
            pass

        self._open = False

    def open(self) -> None:
        """Opens the shelf"""
        self._db = shelve.open(self.filepath, protocol=pickle.DEFAULT_PROTOCOL)
        self._open = True
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
        }
