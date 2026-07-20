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
"""Methods to help with conditional top-down models."""

from pathlib import Path

import numpy as np

import deeplabcut.pose_estimation_pytorch.data as data
from deeplabcut.pose_estimation_pytorch.config.ctd_conditions import (
    ConditionsConfig,
    ConditionsFileConfig,
    ConditionsModelConfig,
    ConditionsShuffleConfig,
)
from deeplabcut.pose_estimation_pytorch.data.ctd import CondFromFile
from deeplabcut.pose_estimation_pytorch.task import Task


def get_conditions_provider_for_video(
    cond_provider: ConditionsModelConfig,
    video: str | Path,
) -> CondFromFile | None:
    """Tries to create a conditions loader from a pre-computed video predictions file.

    Args:
        cond_provider: The resolved ``ConditionsModelConfig``. The scorer must be
            set, or pre-computed conditions files for the video cannot be found.
        video: The path to the video file for which to look for the conditions.

    Returns:
        None if no condition files for this BU model and video can be found.
        The CondFromFile provider to load the conditions for the video from a file.
    """
    if cond_provider.scorer is None:
        return None

    video = Path(video)

    # Load pickle for multi-animal projects
    cond_file = video.parent / f"{video.stem}{cond_provider.scorer}_assemblies.pickle"
    if not cond_file.exists():
        # Load h5 for single-animal projects
        cond_file = video.parent / f"{video.stem}{cond_provider.scorer}.h5"
        if not cond_file.exists():
            return None

    return CondFromFile(filepath=cond_file)


def load_conditions_for_evaluation(loader: data.Loader, images: list[str]) -> dict[str, np.ndarray]:
    """Loads the conditions needed to evaluate a CTD model.

    Evaluation loads pre-computed BU predictions from disk via ``CondFromFile``.
    Supported ``inference.conditions`` forms:

    - ``ConditionsFileConfig`` / path string — load that predictions file
    - ``ConditionsShuffleConfig`` / shuffle dict — resolve the BU shuffle's evaluation
      ``.h5`` (project ``config.yaml`` is injected from the loader when missing)

    ``ConditionsModelConfig`` is not valid here (that form is for live BU inference).
    File-path conditions are evaluation-only and cannot be used with ``analyze_*``.

    Args:
        loader: The Loader for the CTD model to evaluate.
        images: A list of image paths to load conditions for.

    Returns:
        The conditions for the images.
    """
    if loader.pose_task != Task.COND_TOP_DOWN:
        raise ValueError("Conditions can only be loaded for CTD models")

    # load the conditions config
    condition_cfg = loader.model_cfg["inference"].get("conditions")

    # prepare error message
    error_message = (
        f"Misconfigured conditions in the pytorch_config: {condition_cfg}. "
        f"Evaluation accepts file paths or shuffle refs (not a Model config). "
        f"Valid examples:\n" + _CONDITION_EXAMPLES_SHUFFLE + _CONDITION_EXAMPLES_FROM_FILE
    )

    try:
        conditions = ConditionsConfig.build(condition_cfg)
    except (TypeError, ValueError) as e:
        raise ValueError(error_message) from e

    if conditions is None:
        raise ValueError(
            "CTD evaluation requires conditions in the pytorch_config "
            "(`inference.conditions`). Got None.\n" + error_message
        )

    if isinstance(conditions, ConditionsFileConfig):
        cond_provider = CondFromFile(filepath=conditions.filepath)
    elif isinstance(conditions, ConditionsShuffleConfig):
        config = conditions.config
        if config is None and isinstance(loader, data.DLCLoader):
            config = loader.project_root / "config.yaml"
        if config is None:
            raise ValueError(
                "Cannot load shuffle conditions for evaluation: no project config "
                "available. Set 'config' in the shuffle conditions.\n" + error_message
            )
        cond_provider = CondFromFile(
            config=config,
            shuffle=conditions.shuffle,
            trainset_index=conditions.trainset_index,
            modelprefix=conditions.modelprefix,
            snapshot=conditions.snapshot,
            snapshot_index=conditions.snapshot_index,
        )
    else:
        raise ValueError(error_message)

    return cond_provider.load_conditions(images, path_prefix=loader.image_root)


_CONDITION_EXAMPLES_SHUFFLE = """
Example: Loading the predictions for snapshot-250.pt of shuffle 1.
  ```
  inference:
    conditions:
      shuffle: 1
      snapshot: snapshot-250.pt
  ```
Example: Loading the predictions for the snapshot with index 2 of shuffle 1.
  ```
  inference:
    conditions:
      shuffle: 1
      snapshot_index: 2
  ```
"""


_CONDITION_EXAMPLES_FROM_FILE = """
Example: Loading the predictions contained in an h5 file.
  ```
  inference:
    conditions: /path/to/bu_predictions.h5
  ```
Example: Loading the predictions contained in an json file.
  ```
  inference:
    conditions: /path/to/bu_predictions.json
  ```
"""
