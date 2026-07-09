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
)
from deeplabcut.pose_estimation_pytorch.data.ctd import (
    CondFromFile,
    CondFromModel,
)
from deeplabcut.pose_estimation_pytorch.task import Task


def get_condition_provider(
    condition_cfg: ConditionsModelConfig | dict,
    config: str | Path | None = None,
) -> CondFromModel:
    """Creates a CondFromModel conditions provider for a CTD model.

    Args:
        condition_cfg: The conditions configuration. Either a
            ``ConditionsModelConfig`` instance, or a raw dict (e.g. read from
            ``model_cfg["inference"]["conditions"]``).
        config: Path to the DLC project ``config.yaml``. Injected into the
            config when the shuffle shorthand is used and ``config`` is not
            already set on ``condition_cfg``.

    Returns:
        The CondFromModel provider that can be used to generate conditions from a BU
        model for a CTD model.
    """
    # Normalize the conditions configuration
    condition_cfg = ConditionsConfig.build(condition_cfg)

    if isinstance(condition_cfg, ConditionsFileConfig):
        raise ValueError(
            "To run inference with CTD models, you must specify the BU model "
            "you want to use to generate conditions.\n" + _CONDITION_EXAMPLES_INFERENCE
        )

    # Inject the project config for the shuffle shorthand if not already set.
    if config is not None and condition_cfg.config is None:
        condition_cfg.config = config

    kwargs = condition_cfg.to_dict()
    kwargs.pop("source")
    return CondFromModel(**kwargs)


def get_conditions_provider_for_video(
    cond_provider: CondFromModel,
    video: str | Path,
) -> CondFromFile | None:
    """Tries to create a conditions loader.

    Args:
        cond_provider: The CondFromModel condition provider that will be used. The
            scorer must be set, or potential conditions files for the video cannot be
            found.
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
        f"Misconfigured conditions in the pytorch_config: {condition_cfg}. Valid "
        f"examples:\n" + _CONDITION_EXAMPLES_INFERENCE + _CONDITION_EXAMPLES_FROM_FILE
    )

    if isinstance(condition_cfg, (str, Path)):
        condition_filepath = Path(condition_cfg)
        cond_provider = CondFromFile(filepath=condition_filepath)
    # TODO @deruyter92: decide on typed / plain dict
    elif isinstance(condition_cfg, dict):
        if isinstance(loader, data.DLCLoader) and "config" not in condition_cfg:
            condition_cfg["config"] = loader.project_root / "config.yaml"

        cond_provider = CondFromFile(**condition_cfg)
    else:
        raise ValueError(error_message)

    return cond_provider.load_conditions(images, path_prefix=loader.image_root)


_CONDITION_EXAMPLES_INFERENCE = """
Example: Using a bottom-up model for conditions
  ```
  inference:
    conditions:
      config_path: /path/to/model-dir/pytorch_config.yaml
      snapshot_path: /path/to/model-dir/snapshot-best-150.pth
  ```
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
