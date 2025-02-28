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
"""Functions to build weight initialization parameters for SuperAnimal models"""
from pathlib import Path

import deeplabcut.modelzoo.utils as utils
from deeplabcut.core.config import read_config_as_dict
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_super_animal_snapshot_path
)


def build_weight_init(
    cfg: dict | str | Path,
    super_animal: str,
    model_name: str,
    detector_name: str | None,
    with_decoder: bool = False,
    memory_replay: bool = False,
    customized_pose_checkpoint: str | Path | None = None,
    customized_detector_checkpoint: str | Path | None = None,
) -> WeightInitialization:
    """Builds the WeightInitialization from a SuperAnimal model for a project

    Args:
        cfg: The project's configuration, or the path to the project configuration file.
        super_animal: The SuperAnimal model with which to initialize weights.
        model_name: The type of the model architecture for which to load the weights.
        detector_name: The type of detector architecture for which to load the weights.
        with_decoder: Whether to load the decoder weights as well. If this is true,
            a conversion table must be specified for the given SuperAnimal in the
            project configuration file. See
            ``deeplabcut.modelzoo.utils.create_conversion_table`` to create a
            conversion table.
        memory_replay: Only when ``with_decoder=True``. Whether to train the model
            with memory replay, so that it predicts all SuperAnimal bodyparts.
        customized_pose_checkpoint: A customized SuperAnimal pose checkpoint, as an
            alternative to the Hugging Face one
        customized_detector_checkpoint: A customized SuperAnimal detector checkpoint, as
            an alternative to the Hugging Face one

    To build a WeightInitialization instance for a project using the conversion table
    specified in the project configuration file, use:

        ```
        from pathlib import Path
        from deeplabcut.utils.auxiliaryfunctions import read_config
        from deeplabcut.modelzoo import build_weight_init

        project_cfg = read_config("/path/to/my/project/config.yaml")
        super_animal = "superanimal_quadruped"
        weight_init = build_weight_init(
            cfg=project_cfg,
            super_animal="superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            with_decoder=True,
            memory_replay=False,
        )
        ```

    Returns:
        The built WeightInitialization.
    """
    if isinstance(cfg, (str, Path)):
        cfg = read_config_as_dict(cfg)

    conversion_array = None
    bodyparts = None
    if with_decoder:
        conversion_table = utils.get_conversion_table(cfg, super_animal)
        conversion_array = conversion_table.to_array()
        bodyparts = conversion_table.converted_bodyparts()

    snapshot_path = customized_pose_checkpoint
    if snapshot_path is None:
        snapshot_path = get_super_animal_snapshot_path(
            dataset=super_animal,
            model_name=model_name,
            download=True,
        )

    detector_snapshot_path = customized_detector_checkpoint
    if detector_snapshot_path is None and detector_name is not None:
        detector_snapshot_path = get_super_animal_snapshot_path(
            dataset=super_animal,
            model_name=detector_name,
            download=True,
        )

    return WeightInitialization(
        snapshot_path=snapshot_path,
        detector_snapshot_path=detector_snapshot_path,
        dataset=super_animal,
        with_decoder=with_decoder,
        memory_replay=memory_replay,
        conversion_array=conversion_array,
        bodyparts=bodyparts,
    )
