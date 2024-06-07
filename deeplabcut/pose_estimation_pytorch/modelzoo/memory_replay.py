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

from pathlib import Path

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.modelzoo.generalized_data_converter.datasets import (
    COCOPoseDataset,
    MaDLCPoseDataset,
    MultiSourceDataset,
    SingleDLCPoseDataset,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_config_model_paths,
    update_config,
)


def prepare_memory_replay(
    dlc_proj_root: str | Path,
    shuffle: int,
    superanimal_name: str,
    model_name: str,
    device: str,
    max_individuals=3,
    trainingsetindex: int = 0,
):
    """TODO: Documentation"""
    (
        superanimal_model_config,
        project_config,
        pose_model_path,
        detector_path,
    ) = get_config_model_paths(superanimal_name, model_name)

    # in order to fill the num_bodyparts stuff

    if "individuals" in cfg:
        temp_dataset = MaDLCPoseDataset(str(dlc_proj_root), "temp_dataset")
    else:
        temp_dataset = SingleDLCPoseDataset(str(dlc_proj_root), "temp_dataset")

    superanimal_model_config = {**project_config, **superanimal_model_config}
    superanimal_model_config = update_config(
        superanimal_model_config, max_individuals, device
    )

    dlc_proj_root = Path(dlc_proj_root)
    config_path = dlc_proj_root / "config.yaml"

    cfg = af.read_config(config_path)

    trainIndex = 0

    model_folder = dlc_proj_root / af.get_model_folder(
        cfg["TrainingFraction"][trainIndex], shuffle, cfg, engine=Engine.PYTORCH
    )

    memory_replay_folder = model_folder / "memory_replay"

    temp_dataset.materialize(memory_replay_folder, framework="coco")

    original_model_config = af.read_config(
        str(model_folder / "train" / "pytorch_config.yaml")
    )

    weight_init_cfg = original_model_config["train_settings"].get("weight_init")
    if weight_init_cfg is None:
        raise ValueError(
            "You can only train models with memory replay when you are fine-tuning a "
            "SuperAnimal model. Please look at the documentation to see how to create "
            "a training dataset to fine-tune one of the SuperAnimal models."
        )

    weight_init = WeightInitialization.from_dict(weight_init_cfg)
    if not weight_init.with_decoder:
        raise ValueError(
            "You can only train models with memory replay when you are fine-tuning a "
            "SuperAnimal model. Please look at the documentation to see how to create "
            "a training dataset to fine-tune one of the SuperAnimal models. Ensure "
            "that a conversion table is specified for your project and that you select"
            "``with_decoder=True`` for your ``WeightInitialization``."
        )

    # keypoint matching removed here

    dataset = COCOPoseDataset(memory_replay_folder, "memory_replay_dataset")

    conversion_table_path = dlc_proj_root / "memory_replay" / "conversion_table.csv"

    dataset.project_with_conversion_table(str(conversion_table_path))
    dataset.materialize(memory_replay_folder, deepcopy=False, framework="coco")

    """
    # but we can copy all training related parameters from the original model config
    pose_epochs = original_model_config["train_settings"].get("epochs", 10)
    save_epochs = original_model_config["runner"]["snapshots"].get("save_epochs", 10)
    batch_size = original_model_config["train_settings"].get("batch_size", 16)

    detector_cfg = original_model_config.get("detector", {})
    detector_train_settings = detector_cfg.get("train_settings", {})
    detector_epochs = detector_train_settings.get("epochs", 1)
    detector_batch_size = detector_train_settings.get("batch_size", 4)
    detector_save_epochs = (
        detector_cfg.get("runner", {}).get("snapshots", {}).get("save_epochs", 1)
    )
    """
