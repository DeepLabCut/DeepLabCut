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
"""Script to fine-tune a SuperAnimal model with memory replay"""
from pathlib import Path

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.modelzoo.utils import (
    create_conversion_table,
    read_conversion_table_from_csv,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_super_animal_snapshot_path,
)
from deeplabcut.utils.pseudo_label import keypoint_matching


def main(
    dlc_proj_root: Path,
    super_animal_name: str,
    super_animal_model: str = "hrnet_w32",
    super_animal_detector: str = "fasterrcnn_resnet50_fpn_v2",
):
    config_path = str(dlc_proj_root / "config.yaml")
    shuffle = 0
    device = "cuda"

    # keypoint matching before create training dataset
    # keypoint matching creates pseudo prediction and a conversion table
    keypoint_matching(
        config_path,
        super_animal_name,
        super_animal_model,
        super_animal_detector,
    )

    # keypoint matching creates a memory_replay folder in the root. The conversion table
    # can be read from there
    conversion_table_path = dlc_proj_root / "memory_replay" / "conversion_table.csv"

    table = create_conversion_table(
        config=config_path,
        super_animal=super_animal_name,
        project_to_super_animal=read_conversion_table_from_csv(conversion_table_path),
    )

    weight_init = WeightInitialization(
        dataset=super_animal_name,
        snapshot_path=get_super_animal_snapshot_path(
            dataset=super_animal_name,
            model_name=super_animal_model,
            download=True,
        ),
        detector_snapshot_path=get_super_animal_snapshot_path(
            dataset=super_animal_name,
            model_name=super_animal_detector,
            download=True,
        ),
        conversion_array=table.to_array(),
        with_decoder=True,
    )

    deeplabcut.create_training_dataset(
        config_path,
        Shuffles=[shuffle],
        net_type="top_down_hrnet_w32",
        weight_init=weight_init,
        engine=Engine.PYTORCH,
        userfeedback=False,
    )

    # passing pose_threshold controls the behavior of memory replay. We discard
    # predictions that are lower than the threshold
    deeplabcut.train_network(
        config_path, shuffle=shuffle, device=device, pose_threshold=0.1
    )


if __name__ == "__main__":
    main(
        dlc_proj_root=Path("/media/data/myproject"),
        super_animal_name="superanimal_topviewmouse",
    )
