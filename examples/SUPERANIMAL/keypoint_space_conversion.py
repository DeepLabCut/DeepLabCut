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
"""Script to convert a dataset for its keypoint space to match the SuperAnimal space"""
from pathlib import Path

from deeplabcut.modelzoo.generalized_data_converter.datasets import COCOPoseDataset
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path


def main():
    src_proj_root = Path("/media/data/trimouse_coco_original_shuffle0")
    conversion_table_path = (
        Path(get_deeplabcut_path())
        / "modelzoo"
        / "conversion_tables"
        / "conversion_table_topview.csv"
    )
    dataset = COCOPoseDataset(str(src_proj_root), "trimouse")
    dataset.project_with_conversion_table(conversion_table_path)
    dataset.materialize(
        src_proj_root.with_name("trimouse_coco_superanimal_shuffle0_shallow_copy"),
        deepcopy=False,
        framework="coco",
    )


if __name__ == "__main__":
    main()
