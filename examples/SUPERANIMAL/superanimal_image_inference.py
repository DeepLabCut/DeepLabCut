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
from deeplabcut.pose_estimation_pytorch.apis.analyze_images import (
    superanimal_analyze_images,
)


if __name__ == "__main__":
    superanimal_name = "superanimal_quadruped"
    model_name = "hrnetw32"
    device = "cuda"
    max_individuals = 3

    ret = superanimal_analyze_images(
        superanimal_name,
        model_name,
        "test_rodent_images",
        max_individuals,
        "vis_test_rodent_images",
    )
