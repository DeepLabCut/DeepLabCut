#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    TARGET_GENERATORS,
    BaseGenerator,
    SequentialGenerator,
)
from deeplabcut.pose_estimation_pytorch.models.target_generators.dekr_targets import (
    DEKRGenerator,
)
from deeplabcut.pose_estimation_pytorch.models.target_generators.gaussian_targets import (
    GaussianGenerator,
)
from deeplabcut.pose_estimation_pytorch.models.target_generators.pafs_targets import (
    PartAffinityFieldGenerator,
)
from deeplabcut.pose_estimation_pytorch.models.target_generators.plateau_targets import (
    PlateauGenerator,
)
