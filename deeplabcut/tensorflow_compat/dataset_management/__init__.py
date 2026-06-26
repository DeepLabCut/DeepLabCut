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
"""
TensorFlow dataset-management API. Remove this module when TF support is dropped.
"""

from deeplabcut.tensorflow_compat.dataset_management.create_from_existing_split import (
    create_training_dataset_from_existing_split,
)
from deeplabcut.tensorflow_compat.dataset_management.create_multi_animal import (
    create_multianimaltraining_dataset,
)
from deeplabcut.tensorflow_compat.dataset_management.create_single_animal import (
    create_training_dataset,
)
from deeplabcut.tensorflow_compat.dataset_management.model_comparison import (
    create_training_model_comparison,
)
from deeplabcut.tensorflow_compat.dataset_management.pose_yaml import (
    MakeInference_yaml,
    MakeTest_pose_yaml,
    MakeTrain_pose_yaml,
    ParseYaml,
)

__all__ = [
    "MakeInference_yaml",
    "MakeTest_pose_yaml",
    "MakeTrain_pose_yaml",
    "ParseYaml",
    "create_multianimaltraining_dataset",
    "create_training_dataset",
    "create_training_dataset_from_existing_split",
    "create_training_model_comparison",
]
