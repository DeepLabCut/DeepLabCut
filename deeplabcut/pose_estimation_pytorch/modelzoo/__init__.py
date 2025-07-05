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
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    download_super_animal_snapshot,
    get_snapshot_folder_path,
    get_super_animal_model_config_path,
    get_super_animal_project_config_path,
    get_super_animal_snapshot_path,
    load_super_animal_config,
)

from deeplabcut.pose_estimation_pytorch.modelzoo.superanimal_humanbody_video_inference import (
    analyze_videos_superanimal_humanbody,
)
