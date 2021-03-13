"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
from .factory import PoseDatasetFactory
from .pose_deterministic import DeterministicPoseDataset
from .pose_scalecrop import ScalecropPoseDataset
from .pose_imgaug import ImgaugPoseDataset
from .pose_tensorpack import TensorpackPoseDataset
from .pose_multianimal_imgaug import MAImgaugPoseDataset
from .utils import Batch


__all__ = [
    "PoseDatasetFactory",
    "DeterministicPoseDataset",
    "ScalecropPoseDataset",
    "ImgaugPoseDataset",
    "TensorpackPoseDataset",
    "MAImgaugPoseDataset",
    "Batch",
]
