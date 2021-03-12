"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
from .pose_deterministic import DeterministicPoseDataset
from .pose_scalecrop import ScalecropPoseDataset
from .pose_imgaug import ImgaugPoseDataset
from .pose_tensorpack import TensorpackPoseDataset
from .pose_multianimal_imgaug import MAImgaugPoseDataset

__all__ = [
    "DeterministicPoseDataset",
    "ScalecropPoseDataset",
    "ImgaugPoseDataset",
    "TensorpackPoseDataset",
    "MAImgaugPoseDataset",
]
