"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""


from deeplabcut.pose_estimation_tensorflow.datasets import *


class PoseDatasetFactory:
    def __init__(self):
        self._datasets = dict()

    def register_dataset(self, type_, dataset):
        self._datasets[type_] = dataset

    def build_dataset(self, cfg):
        dataset_type = cfg["dataset_type"]
        dataset = self._datasets.get(dataset_type)
        if dataset is None:
            raise ValueError(f"Unsupported datasets of type {dataset_type}")
        return dataset(cfg)


pose_factory = PoseDatasetFactory()
pose_factory.register_dataset("default", ImgaugPoseDataset)
pose_factory.register_dataset("imgaug", ImgaugPoseDataset)
pose_factory.register_dataset("scalecrop", ScalecropPoseDataset)
pose_factory.register_dataset("deterministic", DeterministicPoseDataset)
pose_factory.register_dataset("tensorpack", TensorpackPoseDataset)
pose_factory.register_dataset("multi-animal-imgaug", MAImgaugPoseDataset)
