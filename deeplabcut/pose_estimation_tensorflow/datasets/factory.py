"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

import warnings


class PoseDatasetFactory:
    _datasets = dict()

    @classmethod
    def register(cls, type_):
        def wrapper(dataset):
            if type_ in cls._datasets:
                warnings.warn("Overwriting existing dataset {}.")
            cls._datasets[type_] = dataset
            return dataset

        return wrapper

    @classmethod
    def create(cls, cfg):
        dataset_type = cfg["dataset_type"]
        dataset = cls._datasets.get(dataset_type)
        if dataset is None:
            raise ValueError(f"Unsupported dataset of type {dataset_type}")
        return dataset(cfg)
