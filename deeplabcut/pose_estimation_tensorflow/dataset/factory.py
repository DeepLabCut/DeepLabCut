"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

Updated to allow more data set loaders.
"""


def create(cfg):
    dataset_type = cfg["dataset_type"]
    if dataset_type == "scalecrop":
        print("Starting with scalecrop pose-dataset loader.")
        from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset_scalecrop import (
            PoseDataset,
        )

        data = PoseDataset(cfg)
    elif dataset_type == "deterministic":
        print("Starting with deterministic pose-dataset loader.")
        from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset_deterministic import (
            PoseDataset,
        )

        data = PoseDataset(cfg)

    elif dataset_type == "tensorpack":
        print("Starting with tensorpack pose-dataset loader.")
        from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset_tensorpack import (
            PoseDataset,
        )

        data = PoseDataset(cfg)

    elif dataset_type == "imgaug" or dataset_type == "default":
        print("Starting with imgaug pose-dataset loader (=default).")
        from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset_imgaug import (
            PoseDataset,
        )

        data = PoseDataset(cfg)

    elif dataset_type == "multi-animal-imgaug":
        print("Starting with multi-animal imgaug + adam pose-dataset loader.")
        from deeplabcut.pose_estimation_tensorflow.dataset.pose_multianimal_imgaug import (
            MAPoseDataset,
        )

        data = MAPoseDataset(cfg)
    else:
        raise Exception('Unsupported dataset_type: "{}"'.format(dataset_type))

    return data
