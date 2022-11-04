# Pytorch DLC API
- [data](https://github.com/nastya236/DLCdev/blob/69005057eeac3c1492712863303f8268cee776e6/deeplabcut/pose_estimation_pytorch/data/project.py#L7):
The `deeplabcut.pose_estimations_pytorch.data` package contains all code for pytorch dataset creation and test/train splitting.
    - `Project` class provides train and test splitting and converts dataset to required format. For intance, to [COCO]() format.
    - `PoseTrainDataset` class is a [torch.utils.Dataset](https://pytorch.org/docs/stable/data.html) class, which converts raw images and keypoints to a tensor dataset for training and evaluation.
- [models](https://github.com/nastya236/DLCdev/blob/69005057eeac3c1492712863303f8268cee776e6/deeplabcut/pose_estimation_pytorch/data/models):
The `deeplabcut.pose_estimations_pytorch.models` package contains all related to building a model with `backbone`, `neck` (optional) and `head`.
- [train_module](https://github.com/nastya236/DLCdev/blob/69005057eeac3c1492712863303f8268cee776e6/deeplabcut/pose_estimation_pytorch/data/models): The `deeplabcut.pose_estimations_pytorch.train_module` contains all classes for model training and validation.

