# PyTorch DeepLabCut API

- **Structure of the repo:**

Like any ML model this repo contains: models (architectures), solvers (losses and optimizers), and data (data loaders).

[Models](#models)   
[Solvers](#solvers)   
[Data](#data)  
[APIs](#apis)   

## Models

- [models](models):
The `deeplabcut.pose_estimations_pytorch.models` package contains all components related to building a model with `backbone`, `neck` (optional) and `head`.

We provide sota models such as HRNet, BUCTD, TransPose, ...

If you want to add a novel model, you have to divide it into backbone, neck and head. Often neck will be just the identity function.

For instance, a [standard pose estimation HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation) consists of HRNet backbone, an identity neck and a deconvolution head (Simple Head).




## Solvers

- [solvers](solvers): The `deeplabcut.pose_estimations_pytorch.train_module` contains all classes for model training and validation.

## Data

- [data](data/project.py#L7):
The `deeplabcut.pose_estimations_pytorch.data` package contains all code for pytorch dataset creation and test/train splitting.
    - `Project` class provides train and test splitting and converts dataset to required format. For instance, to [COCO]() format.

    Example:

    ```python3
    import deeplabcut.pose_estimation_pytorch as dlc

    project = dlc.Project(proj_root=config['project_root'])
    project.train_test_split()
    ```
    - `PoseDataset` class is an instance of [torch.utils.Dataset](https://pytorch.org/docs/stable/data.html), which converts raw images and keypoints to a tensor dataset for training and evaluation.

    Example:

    ```python3
    transform = None
    train_dataset = dlc.PoseDataset(project,
                                    transform=transform,
                                    mode='train')
    valid_dataset = dlc.PoseDataset(project,
                                    transform=transform,
                                    mode='test')
    ```

    > **Note**
    > `transform` is a `List` of transformations to be applied to images and keypoints sequentially, `None` by default.

    Example:

    ```python3
    import albumentations as A

    transform = A.Compose([
    A.Resize(width=256, height=256),
    ], keypoint_params=A.KeypointParams(format='xy'))

    ```

    > **Warning**
    > By now supports only [albumentations](https://albumentations.ai), will be extended in the future.


## Apis

- [apis](apis): The `deeplabcut.pose_estimations_pytorch.apis` contains functionalities for training and testing as well as the corresponding configuration file [config.yaml](apis/config.yaml).

## Registry
