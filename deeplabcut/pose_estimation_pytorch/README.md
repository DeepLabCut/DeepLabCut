# Pytorch DLC API

##### The structure of the repo:
[Data](#data)  
[Models](#models)   
[Solvers](#solvers)   
[Apis](#apis)   

## Data
- [data](data/project.py#L7):
The `deeplabcut.pose_estimations_pytorch.data` package contains all code for pytorch dataset creation and test/train splitting.
    - `Project` class provides train and test splitting and converts dataset to required format. For intance, to [COCO]() format.

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
    > `transform` is a `List` of transformations to be applied to images and keypoints sequentialy, `None` by default.
    
    Example:
    
    ```python3
    import albumentations as A
    
    transform = A.Compose([
    A.Resize(width=256, height=256),
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    ```
    
    > **Warning**
    > By now supports only [albumentations](https://albumentations.ai), will be extended in the future.
## Models
- [models](models):
The `deeplabcut.pose_estimations_pytorch.models` package contains all related to building a model with `backbone`, `neck` (optional) and `head`.

## Solvers
- [solvers](solvers): The `deeplabcut.pose_estimations_pytorch.train_module` contains all classes for model training and validation.

## Apis
- [apis](apis): The `deeplabcut.pose_estimations_pytorch.apis` contains functionalities for training and testing as well as the corresponding configuration file [config.yaml](apis/config.yaml).

## Registry

