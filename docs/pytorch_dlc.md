# DeepLabCut: PyTorch API

## Modules

- [data](https://github.com/nastya236/DLCdev/blob/69005057eeac3c1492712863303f8268cee776e6/deeplabcut/pose_estimation_pytorch/data/project.py#L7):
The `deeplabcut.pose_estimations_pytorch.data` package contains all code for pytorch 
dataset creation and test/train splitting.
  - `Project` class provides train and test splitting and converts dataset to required
  format. For instance, to [COCO]() format.
  - `PoseTrainDataset` class is a [torch.utils.Dataset](https://pytorch.org/docs/stable/data.html) class, which converts raw 
  images and keypoints to a tensor dataset for training and evaluation.
- [models](https://github.com/nastya236/DLCdev/blob/69005057eeac3c1492712863303f8268cee776e6/deeplabcut/pose_estimation_pytorch/data/models):
The `deeplabcut.pose_estimations_pytorch.models` package contains all related to 
building a model with `backbone`, `neck` (optional) and `head`.
- [train_module](https://github.com/nastya236/DLCdev/blob/69005057eeac3c1492712863303f8268cee776e6/deeplabcut/pose_estimation_pytorch/data/models):
The `deeplabcut.pose_estimations_pytorch.train_module` contains all classes for model 
training and validation.

## API

The PyTorch implementation of DeepLabCut is very similar to the Tensorflow multi-animal
implementation: the same steps need to be followed, just with slightly different API 
calls (and different model names).

Up until it's time to create the training dataset, there are no changes to the way a
PyTorch or Tensorflow project should be created.

### Creating a Training Dataset

To create a training dataset for a DeepLabCut PyTorch model, simply call:
```python
import deeplabcut
deeplabcut.create_training_dataset(
    path_config_file,
    net_type="dekr_32",
)
```

This will create folders for the training dataset in the same way as the Tensorflow
version, with an addition configuration file in the `train` folder: 
`pytorch_config.yaml`. This is the file that can be edited to modify the model 
architecture or training parameters.

There are currently two "families" of models implemented in PyTorch: DEKR (Geng, Zigang,
et al. "Bottom-up human pose estimation via disentangled keypoint regression." 
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 
2021.) and Tokenpose (Li, Yanjie, et al. "Tokenpose: Learning keypoint tokens for human
pose estimation." Proceedings of the IEEE/CVF International conference on computer 
vision. 2021.). The choices of `net_type` that will create PyTorch training sets are:
- `"dekr_16"`
- `"dekr_32"`
- `"dekr_48"`
- `"token_pose_w16"`
- `"token_pose_w32"`
- `"token_pose_w48"`

Note that Tokenpose models cannot currently be used with projects that contain unique 
keypoints. 

### Training the network
Training a PyTorch model is done in a very similar manner as a tensorflow model, though 
currently the PyTorch API needs to be called directly:
```python
import deeplabcut.pose_estimation_pytorch.apis as api
api.train_network(config_path, shuffle=1, trainingsetindex=0)
```

**Parameters**
```
config : path to the yaml config file of the project
shuffle : index of the shuffle we want to train on
trainingsetindex : training set index
transform: Augmentation pipeline for the images
    if None, the augmentation pipeline is built from config files
    Advice if you want to use custom transformations:
        Keep in mind that in order for transfer learning to be efficient, your
        data statistical distribution should resemble the one used to pretrain your backbone
        In most cases (e.g backbone was pretrained on ImageNet), that means it should be Normalized with
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
transform_cropped: Augmentation pipeline for the cropped images around animals
    if None, the augmentation pipeline is built from config files
    Advice if you want to use custom transformations:
        Keep in mind that in order for transfer learning to be efficient, your
        data statistical distribution should resemble the one used to pretrain your backbone
        In most cases (e.g backbone was pretrained on ImageNet), that means it should be Normalized with
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
modelprefix: directory containing the deeplabcut configuration files to use
    to train the network (and where snapshots will be saved). By default, they
     are assumed to exist in the project folder.
snapshot_path: if resuming training, used to specify the snapshot from which to resume
detector_path: if resuming training of a top down model, used to specify the detector snapshot from
    which to resume
**kwargs : could be any entry of the pytorch_config dictionary. Examples are
    to see the full list see the pytorch_cfg.yaml file in your project folder
```

### Evaluating the network
As for training, the main difference is the need to call the API directly.
```python
import deeplabcut.pose_estimation_pytorch.apis as api
api.evaluate_network(config_path, shuffle=1, trainingsetindex="all")
```

**Parameters**
```
config: path to the project's config file
shuffles: Iterable of integers specifying the shuffle indices to evaluate.
trainingsetindex: Integer specifying which training set fraction to use.
    Evaluates all fractions if set to "all"
snapshotindex: index (starting at 0) of the snapshot we want to load. To
    evaluate the last one, use -1. To evaluate all snapshots, use "all". For
    example if we have 3 models saved
        - snapshot-0.pt
        - snapshot-50.pt
        - snapshot-100.pt
    and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
    the snapshotindex is loaded from the project configuration.
plotting: Plots the predictions on the train and test images. If provided it must
    be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``. Setting
    to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
show_errors: display train and test errors.
transform: transformation pipeline for evaluation
    ** Should normalise the data the same way it was normalised during training **
modelprefix: directory containing the deeplabcut models to use when evaluating
    the network. By default, they are assumed to exist in the project folder.
batch_size: the batch size to use for evaluation
```

### Analyzing novel videos
One big difference between the PyTorch and Tensorflow implementations comes in the way
animal assembly happens (for multi-animal models). While in Tensorflow, assembly was a
separate step that needed to be done from the keypoints, in the PyTorch version it's 
integrated directly into the models. From an API standpoint, that does not change much.

Again, the PyTorch API needs to be invoked directly (it also has the `auto_track` 
option).
```python
import deeplabcut.pose_estimation_pytorch.apis as api
api.analyze_videos(config_path, ["/fullpath/project/videos/test.mp4"], videotype=".mp4")
```

The PyTorch detections need to be converted to tracklets using the PyTorch API, but then
the original tracklet stitching can be used. 
```python
import deeplabcut
import deeplabcut.pose_estimation_pytorch.apis as api
api.convert_detections2tracklets(
    config_path,
    videos=['/fullpath/project/videos/test.mp4'],
    videotype=".mp4",
)
deeplabcut.stitch_tracklets(
    config_path,
    videos=['/fullpath/project/videos/test.mp4'],
    videotype=".mp4",
)
```

Creating labeled videos can then be called in exactly the same way as before. 
```python
import deeplabcut
deeplabcut.create_labeled_video(
    config_path,
    videos=['/fullpath/project/videos/test.mp4'],
    videotype=".mp4",
)
```
