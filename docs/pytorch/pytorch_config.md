(dlc3-pytorch-config)=
# The PyTorch Configuration file

The `pytorch_config.yaml` file specifies the configuration for your PyTorch pose models,
from the model architecture to which optimizer will be used for training, how training 
runs will be logged, the data augmentation that will be applied and which metric should
be used to save the "best" model snapshot. 

You can create default configurations for a shuffle using 
`deeplabcut.create_training_set` or `deeplabcut.create_training_model_comparison`. This 
will create a `pytorch_config.yaml` file for your selected net type. The basic structure
of the file is as follows:

```yaml
data:  # which data augmentations will be used
  ...
device: auto # the default device to use for training and evaluation
metadata:  # metadata regarding the project (bodyparts, individuals, paths, ...) - filled automatically
  ...
method: bu # indicates how pose predictions are made (bottom-up (`bu`) or top-down (`td`))
model:  # configures the model architecture (which backbone, heads, ...)
  ...
net_type: resnet_50 # the type of neural net configured in the file
runner:  # configuring the runner used for training
  ...
train_settings:  # generic training settings, such as batch size and maximum number of epochs
  ...
logger:  # optional: the configuration for a logger if you want one
```

## Sections

### Singleton Parameters

There are a few singleton parameters defined in the PyTorch configuration file:

- `device`: The device to use for training/inference. The default is `auto`, which sets 
the device to `cuda` if an NVIDIA GPU is available, and `cpu` otherwise. For users 
running models on macOS with an M1/M2/M3 chip, this is set to `mps` for certain models
(not all operations are currently supported on Apple GPUs - so some models like HRNets 
need to be trained on CPU, while others like ResNets can take advantage of the GPU).
- `method`: Either `bu` for bottom-up models, or `td` for top-down models.
- `net_type`: The type of pose model configured by the file (e.g. `resnet_50`).

### Data

The data section configures:

- `colormode`: in which format images are given to the model (e.g., `RGB`, `BGR`)
- `inference`: which transformations should be applied to images when running evaluation
or inference
- `train`: which transformations should be applied to images when training

The default configuration for a pose model is:

```yaml
data:
  colormode: RGB  # should never be changed
  inference:  # the augmentations to apply to images during inference 
    normalize_images: true  # this should always be set to true
  train:
    affine:
      p: 0.9
      rotation: 30
      translation: 40
    collate:  # rescales the images when putting them in a batch
      type: ResizeFromDataSizeCollate
      min_scale: 0.4
      max_scale: 1.0
      min_short_side: 128
      max_short_side: 1152
      multiple_of: 32
      to_square: false
    covering: true
    gaussian_noise: 12.75
    hist_eq: true
    motion_blur: true
    normalize_images: true  # this should always be set to true
```

One of the most important elements is the `collate` configuration. If all images in your
dataset have the same size, then it doesn't necessarily need to be added (but might 
still be beneficial). But if you have images of different sizes, then you'll need to 
define a way of "combining" these images into a single tensor of shape `(B, 3, H, W)`.
The default way to do this is to use the `ResizeFromDataSizeCollate` collate function 
(other collate functions are defined in 
`deeplabcut/pose_estimation_pytorch/data/collate.py`). For each batch to collate, this
implementation:
1. Selects the target width & height all images will be resized to by getting the size 
of the first image in the batch, and multiplying it by a scale sampled uniformly at 
random from `(min_scale, max_scale)`.
2. Resizes all images in the batch (while preserving their aspect ratio) such that they 
are the smallest size such that the target size fits entirely in the image.
3. Crops each resulting image into the target size with a random crop.

**Collate**: Defines how images are collated into batches.

```yaml
collate:  # rescales the images when putting them in a batch
  type: ResizeFromDataSizeCollate  # You can also use `ResizeFromListCollate`
  max_shift: 10  # the maximum shift, in pixels, to add to the random crop (this means
    # there can be a slight border around the image)
  max_size: 1024  #  the maximum size of the long edge of the image when resized. If the
    # longest side will be greater than this value, resizes such that the longest side 
    # is this size, and the shortest side is smaller than the desired size. This is 
    # useful to keep some information from images with extreme aspect ratios.
  min_scale: 0.4  # the minimum scale to resize the image with
  max_scale: 1.0  # the maximum scale to resize the image with
  min_short_side: 128  # the minimum size of the target short side
  max_short_side: 1152  # the maximum size of the target short side
  multiple_of: 32  # pads the target height, width such that they are multiples of 32
  to_square: false  # instead of using the aspect ratio of the first image, only the 
    # short side of the first image will be used to sample a "side", and the images will
    # be cropped in squares
```

The following transformations are available for the `train` and `inference` keys.

**Affine**: Applies an affine (rotation, translation, scaling) transformation to the
images. 

```yaml
affine:
  p: 0.9  # float: the probability that an affine transform is applied
  rotation: 30  # int: the maximum angle of rotation applied to the image (in degrees)
  scaling: [ 0.5, 1.25 ]  # [float, float]: the (min, max) scale to use to resize images
  translation: 40  # int: the maximum translation to apply to images (in pixels)
```

**Auto-Padding**: Pads the image to some desired shape (e.g., a minimum height/width or 
such that the height/width are divisible by a given number). Some backbones (such as
HRNets) require the height and width of images to be multiples of 32. Setting up
auto-padding with `pad_height_divisor: 32` and `pad_width_divisor: 32` ensures that is
the case. Note that **not all keys need to be set**! The values shown are the default
values. Only one of 'min_height' and 'pad_height_divisor' parameters must be set, and 
only one of 'min_width' and 'pad_width_divisor' parameters must be set.

```yaml
auto_padding:
  min_height: null  # int: if not None, the minimum height of the image
  min_width: null  # int: if not None, the minimum width of the image
  pad_height_divisor: null  # int: if not None, ensures image height is dividable by value of this argument.
  pad_width_divisor: null  # int: if not None, ensures image width is dividable by value of this argument.
  position: random  # str: position of the image, one of 'A.PadIfNeeded.Position'
  border_mode: reflect_101  # str: 'constant' or 'reflect_101' (see cv2.BORDER modes)
  border_value: null  # str: padding value if border_mode is 'constant'
  border_mask_value: null  # str: padding value for mask if border_mode is 'constant'
```

**Covering**: Based on Albumentations's [CoarseDropout](
https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#albumentations.augmentations.dropout.coarse_dropout)
augmentation, this "cuts" holes out of the image. As defined in 
[Improved Regularization of Convolutional Neural Networks with Cutout](
https://arxiv.org/abs/1708.04552).

```yaml
covering: true  # bool: if true, applies a coarse dropout with probability 50%
```

**Gaussian Noise**: Applies gaussian noise to the input image. Can either be a float 
(the standard deviation of the noise) or simply a boolean (the standard deviation of 
the noise will be set as 12.75).

```yaml
gaussian_noise: 12.75  # bool, float: add gaussian noise
```


**Horizontal Flips**: This flips the image horizontally around the y-axis. As the 
resulting image is mirrored, it does not preserve labels (the left hand would become the
right hand, and vice-versa). This augmentation should not be used for pose models if you
have symmetric keypoints! However, it is safe to use it to train detectors.

```yaml
# if float > 0, the probability of applying a horizontal flip
# if true, applies a horizontal flip with probability 0.5
hflip: true  # bool, float
```

**Histogram Equalization**: Applies histogram equalization with probability 50%.

```yaml
hist_eq: true  # bool: whether to apply histogram equalization
```

**Motion Blur**: Applies motion blur to the image with probability 50%.

```yaml
motion_blur: true  # bool: whether to apply motion blur
```

**Normalization**

```yaml
normalize_images: true  # normalizes images
```

**Resizing**: Resizes the images while preserving the aspect ratio (first resizes to the
maximum possible size, then adds padding for the missing pixels).

```yaml
resize:
  height: 640 # int: the height to which all images will be resized
  width: 480 # int: the width to which all images will be resized
  keep_ratio: true  # bool: the 
```

**Resizing - Crop Sampling**: An alternative way to ensure all images have the same size
is through cropping. The `crop_sampling` crops images down to a maximum width and 
height, with options to sample the center of the crop according to the positions of the
keypoints.

```yaml
crop_sampling:
  height: 400  # int: the height of the crop 
  width: 400  # int: the height of the crop 
  max_shift: 0.4  # float: maximum allowed shift of the cropping center position as a fraction of the crop size.
  method: hybrid # str: how to sample the center of crops (one of 'uniform', 'keypoints', 'density', 'hybrid') 
```

### Model

The model configuration is further split into a `backbone`, optionally a `neck` and a 
number of heads.

Changing the `model` configuration should only be done by expert users, and in rare 
occasions. When updating a model configuration (e.g. adding more deconvolution layers 
to a `HeatmapHead`) must be done in a way where the model configuration still makes 
sense for the project (e.g. the number of heatmaps output needs to match the number of 
bodyparts in the project).

An example model configuration for a single-animal HRNet would look something like:

```yaml
model:
  backbone:  # the BaseBackbone used by the pose model
    type: HRNet
    model_name: hrnet_w18  # creates an HRNet W18 backbone
  backbone_output_channels: 18
  heads:  # configures how the different heads will make predictions
    bodypart:  # configures how pose will be predicted for bodyparts
      type: HeatmapHead
      predictor:  # the BasePredictor used to make predictions from the head's outputs
        type: HeatmapPredictor
          ...
      target_generator:  # the BaseTargetGenerator used to create targets for the head
        type: HeatmapPlateauGenerator
          ...
      criterion:  # the loss criterion used for the head
        ...
      ...  # head-specific options, such as `heatmap_config` or `locref_config` for a "HeatmapHead"
```

The `backbone`, `neck` and `head` configurations are loaded using the
`deeplabcut.pose_estimation_pytorch.models.backbones.base.BACKBONES`,
`deeplabcut.pose_estimation_pytorch.models.necks.base.NECKS` and 
`deeplabcut.pose_estimation_pytorch.models.heads.base.HEADS` registries. You specify 
which type to load with the `type` parameter. Any argument for the head can then be used
in the configuration.

So to use an `HRNet` backbone for your model (as defined in 
`deeplabcut.pose_estimation_pytorch.models.backbones.hrnet.HRNet`), you could set:

```yaml
model:
  backbone:
    type: HRNet
    model_name: hrnet_w32  # creates an HRNet W32
    pretrained: true  # the backbone weights for training will be loaded from TIMM (pre-trained on ImageNet)
    interpolate_branches: false  # don't interpolate & concatenate channels from all branches 
    increased_channel_count: true  # use the incre_modules defined in the TIMM HRNet
  backbone_output_channels: 128  # number of channels output by the backbone
```

### Runner

The runner contains elements relating to the runner to use (including the optimizer and 
learning rate schedulers). Unless you're experienced with machine learning and training 
models **it is not recommended to change the optimizer or scheduler**.

```yaml
runner:
  type: PoseTrainingRunner  # should not need to modify this
  key_metric: "test.mAP"  # the metric to use to select the "best snapshot"
  key_metric_asc: true  # whether "larger=better" for the key_metric
  eval_interval: 1  # the interval between each passes through the evaluation dataset
  optimizer:  # the optimizer to use to train the model
    ...
  scheduler:  # optional: a learning rate scheduler
    ...
  snapshots:  # parameters for the TorchSnapshotManager
    max_snapshots: 5  # the maximum number of snapshots to save (the "best" model does not count as one of them)
    save_epochs: 25  # the interval between each snapshot save  
    save_optimizer_state: false  # whether the optimizer state should be saved with the model snapshots (very little reason to set to true)
```

**Key metric**: Every time the model is evaluated on the test set, metrics are computed 
to see how the model is performing. The key metric is used to determine whether the 
current model is the "best" so far. If it is, the snapshot is saved as `...-best.pt`. 
For pose models, metrics to choose from would be `test.mAP` (with `key_metric_asc: true`
) or `test.rmse` (with `key_metric_asc: false`). 

**Evaluation interval**: Evaluation slows down training (it takes time to go through all
the evaluation images, make predictions and log results!). So instead of evaluating 
after every epoch, you could decide to evaluate every 5 epochs (by setting
`eval_interval: 5`). While this means you get coarser information about how your model 
is training, it can speed up training on large datasets.

**Optimizer**: Any optimizer inheriting `torch.optim.Optimizer`. More information about 
optimizers can be found in [PyTorch's documentation](
https://pytorch.org/docs/stable/optim.html). Examples:

```yaml
  # SGD with initial learning rate 1e-3 and momentum 0.9
  #  see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  optimizer:
    type: SGD
    params:
      lr: 1e-3
      momentum: 0.9

  # AdamW optimizer with initial learning rate 1e-4
  #  see https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  optimizer:
    type: AdamW
    params:
      lr: 1e-4
```

**Scheduler**: YYou can use [any scheduler](
https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) defined in
`torch.optim.lr_scheduler`, where the arguments given are arguments of the scheduler. 
The default scheduler is an LRListScheduler, which changes the learning rates at each 
milestone to the corresponding values in `lr_list`. Examples:

```yaml
  # reduce to 1e-5 at epoch 160 and 1e-6 at epoch 190
  scheduler:
    type: LRListScheduler
    params:
      lr_list: [ [ 1e-5 ], [ 1e-6 ] ]
      milestones: [ 160, 190 ]

  # Decays the learning rate of each parameter group by gamma every step_size epochs
  #   see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
  scheduler:
    type: StepLR
    params:
      step_size: 100
      gamma: 0.1
```

### Train Settings

The `train_settings` key contains parameters that are specific to training. For more 
information about the `dataloader_workers` and `dataloader_pin_memory` settings, see
[Single- and Multi-process Data Loading](
https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)
and [memory pinning](https://pytorch.org/docs/stable/data.html#memory-pinning). Setting
`dataloader_workers: 0` uses single-process data loading, while setting it to 1 or more
will use multi-process data loading. You should always keep 
`dataloader_pin_memory: true` when training on an NVIDIA GPU. 

```yaml
train_settings:
  batch_size: 1  # the batch size used for training
  dataloader_workers: 0  # the number of workers for the PyTorch Dataloader 
  dataloader_pin_memory: true  # pin DataLoader memory
  display_iters: 500  # the number of iterations (steps) between each log print
  epochs: 200  # the maximum number of epochs for which to train the model
  seed: 42  # the random seed to set for reproducibility
```

### Logger

Training runs are logged to the model folder (where the snapshots are stored) by 
default.

Additionally, you can log results to [Weights and Biases](https://wandb.ai/site), by adding a
`WandbLogger`. Just make sure you're logged in to your `wandb` account before starting 
your training run (with `wandb login` from your shell). For more information, see their
[tutorials](https://docs.wandb.ai/tutorials) and their documentation for 
[`wandb.init`](https://docs.wandb.ai/ref/python/init). You can also log images as they are seen by the model to `wandb` 
with the `image_log_interval`. This logs a random train and test image, as well as the 
targets and heatmaps for that image.

Logging to `wandb` is a good way to keep track of what you've run, including performance
and metrics.

```yaml
logger:
 type: WandbLogger
 image_log_interval: 5  # how often images are logged to wandb (in epochs)
 project_name: my-dlc3-project  # the name of the project where the run should be logged
 run_name: dekr-w32-shuffle0  # the name of the run to log
 ...  # any other argument you can pass to `wandb.init`, such as `tags: ["dekr", "split=0"]`
```

## Training Top-Down Models

Top-down models are split into two main elements: a detector (localizing individuals in
the images) and a pose model predicting each individual's pose (once localization is 
done, obtaining pose is just like getting pose in a single-animal model!).

The "pose" part of the model configuration is exactly the same as for single-animal or
bottom-up models (configured through the `data`, `model`, `runner` and `train_settings`
). The detector is configured through a detector key, at the top-level of the
configuration.

### Detector Configuration

When training top-down models, you also need to configure how the detector will be 
trained. All information relating to the detector is placed under the `detector` key.

```yaml
detector:
  data:  # which data augmentations will be used, same options as for the pose model
    colormode: RGB
    inference:  # default inference configuration for detectors
      normalize_images: true
    train:  # default train configuration for detectors
      affine:
        p: 0.9
        rotation: 30
        scaling: [ 0.5, 1.25 ]
        translation: 40
      hflip: true
      normalize_images: true
  model:  # the detector to train
    type: FasterRCNN
    variant: fasterrcnn_mobilenet_v3_large_fpn
    pretrained: true
  runner:  #  detector train runner configuration (same keys as for the pose model)
    type: DetectorTrainingRunner
    ...
  train_settings: # detector train settings (same keys as for the pose model)
    ...
```

Currently, the only detector available is a `FasterRCNN`. However, multiple variants are
available (you can view the different variants on [torchvision's object detection page](
https://pytorch.org/vision/stable/models.html#object-detection)). It's recommended to
use the fastest detector that brings enough performance. The recommended variants
are the following (from fastest to most powerful, taken from torchvision's
documentation):

| name                              |  Box MAP (larger = more powerful) | Params (larger = more powerful) |    GFLOPS (larger = slower) |
|-----------------------------------|----------------------------------:|--------------------------------:|----------------------------:|
| fasterrcnn_mobilenet_v3_large_fpn |                              32.8 |                           19.4M |                        4.49 |
| fasterrcnn_resnet50_fpn           |                                37 |                           41.8M |                      134.38 |
| fasterrcnn_resnet50_fpn_v2        |                              46.7 |                           43.7M |                      280.37 |
