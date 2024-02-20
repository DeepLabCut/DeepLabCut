(dlc3-pytorch-config)=
# The PyTorch Configuration file

The `pytorch_config.yaml` file specifies everything about how you'll train a model for a
project.

You can create "base" configurations using `deeplabcut.create_training_set` or 
`deeplabcut.create_training_model_comparison`.

## Model Architectures

### Bottom-Up

There are a few keys which define your model architecture and how it will be trained:

- `batch_size`: the batch size to train with (inference always runs with batch size 1)
- `data`: the data augmentations you'll apply to images when training (during inference,
only normalization, rescaling and auto-padding are kept - this is subject to change)
- `device`: the device to use to train (such as `cpu`, `cuda`, `cuda:0`, ...)
- `epochs`: the number of epochs to train for
- `method`: either `bu` for bottom-up or `td` for `top-down` 
- `model`: the architecture of the model you'll train (including loss criterions)
- `with_center_keypoints`: for models (like DEKR) that need to know the position of the
center of individuals
- `optimizer`: the optimizer to use (the name of any Torch optimizer works)
- `save_epochs`: the number of epochs between each model snapshot
- `scheduler`: learning rate scheduler

### Top-Down

Top-down models are configured in a very similar way to bottom up ones. The keys used to
configure the pose model are exactly the same, and a few additional keys are added to 
configure how you want your detector to be trained:

- `detector`: the configuration for the detector, with `model`, `optimizer`, 
`scheduler`, `batch_size`, `epochs` and `save_epochs` options
- `data_detector`: the data augmentations to use to train the detector (in the same 
format as the ones for the pose model)

In this case, the augmentations described in `data` are applied to the pose model. 

## Data Augmentations

### Resizing Images

Resizes the images while preserving the aspect ratio (first resizes to the maximum 
possible size, then adds padding for the missing pixels).

```yaml
data:
  resize:
    width: 1300
    height: 800
    keep_ratio: true
```

### Logging Results

Logs results to Weights & Biases.

```yaml
logger:
 type: 'WandbLogger'
 project_name: 'my-dlc3-project'
 run_name: 'dekr-w32-shuffle0'
```
