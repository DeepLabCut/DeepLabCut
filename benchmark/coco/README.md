# Training DeepLabCut models on COCO Projects

## Training and Evaluation Process

There are three essential steps to follow to train a model

1. Creating a `pytorch_config.yaml` model configuration file, which specifies the
architecture of the model, but also the optimizer, learning rate and data augmentations.
2. Training the network. For bottom up models, this means only training the pose model,
while for top-down models you can either just train the pose model (if you already have 
a detector), or train a detector and then a pose model.
3. Evaluation. Once you have a trained model, you can evaluate all of the snapshots
you've saved. For top-down models, you can evaluate using ground-truth bounding boxes or
detector bounding boxes (by passing a detector snapshot as well).

### Creating a Model Configuration File

You can either copy an existing model configuration file and modify it to fit your 
updated project (or run a new experiment), or you can create a default one for a given
model architecture using `make_config.py`.

This will create a `train` and a `test` folder in the `output` folder that you've 
specified. The configuration file will be saved in the `train` folder, and an 
`inference_cfg.yaml` file will be created in the `test` folder, which contains
parameters for tracking.

You can log to Weights & Biases by adding a logger to your configuration:

```yaml
logger:
 type: 'WandbLogger'
 project_name: 'dlc3-rodent'
 run_name: 'test-tokenpose'
```

### Training a Model

If you're training a top-down model and don't want to train a detector, simply pass
`--detector-epochs 0` as a command line parameter.

### Evaluation

For top-down models, running `evaluate.py` without given the path to a
`--detector-snapshot` will use ground-truth bounding boxes. If you specify a detector
snapshot, it will first compute bounding boxes in the images using the trained detector,
and then will use these bounding boxes for inference.

## Training on Images of Variable Size

If your images have different sizes (and you want to make sure all of them have the same
size when being given to the models), you can add a `resize` parameter to your 
`pytorch_config.yaml` file:

```yaml
data:
  resize:
    width: 1300
    height: 800
    keep_ratio: true
```

This will resize images using 
`deeplabcut.pose_estimation_pytorch.data.transforms.DLCResize`, which resizes while
preserving the aspect ratio, and then pads the image to the correct size. If all of 
your images have the same aspect ratio (and your target width/height), you can resize
using `keep_ratio: false`.

## Example

Creating the model configuration file

```bash
python make_config.py \
  /home/niels/datasets/rodent \
  /home/niels/datasets/rodent/experiments/exp_1 \
  dekr_w32 \
  --train_file corrected_train.json
```

Then modify that configuration (such as the data augmentations) to have it match 
whatever is required for your project. Once you're done and happy with your project 
configuration, you can use it to train a model:

```bash
python train.py \
  /home/niels/datasets/rodent \
  /home/niels/datasets/rodent/experiments/exp_4/train/pytorch_config.yaml \
  --detector-epochs 20 \
  --detector-save-epochs 5 \
  --epochs 50 \
  --save-epochs 10 \
  --train_file train.json \
  --test_file test.json \
  --device cuda:0
```

Evaluating a trained model:

```bash
python evaluate.py \
  /home/niels/datasets/rodent \
  /home/niels/datasets/rodent/experiments/exp_2/train/pytorch_config.yaml \
  /home/niels/datasets/rodent/experiments/exp_2/train/snapshot-10.pt \
  --detector_path /home/niels/datasets/rodent/experiments/exp_2/train/detector-snapshot-200.pt
  --train_file train.json \
  --test_file test.json \
  --device cuda:0
```
