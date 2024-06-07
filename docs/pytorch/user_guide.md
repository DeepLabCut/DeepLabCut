(dlc3-user-guide)=
# DeepLabCut 3.0 - Pytorch User Guide

## Using DeepLabCut 3.0

**DeepLabCut 3.0 keeps the same high-level API that you know, but has a full new PyTorch backend. 
Moreover, it is a rewrite that more developer friendly, more powerful, and built for modern deep
learning-based computer vision applications.**

**NOTE**🔥: We suggest that if you're just starting with DeepLabCut you start with the PyTorch backend. 
You will easily know which "engine" you are using by looking at the main `config.yaml` file, or top right corner in the GUI. 
If you have DeepLabCut projects in TensorFlow, we've got you covered too: you can seamlessly switch to train your already labeled data
by simply switching the engine (and thereby also compare performance). In short, expect a boost 🔥.

In short, PyTorch models can be trained in any DeepLabCut project. If you have a project already made,  simply add a new key to your 
project `config.yaml` file specifying `engine: pytorch`. Then any new training dataset
that will be created will be a PyTorch model (see 
[Creating Shuffles and Model Configuration](#Creating-Shuffles-and-Model-Configuration))
to learn more about training PyTorch models. To train Tensorflow models again, you can
set `engine: tensorflow`.

### Using the GUI

You can use the GUI to train DeepLabCut projects. However, you cannot switch between 
PyTorch and Tensorflow models while using the GUI. If you have set your engine to
`pytorch`, then the GUI will only offer the creation of PyTorch shuffles.

You can create `tensorflow` shuffles and train them again by setting the 
`engine: tensorflow` in the top right corner of the GUI.

## Major changes

### From iterations to epochs

Pytorch models in DeepLabCut 3.0 are trained for a set number of `epochs`, instead of a 
maximum number of `iterations`. An epoch is a single pass through the training dataset, 
which means your model has seen each training image exactly once.

- So if you have 64 training images for your network, an epoch is 64 iterations with batch
size 1 (or 32 iterations with batch size 2, 16 with batch size 4, etc.).

## API

### Creating Shuffles and Model Configuration

You can configure models using the `pytorch_config.yaml` file, as described
[here](dlc3-pytorch-config). You can use the same methods to create new shuffles in 
DeepLabCut 3.0 as you did for Tensorflow models (`deeplabcut.create_training_dataset`
and `deeplabcut.create_training_model_comparison`).

More information about the different PyTorch model architectures available in DeepLabCut
is available [here](dlc3-pytorch-config). You can see a list of supported 
architectures/variants by using:

```python
from deeplabcut.pose_estimation_pytorch import available_models
print(available_models())
```

### Development State and Road Map 🚧

The table below describes the DeepLabCut API methods that have been implemented,
as well as indications which options are not yet implemented, and which parameters
are not valid for the DLC 3.0 API.


| API Method                     | Implemented | Parameters not yet implemented                                                                                                | Parameters invalid for pytorch                      |
|--------------------------------|:-----------:|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `train_network`                |     🟢      | `keepdeconvweights`                                                                                                           | `maxiters`, `saveiters`, `allow_growth`, `autotune` |
| `return_train_network_path`    |     🟢      |                                                                                                                               |                                                     |
| `evaluate_network`             |     🟢      | `comparisonbodyparts`, `rescale`, `per_keypoint_evaluation`                                                                   |                                                     |
| `return_evaluate_network_data` |     🔴      |                                                                                                                               | `TFGPUinference`, `allow_growth`                    |
| `analyze_videos`               |     🟢      | `use_shelve`, `save_as_csv`, `in_random_order`, `batchsize`, `cropping`, `dynamic`, `robust_nframes`, `n_tracks`, `calibrate` |                                                     |
| `create_tracking_dataset`      |     🔴      |                                                                                                                               |                                                     |
| `analyze_time_lapse_frames`    |     🟠      |   the name has changed to  `analyze_images` to better reflect what it actually does (no video needed)                         |                                                     |
| `convert_detections2tracklets` |     🟢      | `greedy`, `calibrate`, `window_size`                                                                                          |                                                     |
| `extract_maps`                 |     🔴      |                                                                                                                               |                                                     |
| `visualize_scoremaps`          |     🔴      |                                                                                                                               |                                                     |
| `visualize_locrefs`            |     🔴      |                                                                                                                               |                                                     |
| `visualize_paf`                |     🔴      |                                                                                                                               |                                                     |
| `extract_save_all_maps`        |     🔴      |                                                                                                                               |                                                     |
| `export_model`                 |     🔴      |                                                                                                                               |                                                     |
