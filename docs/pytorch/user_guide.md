(dlc3-user-guide)=
# DeepLabCut 3.0 - Pytorch User Guide

## Using DeepLabCut 3.0

PyTorch models can be trained on any DeepLabCut project. Simply add a new key to your 
project `config.yaml` file specifying `engine: pytorch`. Then any new training dataset
that will be created will be a PyTorch model (see 
[Creating Shuffles and Model Configuration](#Creating-Shuffles-and-Model-Configuration))
to learn more about training PyTorch models. To train Tensorflow models again, you can
set `engine: tensorflow`.

### Using the GUI

You can use the GUI to train DeepLabCut projects. However, you cannot switch between 
PyTorch and Tensorflow models while using the GUI. If you have set your engine to
`pytorch`, then the GUI will only offer the creation of Pytorch shuffles.

You can create `tensorflow` shuffles and train them again by setting the 
`engine: tensorflow` key and either re-loading the project or closing the GUI entirely
and opening the project again.

## Major changes

### From iterations to epochs

Pytorch models in DeepLabCut 3.0 are trained for a set number of epochs, instead of a 
maximum number of iterations. An epoch is a single pass through the training dataset, 
which means your model has seen each training image exactly once.

So if you have 64 training images for your network, an epoch is 64 iterations with batch
size 1 (or 32 iterations with batch size 2, 16 with batch size 4, etc.).

## API

### Development state

The table below describes the DeepLabCut API methods that have been implemented,
as well as indications which options are not yet implemented, and which parameters
are not valid for the DLC 3.0 API.

You can find all DLC 3.0 API methods and the parameters they can be called with in


| API Method                     | Implemented | Parameters not yet implemented                                                                                                | Parameters invalid for pytorch                      |
|--------------------------------|:-----------:|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `train_network`                |     🟢      | `max_snapshots_to_keep`, `keepdeconvweights`                                                                                  | `maxiters`, `saveiters`, `allow_growth`, `autotune` |
| `return_train_network_path`    |     🟢      |                                                                                                                               |                                                     |
| `evaluate_network`             |     🟢      | `comparisonbodyparts`, `rescale`, `per_keypoint_evaluation`                                                                   |                                                     |
| `return_evaluate_network_data` |     🔴      |                                                                                                                               | `TFGPUinference`, `allow_growth`                    |
| `analyze_videos`               |     🟢      | `use_shelve`, `save_as_csv`, `in_random_order`, `batchsize`, `cropping`, `dynamic`, `robust_nframes`, `n_tracks`, `calibrate` |                                                     |
| `create_tracking_dataset`      |     🔴      |                                                                                                                               |                                                     |
| `analyze_time_lapse_frames`    |     🔴      |                                                                                                                               |                                                     |
| `convert_detections2tracklets` |     🟢      | `greedy`, `calibrate`, `window_size`                                                                                          |                                                     |
| `extract_maps`                 |     🔴      |                                                                                                                               |                                                     |
| `visualize_scoremaps`          |     🔴      |                                                                                                                               |                                                     |
| `visualize_locrefs`            |     🔴      |                                                                                                                               |                                                     |
| `visualize_paf`                |     🔴      |                                                                                                                               |                                                     |
| `extract_save_all_maps`        |     🔴      |                                                                                                                               |                                                     |
| `export_model`                 |     🔴      |                                                                                                                               |                                                     |


### Creating Shuffles and Model Configuration

You can configure models using the `pytorch_config.yaml` file, as described
[here](dlc3-pytorch-config). You can use the same methods to create new shuffles in 
DeepLabCut 3.0 as you did for Tensorflow models (`deeplabcut.create_training_dataset`
and `deeplabcut.create_training_model_comparison`).

You can see a list of supported
architectures/variants by using:

```python
from deeplabcut.pose_estimation_pytorch import available_models
print(available_models())
```
