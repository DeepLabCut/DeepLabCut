"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

cfg = dict()

cfg["stride"] = 8.0
cfg["weigh_part_predictions"] = False
cfg["weigh_negatives"] = False
cfg["fg_fraction"] = 0.25

# imagenet mean for resnet pretraining:
cfg["mean_pixel"] = [123.68, 116.779, 103.939]
cfg["shuffle"] = True
cfg["snapshot_prefix"] = "./snapshot"
cfg["log_dir"] = "log"
cfg["global_scale"] = 1.0
cfg["location_refinement"] = False
cfg["locref_stdev"] = 7.2801
cfg["locref_loss_weight"] = 1.0
cfg["locref_huber_loss"] = True
cfg["optimizer"] = "sgd"
cfg["intermediate_supervision"] = False
cfg["intermediate_supervision_layer"] = 12
cfg["regularize"] = False
cfg["weight_decay"] = 0.0001
cfg["crop_pad"] = 0
cfg["scoremap_dir"] = "test"

cfg["batch_size"] = 1

# types of datasets, see factory: deeplabcut/pose_estimation_tensorflow/dataset/factory.py
cfg["dataset_type"] = "imgaug"  # >> imagaug default as of 2.2
# you can also set this to deterministic, see https://github.com/DeepLabCut/DeepLabCut/pull/324
cfg["deterministic"] = False
cfg["mirror"] = False

# for DLC 2.2. (here all set False to not use PAFs/pairwise fields)
cfg["pairwise_huber_loss"] = True
cfg["weigh_only_present_joints"] = False
cfg["partaffinityfield_predict"] = False
cfg["pairwise_predict"] = False
