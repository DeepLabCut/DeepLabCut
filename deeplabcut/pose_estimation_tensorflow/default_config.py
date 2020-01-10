"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

from easydict import EasyDict as edict

cfg = edict()

cfg.stride = 8.0
cfg.weigh_part_predictions = False
cfg.weigh_negatives = False
cfg.fg_fraction = 0.25
cfg.weigh_only_present_joints = False
cfg.mean_pixel = [123.68, 116.779, 103.939]
cfg.shuffle = True
cfg.snapshot_prefix = "./snapshot"
cfg.log_dir = "log"
cfg.global_scale = 1.0
cfg.location_refinement = False
cfg.locref_stdev = 7.2801
cfg.locref_loss_weight = 1.0
cfg.locref_huber_loss = True
cfg.optimizer = "sgd"
cfg.intermediate_supervision = False
cfg.intermediate_supervision_layer = 12
cfg.regularize = False
cfg.weight_decay = 0.0001
cfg.mirror = False
cfg.crop_pad = 0
cfg.scoremap_dir = "test"

cfg.batch_size = 1

#types of datasets, see factory: deeplabcut/pose_estimation_tensorflow/dataset/factory.py
cfg.dataset_type = "default"
#you can also set this to deterministic, see https://github.com/AlexEMG/DeepLabCut/pull/324
cfg.deterministic = False

# Parameters for augmentation with regard to cropping
# Added and described in "Using DeepLabCut for 3D markerless pose estimation across species and behaviors" 
# Source: https://www.nature.com/articles/s41596-019-0176-0
cfg.crop = False

cfg.cropratio= 0.25 #what is the fraction of training samples with cropping?
cfg.minsize= 100 #what is the minimal frames size for cropping plus/minus ie.. [-100,100]^2 for an arb. joint
cfg.leftwidth= 400
#limit width  [-leftwidth*u-100,100+u*rightwidth] x [-bottomwith*u-100,100+u*topwidth] where u is always a (different) random number in unit interval
cfg.rightwidth= 400
cfg.topheight= 400
cfg.bottomheight= 400
