#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""

This is a test script to compare the loaders and models. 

This script creates one identical splits for the openfield test dataset and trains it with imgaug (default), scalecrop
and the tensorpack loader. We also compare 3 backbones (mobilenet, resnet, efficientnet)

My results were (Run with DLC *2.2.0.4* in Jan 6 2022) for 50 k iterations

DLC_mobnet_35_openfieldOct30shuffle0_50000 and Imgaug with # of training iterations: 50000
Results for 50000 training iterations: 95 0 train error: 3.06 pixels. Test error: 3.44 pixels.
With pcutoff of 0.4 train error: 3.06 pixels. Test error: 3.44 pixel

DLC_mobnet_35_openfieldOct30shuffle1_50000 and scalecrop with # of training iterations: 50000
Results for 50000 training iterations: 95 1 train error: 2.44 pixels. Test error: 3.84 pixels.
With pcutoff of 0.4 train error: 2.44 pixels. Test error: 3.84 pixels

DLC_mobnet_35_openfieldOct30shuffle2_50000 and tensorpack with # of training iterations: 50000
Results for 50000 training iterations: 95 2 train error: 2.41 pixels. Test error: 3.04 pixels.
With pcutoff of 0.4 train error: 2.41 pixels. Test error: 3.04 pixels

DLC_resnet50_openfieldOct30shuffle3_50000 and Imgaug with # of training iterations: 50000
Results for 50000 training iterations: 95 3 train error: 2.69 pixels. Test error: 2.97 pixels.
With pcutoff of 0.4 train error: 2.69 pixels. Test error: 2.97 pixels

DLC_resnet50_openfieldOct30shuffle4_50000 and scalecrop with # of training iterations: 50000
Results for 50000 training iterations: 95 4 train error: 2.0 pixels. Test error: 2.69 pixels.
With pcutoff of 0.4 train error: 2.0 pixels. Test error: 2.69 pixels

DLC_resnet50_openfieldOct30shuffle5_50000 and tensorpack with # of training iterations: 50000
Results for 50000 training iterations: 95 5 train error: 1.96 pixels. Test error: 2.65 pixels.
With pcutoff of 0.4 train error: 1.96 pixels. Test error: 2.65 pixels

DLC_effnet_b3_openfieldOct30shuffle6_50000 with Imgaug with # of training iterations: 50000
Results for 50000 training iterations: 95 6 train error: 2.63 pixels. Test error: 2.65 pixels.
With pcutoff of 0.4 train error: 2.63 pixels. Test error: 2.65 pixels

effnet with tensorpack and scalecrop didn't converge.


Notice: despite the higher RMSE for imgaug due to the augmentation,
the network performs much better on the testvideo (see Neuron Primer: https://www.cell.com/neuron/pdf/S0896-6273(20)30717-0.pdf)

"""


import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import deeplabcut
import numpy as np

# Loading example data set
path_config_file = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30/config.yaml")
cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)

maxiters = 50000
saveiters = 10000
displayiters = 500

deeplabcut.load_demo_data(path_config_file, createtrainingset=False)
## Create one identical splits for 3 networks and 3 augmentations

###Note that the new function in DLC 2.1 simplifies network/augmentation comparisons greatly:
Shuffles = deeplabcut.create_training_model_comparison(
    path_config_file,
    num_shuffles=1,
    net_types=["mobilenet_v2_0.35", "resnet_50", "efficientnet-b3"],
    augmenter_types=["imgaug", "scalecrop", "tensorpack"],
)

for idx, shuffle in enumerate(Shuffles):
    posefile, _, _ = deeplabcut.return_train_network_path(
        path_config_file, shuffle=shuffle
    )

    # Setting specific parameters for training
    if idx % 3 == 0:  # imgaug
        edits = {"rotation": 180, "motion_blur": True}
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)
    elif idx % 3 == 2:  # Tensorpack
        edits = {"rotation": 180, "noise_sigma": 0.01}
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

    if idx > 5:  # EfficientNet
        print(posefile, "changing now!!")
        edits = {
            "decay_steps": maxiters,
            "lr_init": 0.0005,
        }
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

for shuffle in Shuffles:
    print("TRAIN NETWORK", shuffle)
    deeplabcut.train_network(
        path_config_file,
        shuffle=shuffle,
        saveiters=saveiters,
        displayiters=displayiters,
        maxiters=maxiters,
        max_snapshots_to_keep=11,
    )

    print("Analyze Video")

    videofile_path = os.path.join(
        os.getcwd(), "openfield-Pranav-2018-10-30", "videos", "m3v1mp4.mp4"
    )

    deeplabcut.analyze_videos(path_config_file, [videofile_path], shuffle=shuffle)

    print("Create Labeled Video and plot")
    deeplabcut.create_labeled_video(path_config_file, [videofile_path], shuffle=shuffle)
    deeplabcut.plot_trajectories(path_config_file, [videofile_path], shuffle=shuffle)

print("EVALUATE")
deeplabcut.evaluate_network(path_config_file, Shuffles=Shuffles, plotting=False)
