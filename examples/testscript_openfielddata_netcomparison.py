#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a test script to compare the networks. On Jan 3rd 2020:

Jan 2020:
MobileNetV2 0.35
Results for 10000  training iterations: 95 1 train error: 5.79 pixels. Test error: 5.63  pixels.
With pcutoff of 0.4  train error: 5.79 pixels. Test error: 5.63 pixels

ResNet 50
Results for 10000  training iterations: 95 2 train error: 3.61 pixels. Test error: 3.7  pixels.
With pcutoff of 0.4  train error: 3.61 pixels. Test error: 3.7 pixels

EffNet-b3
Results for 10000  training iterations: 95 3 train error: 6.86 pixels. Test error: 6.63  pixels.
With pcutoff of 0.4  train error: 6.86 pixels. Test error: 6.63 pixels

Note: Not too good on video either!

TODO: Note we should still optimize the MobNet & EffNet learning rates for this dataset (also training is pretty short!)
TODO: change to frozen backbone!
"""


import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import deeplabcut
import numpy as np

# Loading example data set
path_config_file = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30/config.yaml")
cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
maxiters = 10000

deeplabcut.load_demo_data(path_config_file)

## Create one split and make Shuffle 2 and 3 have the same split.
###Note that the new function in DLC 2.1 simplifies network/augmentation comparisons greatly:
deeplabcut.create_training_model_comparison(
    path_config_file,
    num_shuffles=1,
    net_types=["mobilenet_v2_0.35", "resnet_50", "efficientnet-b3"],
    augmenter_types=["imgaug"],
)

freezeencoder = False  # True
for shuffle in 1 + np.arange(3):

    posefile, _, _ = deeplabcut.return_train_network_path(
        path_config_file, shuffle=shuffle
    )

    # for EfficientNet
    edits = {
        "decay_steps": maxiters,
        "lr_init": 0.0005 * 12,
        "freezeencoder": freezeencoder,
    }
    DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)
    # imgaug
    edits = {"rotation": 180, "motion_blur": True, "freezeencoder": freezeencoder}
    DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

    print("TRAIN NETWORK", shuffle)
    deeplabcut.train_network(
        path_config_file,
        shuffle=shuffle,
        saveiters=10000,
        displayiters=200,
        maxiters=maxiters,
        max_snapshots_to_keep=11,
    )

    print("EVALUATE")
    deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle], plotting=True)

    print("Analyze Video")

    videofile_path = os.path.join(
        os.getcwd(), "openfield-Pranav-2018-10-30", "videos", "m3v1mp4.mp4"
    )

    deeplabcut.analyze_videos(path_config_file, [videofile_path], shuffle=shuffle)

    print("Create Labeled Video and plot")
    deeplabcut.create_labeled_video(path_config_file, [videofile_path], shuffle=shuffle)
    deeplabcut.plot_trajectories(path_config_file, [videofile_path], shuffle=shuffle)
