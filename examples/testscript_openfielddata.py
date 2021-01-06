#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5 18:06:13 2018

@author: alex

This is a test script that should achieve a certain performance on your system.

See values on our system, the DLC 2.0 docker with TF 1.8 on a NVIDIA GTX 1080Ti.
https://github.com/MMathisLab/Docker4DeepLabCut2.0

This test trains on the open field data set for about 30 minutes (15k iterations).

The results will be something like this:

Results for 15001  training iterations: 95 1 train error: 2.89 pixels. Test error: 2.81  pixels.
With pcutoff of 0.1  train error: 2.89 pixels. Test error: 2.81 pixels

The analysis of the video takes 41 seconds (batch size 32) and creating the frames 8 seconds (+ a few seconds for ffmpeg) to create the video.
"""
import os

os.environ["DLClight"] = "True"

import deeplabcut
import os


if __name__ == '__main__':
    # Loading example data set
    path_config_file = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30/config.yaml")
    deeplabcut.load_demo_data(path_config_file)
    shuffle = 13

    deeplabcut.create_training_dataset(path_config_file, Shuffles=[shuffle])
    cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)

    # example how to set pose config variables:
    posefile, _, _ = deeplabcut.return_train_network_path(path_config_file, shuffle=shuffle)
    edits = {"save_iters": 15000, "display_iters": 1000, "multi_step": [[0.005, 15001]]}
    DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

    print("TRAIN NETWORK")
    deeplabcut.train_network(path_config_file, shuffle=shuffle, max_snapshots_to_keep=3)

    print("EVALUATE")
    deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle], plotting=True)

    print("Analyze Video")
    videofile_path = os.path.join(
        os.getcwd(), "openfield-Pranav-2018-10-30", "videos", "m3v1mp4.mp4"
    )
    deeplabcut.analyze_videos(
        path_config_file, [videofile_path], shuffle=shuffle
    )  # ,videotype='.mp4')

    print("Create Labeled Video")
    deeplabcut.create_labeled_video(
        path_config_file, [videofile_path], save_frames=False, shuffle=shuffle
    )  # ,videotype='.mp4')
