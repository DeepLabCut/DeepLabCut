#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:06:13 2018

@author: alex

This is a test script that should achieve a certain performance on your system.

See values on our system, the DLC 2.0 docker with TF 1.8 on a NVIDIA GTX 1080Ti.
https://github.com/MMathisLab/Docker4DeepLabCut2.0

This test trains on the open field data set for about 30 minutes (15k iterations).

The results will be something like this (Run for DLC 2.0.x)

The analysis of the video takes 41 seconds (batch size 32) and creating the frames 8 seconds (+ a few seconds for ffmpeg) to create the video.
"""

# Importing the toolbox (takes several seconds)
import deeplabcut
import os
from pathlib import Path


# Loading example data set
path_config_file = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30/config.yaml")
deeplabcut.load_demo_data(path_config_file)

Nets = ["mobilenet_v2_0.35", "mobilenet_v2_1.0", "resnet_50"]  # ,'mobilenet_v2_1.0'
numvariants = 8
uf = False
nunments = len(Nets)
trainIndexes, testIndexes = deeplabcut.mergeandsplit(
    path_config_file, trainindex=0, uniform=True
)

for sh, net_type in enumerate(Nets):  #'mobilenet_v2_1.0']): # 'resnet_50']):
    for jj in range(numvariants):
        # use the same split!
        shuffle = numvariants * sh + jj + 30

        deeplabcut.create_training_dataset(
            path_config_file,
            Shuffles=[shuffle],
            userfeedback=uf,
            net_type=net_type,
            trainIndexes=trainIndexes,
            testIndexes=testIndexes,
        )

        cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
        posefile = os.path.join(
            cfg["project_path"],
            "dlc-models/iteration-"
            + str(cfg["iteration"])
            + "/"
            + cfg["Task"]
            + cfg["date"]
            + "-trainset"
            + str(int(cfg["TrainingFraction"][0] * 100))
            + "shuffle"
            + str(shuffle),
            "train/pose_cfg.yaml",
        )

        edits = dict()
        edits["display_iters"] = 1000
        edits["multi_step"] = [[0.005, 5000], [0.02, 100001]]
        edits["save_iters"] = 10000

        if jj == 4:
            edits["dataset_type"] = "default"
            edits["global_scale"] = 0.25

        elif jj == 5:
            edits["dataset_type"] = "default"
            edits["global_scale"] = 0.5
        elif jj == 6:
            edits["dataset_type"] = "default"
            edits["global_scale"] = 0.75

        elif jj == 7:
            edits["dataset_type"] = "default"
            edits["global_scale"] = 1.0

        elif jj == 0:
            edits["dataset_type"] = "imgaug"
            edits["global_scale"] = 0.25
            edits["batch_size"] = 1
            edits["optimizer"] = "sgd"
            edits["scmap_type"] = "plateau"
            edits["grayscale"] = True
            edits["rotation"] = True
            edits["covering"] = True
            edits["hist_eq"] = True
            edits["motion_blur"] = True  # [["k", 7],["angle", [-90, 90]]]

        elif jj == 1:
            edits["dataset_type"] = "imgaug"
            edits["global_scale"] = 0.5
            edits["batch_size"] = 1
            edits["optimizer"] = "sgd"
            edits["scmap_type"] = "plateau"
            edits["grayscale"] = True
            edits["rotation"] = True
            edits["covering"] = True
            edits["hist_eq"] = True
            edits["motion_blur"] = True  # [["k", 7],["angle", [-90, 90]]]

        elif jj == 2:
            edits["dataset_type"] = "imgaug"
            edits["global_scale"] = 0.75
            edits["batch_size"] = 1
            edits["optimizer"] = "sgd"
            edits["scmap_type"] = "plateau"
            edits["grayscale"] = True
            edits["rotation"] = True
            edits["covering"] = True
            edits["hist_eq"] = True
            edits["motion_blur"] = True  # [["k", 7],["angle", [-90, 90]]]

        elif jj == 3:
            edits["dataset_type"] = "imgaug"
            edits["global_scale"] = 1.0
            edits["batch_size"] = 1
            edits["optimizer"] = "sgd"
            edits["scmap_type"] = "plateau"
            edits["grayscale"] = True
            edits["rotation"] = True
            edits["covering"] = True
            edits["hist_eq"] = True
            edits["motion_blur"] = True  # [["k", 7],["angle", [-90, 90]]]
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

        posefile = os.path.join(
            cfg["project_path"],
            "dlc-models/iteration-"
            + str(cfg["iteration"])
            + "/"
            + cfg["Task"]
            + cfg["date"]
            + "-trainset"
            + str(int(cfg["TrainingFraction"][0] * 100))
            + "shuffle"
            + str(shuffle),
            "test/pose_cfg.yaml",
        )

        edits = dict()
        if jj == 0:
            edits["global_scale"] = 0.25
        elif jj == 1:
            edits["global_scale"] = 0.5
        elif jj == 2:
            edits["global_scale"] = 0.75
        elif jj == 3:
            edits["global_scale"] = 1.0
        elif jj == 4:
            edits["global_scale"] = 0.25
        elif jj == 5:
            edits["global_scale"] = 0.5
        elif jj == 6:
            edits["global_scale"] = 0.75
        elif jj == 7:
            edits["global_scale"] = 1.0
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

        print("TRAIN NETWORK")
        deeplabcut.train_network(
            path_config_file, shuffle=shuffle, max_snapshots_to_keep=15
        )  # ,saveiters=5000,displayiters=100)

        cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
        cfg["snapshotindex"] = "all"
        deeplabcut.auxiliaryfunctions.write_config(path_config_file, cfg)

        print("EVALUATE")
        deeplabcut.evaluate_network(
            path_config_file, plotting=False, Shuffles=[shuffle], rescale=True
        )
        cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
        cfg["snapshotindex"] = -1

        deeplabcut.auxiliaryfunctions.write_config(path_config_file, cfg)
        deeplabcut.evaluate_network(
            path_config_file, plotting=True, Shuffles=[shuffle], rescale=True
        )

        # print("Analyze Video")
        # videofile_path = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30','videos','m3v1mp4.mp4')
        # deeplabcut.analyze_videos(path_config_file,[videofile_path], shuffle=shuffle,batchsize=64)

        # print("Create Labeled Video")
        # deeplabcut.create_labeled_video(path_config_file,[videofile_path],save_frames=False, shuffle=shuffle)
