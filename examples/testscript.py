#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

It should take about 3:30 minutes to run this in a CPU.
It should take about 1:30 minutes on a GPU (incl. downloading the ResNet weights)

It produces nothing of interest scientifically.
"""
import os
import deeplabcut
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


if __name__ == "__main__":
    task = "TEST"  # Enter the name of your experiment Task
    scorer = "Alex"  # Enter the name of the experimenter/labeler

    print("Imported DLC!")
    basepath = os.path.dirname(os.path.realpath(__file__))
    videoname = "reachingvideo1"
    video = [
        os.path.join(
            basepath, "Reaching-Mackenzie-2018-08-30", "videos", videoname + ".avi"
        )
    ]

    # For testing a color video:
    # videoname='baby4hin2min'
    # video=[os.path.join('/home/alex/Desktop/Data',videoname+'.mp4')]
    # to test destination folder:
    dfolder = basepath

    dfolder = None
    net_type = "resnet_50"  
    #net_type = "mobilenet_v2_0.35"
    #net_type = "efficientnet-b0"  # to -b6

    augmenter_type = "default"  # = imgaug!!
    augmenter_type2 = "scalecrop"

    if platform.system() == "Darwin" or platform.system() == "Windows":
        print("On Windows/OSX tensorpack is not tested by default.")
        augmenter_type3 = "imgaug"
    else:
        augmenter_type3 = "tensorpack"  # Does not work on WINDOWS

    numiter = 5

    print("CREATING PROJECT")
    path_config_file = deeplabcut.create_new_project(
        task, scorer, video, copy_videos=True
    )

    cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
    cfg["numframes2pick"] = 5
    cfg["pcutoff"] = 0.01
    cfg["TrainingFraction"] = [0.8]
    cfg["skeleton"] = [["bodypart1", "bodypart2"], ["bodypart1", "bodypart3"]]

    deeplabcut.auxiliaryfunctions.write_config(path_config_file, cfg)

    print("EXTRACTING FRAMES")
    deeplabcut.extract_frames(path_config_file, mode="automatic", userfeedback=False)

    print("CREATING-SOME LABELS FOR THE FRAMES")
    frames = os.listdir(os.path.join(cfg["project_path"], "labeled-data", videoname))
    # As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
    for index, bodypart in enumerate(cfg["bodyparts"]):
        columnindex = pd.MultiIndex.from_product(
            [[scorer], [bodypart], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
        )
        frame = pd.DataFrame(
            100 + np.ones((len(frames), 2)) * 50 * index,
            columns=columnindex,
            index=[os.path.join("labeled-data", videoname, fn) for fn in frames],
        )
        if index == 0:
            dataFrame = frame
        else:
            dataFrame = pd.concat([dataFrame, frame], axis=1)

    dataFrame.to_csv(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            videoname,
            "CollectedData_" + scorer + ".csv",
        )
    )

    dataFrame.to_hdf(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            videoname,
            "CollectedData_" + scorer + ".h5",
        ),
        "df_with_missing",
        format="table",
        mode="w",
    )

    print("Plot labels...")

    deeplabcut.check_labels(path_config_file)

    print("CREATING TRAININGSET")
    deeplabcut.create_training_dataset(
        path_config_file, net_type=net_type, augmenter_type=augmenter_type
    )

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
        + str(1),
        "train/pose_cfg.yaml",
    )

    DLC_config = deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
    DLC_config["save_iters"] = numiter
    DLC_config["display_iters"] = 2
    DLC_config["multi_step"] = [[0.001, numiter]]

    print("CHANGING training parameters to end quickly!")
    deeplabcut.auxiliaryfunctions.write_plainconfig(posefile, DLC_config)

    print("TRAIN")
    deeplabcut.train_network(path_config_file)

    print("EVALUATE")
    deeplabcut.evaluate_network(path_config_file, plotting=True)
    # deeplabcut.evaluate_network(path_config_file,plotting=True,trainingsetindex=33)
    print("CUT SHORT VIDEO AND ANALYZE (with dynamic cropping!)")

    # Make super short video (so the analysis is quick!)

    try:  # you need ffmpeg command line interface
        # subprocess.call(['ffmpeg','-i',video[0],'-ss','00:00:00','-to','00:00:00.4','-c','copy',newvideo])
        newvideo = deeplabcut.ShortenVideo(
            video[0],
            start="00:00:00",
            stop="00:00:01",
            outsuffix="short",
            outpath=os.path.join(cfg["project_path"], "videos"),
        )
    except:  # if ffmpeg is broken/missing
        print("using alternative method")
        newvideo = os.path.join(cfg["project_path"], "videos", videoname + "short.mp4")
        from moviepy.editor import VideoFileClip, VideoClip

        clip = VideoFileClip(video[0])
        clip.reader.initialize()

        def make_frame(t):
            return clip.get_frame(1)

        newclip = VideoClip(make_frame, duration=1)
        newclip.write_videofile(newvideo, fps=30)

    vname = Path(newvideo).stem

    deeplabcut.analyze_videos(
        path_config_file,
        [newvideo],
        save_as_csv=True,
        destfolder=dfolder,
        dynamic=(True, 0.1, 5),
    )

    print("analyze again...")
    deeplabcut.analyze_videos(
        path_config_file, [newvideo], save_as_csv=True, destfolder=dfolder
    )

    print("CREATE VIDEO")
    deeplabcut.create_labeled_video(
        path_config_file, [newvideo], destfolder=dfolder, save_frames=True
    )

    print("Making plots")
    deeplabcut.plot_trajectories(path_config_file, [newvideo], destfolder=dfolder)

    print("EXTRACT OUTLIERS")
    deeplabcut.extract_outlier_frames(
        path_config_file,
        [newvideo],
        outlieralgorithm="jump",
        epsilon=0,
        automatic=True,
        destfolder=dfolder,
    )

    deeplabcut.extract_outlier_frames(
        path_config_file,
        [newvideo],
        outlieralgorithm="fitting",
        automatic=True,
        destfolder=dfolder,
    )

    file = os.path.join(
        cfg["project_path"],
        "labeled-data",
        vname,
        "machinelabels-iter" + str(cfg["iteration"]) + ".h5",
    )

    print("RELABELING")
    DF = pd.read_hdf(file, "df_with_missing")
    DLCscorer = np.unique(DF.columns.get_level_values(0))[0]
    DF.columns.set_levels([scorer.replace(DLCscorer, scorer)], level=0, inplace=True)
    DF = DF.drop("likelihood", axis=1, level=2)
    DF.to_csv(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            vname,
            "CollectedData_" + scorer + ".csv",
        )
    )
    DF.to_hdf(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            vname,
            "CollectedData_" + scorer + ".h5",
        ),
        "df_with_missing",
        format="table",
        mode="w",
    )

    print("MERGING")
    deeplabcut.merge_datasets(path_config_file)  # iteration + 1

    print("CREATING TRAININGSET")
    deeplabcut.create_training_dataset(
        path_config_file, net_type=net_type, augmenter_type=augmenter_type2
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
        + str(1),
        "train/pose_cfg.yaml",
    )
    DLC_config = deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
    DLC_config["save_iters"] = numiter
    DLC_config["display_iters"] = 1
    DLC_config["multi_step"] = [[0.001, numiter]]

    print("CHANGING training parameters to end quickly!")
    deeplabcut.auxiliaryfunctions.write_config(posefile, DLC_config)

    print("TRAIN")
    deeplabcut.train_network(path_config_file)

    try:  # you need ffmpeg command line interface
        # subprocess.call(['ffmpeg','-i',video[0],'-ss','00:00:00','-to','00:00:00.4','-c','copy',newvideo])
        newvideo2 = deeplabcut.ShortenVideo(
            video[0],
            start="00:00:00",
            stop="00:00:01",
            outsuffix="short2",
            outpath=os.path.join(cfg["project_path"], "videos"),
        )

    except:  # if ffmpeg is broken
        newvideo2 = os.path.join(
            cfg["project_path"], "videos", videoname + "short2.mp4"
        )
        from moviepy.editor import VideoFileClip, VideoClip

        clip = VideoFileClip(video[0])
        clip.reader.initialize()

        def make_frame(t):
            return clip.get_frame(1)

        newclip = VideoClip(make_frame, duration=1)
        newclip.write_videofile(newvideo2, fps=30)

    vname = Path(newvideo2).stem

    print("Inference with direct cropping")
    deeplabcut.analyze_videos(
        path_config_file,
        [newvideo2],
        save_as_csv=True,
        destfolder=dfolder,
        cropping=[0, 50, 0, 50],
        allow_growth=True
    )

    print("Extracting skeleton distances, filter and plot filtered output")
    deeplabcut.analyzeskeleton(
        path_config_file, [newvideo2], save_as_csv=True, destfolder=dfolder
    )
    deeplabcut.filterpredictions(path_config_file, [newvideo2])

    deeplabcut.create_labeled_video(
        path_config_file,
        [newvideo2],
        destfolder=dfolder,
        displaycropped=True,
        filtered=True,
    )

    print("Creating a Johansson video!")
    deeplabcut.create_labeled_video(
        path_config_file, [newvideo2], destfolder=dfolder, keypoints_only=True
    )

    deeplabcut.plot_trajectories(
        path_config_file, [newvideo2], destfolder=dfolder, filtered=True
    )

    print("ALL DONE!!! - default cases without Tensorpack loader are functional.")

    print("CREATING TRAININGSET for shuffle 2")
    print("will be used for 3D testscript...")
    # TENSORPACK could fail in WINDOWS...
    deeplabcut.create_training_dataset(
        path_config_file,
        Shuffles=[2],
        net_type=net_type,
        augmenter_type=augmenter_type3,
    )

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
        + str(2),
        "train/pose_cfg.yaml",
    )

    DLC_config = deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
    DLC_config["save_iters"] = 10
    DLC_config["display_iters"] = 2
    DLC_config["multi_step"] = [[0.001, 10]]

    print("CHANGING training parameters to end quickly!")
    deeplabcut.auxiliaryfunctions.write_plainconfig(posefile, DLC_config)

    print("TRAINING shuffle 2, with smaller allocated memory")
    deeplabcut.train_network(path_config_file, shuffle=2, allow_growth=True)

    print("ANALYZING some individual frames")
    deeplabcut.analyze_time_lapse_frames(
        path_config_file,
        os.path.join(cfg["project_path"], "labeled-data/reachingvideo1/"),
    )

    print("Export model...")
    deeplabcut.export_model(path_config_file, shuffle=2, make_tar=False)

    print("Merging datasets...")
    trainIndices, testIndices = deeplabcut.mergeandsplit(
        path_config_file, trainindex=0, uniform=True
    )

    print("Creating two identical splits...")
    deeplabcut.create_training_dataset(
        path_config_file,
        Shuffles=[4, 5],
        trainIndices=[trainIndices, trainIndices],
        testIndices=[testIndices, testIndices],
    )

    print("ALL DONE!!! - default cases are functional.")
