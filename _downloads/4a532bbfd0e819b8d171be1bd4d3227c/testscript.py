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
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

It should take about 3:30 minutes to run this in a CPU.
It should take about 1:30 minutes on a GPU (incl. downloading the ResNet weights)

It produces nothing of interest scientifically.
"""
import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.utils import auxiliaryfunctions

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, for CI/CD on Windows

USE_SHELVE = random.choice([True, False])
MODELS = ["resnet_50", "efficientnet-b0"]


if __name__ == "__main__":
    task = "TEST"  # Enter the name of your experiment Task
    scorer = "Alex"  # Enter the name of the experimenter/labeler
    engine = Engine.TF

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
    DESTFOLDER = basepath

    DESTFOLDER = None
    NET = random.choice(MODELS)

    augmenter_type = "default"  # = imgaug!!
    augmenter_type2 = "scalecrop"

    if platform.system() == "Darwin" or platform.system() == "Windows":
        print("On Windows/OSX tensorpack is not tested by default.")
        augmenter_type3 = "imgaug"
    else:
        augmenter_type3 = "tensorpack"  # Does not work on WINDOWS

    N_ITER = 6
    SAVE_ITER = 3

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
    frames = [fn for fn in frames if fn.endswith(".png")]
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
        key="df_with_missing",
        format="table",
        mode="w",
    )

    print("Plot labels...")

    deeplabcut.check_labels(path_config_file)

    print("CREATING TRAININGSET")
    deeplabcut.create_training_dataset(
        path_config_file, net_type=NET, augmenter_type=augmenter_type, engine=engine,
    )

    # Check the training image paths are correctly stored as arrays of strings
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    datafile, _ = auxiliaryfunctions.get_data_and_metadata_filenames(
        trainingsetfolder,
        0.8,
        1,
        cfg,
    )
    mlab = sio.loadmat(os.path.join(cfg["project_path"], datafile))["dataset"]
    num_images = mlab.shape[1]
    for i in range(num_images):
        imgpath = mlab[0, i][0][0]
        assert len(imgpath) == 3
        assert imgpath.dtype.char == "U"

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
    DLC_config["save_iters"] = SAVE_ITER
    DLC_config["display_iters"] = 2

    print("CHANGING training parameters to end quickly!")
    deeplabcut.auxiliaryfunctions.write_plainconfig(posefile, DLC_config)

    print("TRAIN")
    deeplabcut.train_network(path_config_file, maxiters=N_ITER)

    print("EVALUATE")
    deeplabcut.evaluate_network(
        path_config_file,
        plotting=True,
        per_keypoint_evaluation=True,
        snapshots_to_evaluate=[
            "snapshot-3",
            "snapshot-5",
            "snapshot-6",
        ],  # snapshot-5 intentionally missing :)
    )
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
        from moviepy.editor import VideoClip, VideoFileClip

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
        destfolder=DESTFOLDER,
        dynamic=(True, 0.1, 5),
    )

    print("analyze again...")
    deeplabcut.analyze_videos(
        path_config_file, [newvideo], save_as_csv=True, destfolder=DESTFOLDER
    )

    print("CREATE VIDEO")
    successful = deeplabcut.create_labeled_video(
        path_config_file, [newvideo], destfolder=DESTFOLDER, save_frames=True
    )
    assert all(successful), f"Failed to create a labeled video!"

    print("Making plots")
    deeplabcut.plot_trajectories(path_config_file, [newvideo], destfolder=DESTFOLDER)

    print("EXTRACT OUTLIERS")
    deeplabcut.extract_outlier_frames(
        path_config_file,
        [newvideo],
        outlieralgorithm="jump",
        epsilon=0,
        automatic=True,
        destfolder=DESTFOLDER,
    )

    deeplabcut.extract_outlier_frames(
        path_config_file,
        [newvideo],
        outlieralgorithm="fitting",
        automatic=True,
        destfolder=DESTFOLDER,
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
    DF.columns = DF.columns.set_levels([scorer.replace(DLCscorer, scorer)], level=0)
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
        key="df_with_missing",
    )

    print("MERGING")
    deeplabcut.merge_datasets(path_config_file)  # iteration + 1

    print("CREATING TRAININGSET")
    deeplabcut.create_training_dataset(
        path_config_file, net_type=NET, augmenter_type=augmenter_type2, engine=engine
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
    DLC_config["save_iters"] = SAVE_ITER
    DLC_config["display_iters"] = 1

    print("CHANGING training parameters to end quickly!")
    deeplabcut.auxiliaryfunctions.write_config(posefile, DLC_config)

    print("TRAIN")
    deeplabcut.train_network(path_config_file, maxiters=N_ITER)

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
        from moviepy.editor import VideoClip, VideoFileClip

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
        destfolder=DESTFOLDER,
        cropping=[0, 50, 0, 50],
        allow_growth=True,
        use_shelve=USE_SHELVE,
    )

    print("Extracting skeleton distances, filter and plot filtered output")
    deeplabcut.analyzeskeleton(
        path_config_file, [newvideo2], save_as_csv=True, destfolder=DESTFOLDER
    )
    deeplabcut.filterpredictions(path_config_file, [newvideo2])

    successful = deeplabcut.create_labeled_video(
        path_config_file,
        [newvideo2],
        destfolder=DESTFOLDER,
        displaycropped=True,
        filtered=True,
    )
    assert all(successful), f"Failed to create a labeled video!"

    print("Creating a Johansson video!")
    successful = deeplabcut.create_labeled_video(
        path_config_file, [newvideo2], destfolder=DESTFOLDER, keypoints_only=True
    )
    assert all(successful), f"Failed to create a labeled video!"

    deeplabcut.plot_trajectories(
        path_config_file, [newvideo2], destfolder=DESTFOLDER, filtered=True
    )

    print("ALL DONE!!! - default cases without Tensorpack loader are functional.")

    print("CREATING TRAININGSET for shuffle 2")
    print("will be used for 3D testscript...")
    # TENSORPACK could fail in WINDOWS...
    deeplabcut.create_training_dataset(
        path_config_file,
        Shuffles=[2],
        net_type=NET,
        augmenter_type=augmenter_type3,
        engine=engine,
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
    updated_max_iters = 10
    DLC_config["save_iters"] = updated_max_iters
    DLC_config["display_iters"] = 2

    print("CHANGING training parameters to end quickly!")
    deeplabcut.auxiliaryfunctions.write_plainconfig(posefile, DLC_config)

    print("TRAINING shuffle 2, with smaller allocated memory")
    deeplabcut.train_network(
        path_config_file, shuffle=2, allow_growth=True, maxiters=updated_max_iters
    )

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
        engine=engine,
    )

    print("ALL DONE!!! - default cases are functional.")
