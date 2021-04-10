#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities (creating project ,training, evaluating, outlierextraction, retraining...) in an automatic way.
For that purpose, it trains ResNet and MobileNet briefly on a "fake" dataset.

It should take about 4:15 minutes to run this in a CPU. (incl. downloading the ResNet + MobileNet weights)

It produces nothing of interest scientifically.
"""
import os
os.environ["DLClight"] = "True"
import deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np


def Cuttrainingschedule(
    path_config_file, shuffle, trainingsetindex=0, initweights="imagenet", lastvalue=10
):
    cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
    posefile = os.path.join(
        cfg["project_path"],
        "dlc-models/iteration-"
        + str(cfg["iteration"])
        + "/"
        + cfg["Task"]
        + cfg["date"]
        + "-trainset"
        + str(int(cfg["TrainingFraction"][trainingsetindex] * 100))
        + "shuffle"
        + str(shuffle),
        "train/pose_cfg.yaml",
    )

    edits = {
        "save_iters": lastvalue,
        "display_iters": 1,
        "multi_step": [[0.001, lastvalue]],
        "intermediate_supervision": False,
    }

    if initweights == "previteration":
        edits["init_weights"] = os.path.join(
            cfg["project_path"],
            "dlc-models/iteration-"
            + str(cfg["iteration"] - 1)
            + "/"
            + cfg["Task"]
            + cfg["date"]
            + "-trainset"
            + str(int(cfg["TrainingFraction"][trainingsetindex] * 100))
            + "shuffle"
            + str(shuffle),
            "train/snapshot-" + str(lastvalue),
        )

    print("CHANGING training parameters to end quickly!")
    DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)
    return


if __name__ == "__main__":
    task = "TEST-multipleNets"  # Enter the name of your experiment Task
    scorer = "Alex"  # Enter the name of the experimenter/labeler
    print("Imported DLC!")
    basepath = os.path.dirname(os.path.realpath(__file__))
    videoname = "reachingvideo1"
    video = [
        os.path.join(
            basepath, "Reaching-Mackenzie-2018-08-30", "videos", videoname + ".avi"
        )
    ]

    # to test destination folder:
    dfolder = os.path.join(basepath, "OUT")
    deeplabcut.auxiliaryfunctions.attempttomakefolder(dfolder)

    # dfolder=None
    augmenter_type = "tensorpack"  # imgaug'

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

    stoptrain = 5  # 0
    keepdeconvweights = True

    print("Plot labels...")
    deeplabcut.check_labels(path_config_file)
    for shuffle, net_type in enumerate(
        ["mobilenet_v2_0.35", "resnet_50"]
    ):  #'mobilenet_v2_1.0']): # 'resnet_50']):
        """
        if shuffle==0:
            keepdeconvweights=True
        else:
            keepdeconvweights=False
        """
        print("CREATING TRAININGSET", net_type)
        if "resnet_50" == net_type:  # this tests the default condition...
            deeplabcut.create_training_dataset(
                path_config_file, Shuffles=[shuffle], augmenter_type=augmenter_type
            )
        else:
            deeplabcut.create_training_dataset(
                path_config_file,
                Shuffles=[shuffle],
                net_type=net_type,
                augmenter_type=augmenter_type,
            )
        Cuttrainingschedule(path_config_file, shuffle, lastvalue=stoptrain)

        print("TRAIN")
        deeplabcut.train_network(path_config_file, shuffle=shuffle)

        print("EVALUATE")
        deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle], plotting=True)

        print("CREATE A SHORT VIDEO AND ANALYZE")
        if shuffle == 0:
            # Make super short video (so the analysis is quick!)
            newvideo = deeplabcut.ShortenVideo(
                video[0],
                start="00:00:00",
                stop="00:00:01",
                outsuffix="short",
                outpath=os.path.join(cfg["project_path"], "videos"),
            )
            vname = Path(newvideo).stem

        deeplabcut.analyze_videos(
            path_config_file,
            [newvideo],
            shuffle=shuffle,
            save_as_csv=True,
            destfolder=dfolder,
            videotype="avi",
        )

        print("CREATE VIDEO")
        deeplabcut.create_labeled_video(
            path_config_file,
            [newvideo],
            shuffle=shuffle,
            destfolder=dfolder,
            videotype="avi",
        )

        print("Making plots")
        deeplabcut.plot_trajectories(
            path_config_file,
            [newvideo],
            shuffle=shuffle,
            destfolder=dfolder,
            videotype="avi",
        )

        print("EXTRACT OUTLIERS")
        deeplabcut.extract_outlier_frames(
            path_config_file,
            [newvideo],
            shuffle=shuffle,
            outlieralgorithm="jump",
            epsilon=0,
            automatic=True,
            destfolder=dfolder,
            videotype="avi",
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
        DF.columns.set_levels(
            [scorer.replace(DLCscorer, scorer)], level=0, inplace=True
        )
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
        deeplabcut.merge_datasets(path_config_file)

        print("CREATING TRAININGSET")
        deeplabcut.create_training_dataset(
            path_config_file, Shuffles=[shuffle], net_type=net_type
        )
        Cuttrainingschedule(
            path_config_file, shuffle, lastvalue=stoptrain, initweights="previteration"
        )

        print("TRAINING from previous snapshot!!!!!")
        deeplabcut.train_network(
            path_config_file, shuffle=shuffle, keepdeconvweights=keepdeconvweights
        )

        print("ANALYZING some individual frames")
        deeplabcut.analyze_time_lapse_frames(
            path_config_file,
            os.path.join(cfg["project_path"], "labeled-data/reachingvideo1/"),
            shuffle=shuffle,
        )

    print("ALL DONE!!! - Mobilnets and ResNets are good!")
