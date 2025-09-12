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
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, for CI/CD on Windows

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import VideoReader


MODELS = ["dlcrnet_ms5", "dlcr101_ms5", "efficientnet-b0"]


N_ITER = 5
TESTTRACKER = "ellipse"

USE_SHELVE = False  # random.choice([True, False])

if __name__ == "__main__":
    TASK = "multi_mouse"
    SCORER = "dlc_team"
    NUM_FRAMES = 5
    TRAIN_SIZE = 0.8
    ENGINE = Engine.TF

    # NET = "dlcr101_ms5"
    NET = "dlcrnet_ms5"

    # Always test a different model from list above
    NET = random.choice(MODELS)

    basepath = os.path.dirname(os.path.realpath(__file__))
    DESTFOLDER = basepath

    video = "m3v1mp4"
    video_path = os.path.join(
        basepath, "openfield-Pranav-2018-10-30", "videos", video + ".mp4"
    )

    print("Creating project...")
    config_path = deeplabcut.create_new_project(
        TASK, SCORER, [video_path], copy_videos=True, multianimal=True
    )

    print("Project created.")

    print("Editing config...")
    cfg = auxiliaryfunctions.edit_config(
        config_path,
        {
            "numframes2pick": NUM_FRAMES,
            "TrainingFraction": [TRAIN_SIZE],
            "identity": True,
            "uniquebodyparts": ["corner1", "corner2"],
        },
    )
    print("Config edited.")

    print("Extracting frames...")
    deeplabcut.extract_frames(config_path, mode="automatic", userfeedback=False)
    print("Frames extracted.")

    print("Creating artificial data...")
    rel_folder = os.path.join("labeled-data", os.path.splitext(video)[0])
    image_folder = os.path.join(cfg["project_path"], rel_folder)
    n_animals = len(cfg["individuals"])
    (
        animals,
        bodyparts_single,
        bodyparts_multi,
    ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)
    animals_id = [i for i in range(n_animals) for _ in bodyparts_multi] + [
        n_animals
    ] * len(bodyparts_single)
    map_ = dict(zip(range(len(animals)), animals))
    individuals = [map_[ind] for ind in animals_id for _ in range(2)]
    scorer = [SCORER] * len(individuals)
    coords = ["x", "y"] * len(animals_id)
    bodyparts = [
        bp for _ in range(n_animals) for bp in bodyparts_multi for _ in range(2)
    ]
    bodyparts += [bp for bp in bodyparts_single for _ in range(2)]
    columns = pd.MultiIndex.from_arrays(
        [scorer, individuals, bodyparts, coords],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    index = [
        os.path.join(rel_folder, image)
        for image in auxiliaryfunctions.grab_files_in_folder(image_folder, "png")
    ]
    fake_data = np.tile(
        np.repeat(50 * np.arange(len(animals_id)) + 50, 2), (len(index), 1)
    )
    df = pd.DataFrame(fake_data, index=index, columns=columns)
    output_path = os.path.join(image_folder, f"CollectedData_{SCORER}.csv")
    df.to_csv(output_path)
    df.to_hdf(
        output_path.replace("csv", "h5"), "df_with_missing", format="table", mode="w"
    )
    print("Artificial data created.")

    print("Checking labels...")
    deeplabcut.check_labels(config_path, draw_skeleton=False)
    print("Labels checked.")

    print("Creating train dataset...")
    deeplabcut.create_multianimaltraining_dataset(
        config_path,
        net_type=NET,
        crop_size=(200, 200),
        engine=ENGINE,
    )
    print("Train dataset created.")

    # Check the training image paths are correctly stored as arrays of strings
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    datafile, _ = auxiliaryfunctions.get_data_and_metadata_filenames(
        trainingsetfolder,
        0.8,
        1,
        cfg,
    )
    datafile = datafile.split(".mat")[0] + ".pickle"
    with open(os.path.join(cfg["project_path"], datafile), "rb") as f:
        pickledata = pickle.load(f)
    num_images = len(pickledata)
    assert all(len(pickledata[i]["image"]) == 3 for i in range(num_images))

    print("Editing pose config...")
    model_folder = auxiliaryfunctions.get_model_folder(
        TRAIN_SIZE, 1, cfg, engine=ENGINE, modelprefix=cfg["project_path"]
    )
    pose_config_path = os.path.join(model_folder, "train", "pose_cfg.yaml")
    edits = {
        "global_scale": 0.5,
        "batch_size": 1,
        "save_iters": N_ITER,
        "display_iters": N_ITER // 2,
        "crop_size": [200, 200],
        # "multi_step": [[0.001, N_ITER]],
    }
    deeplabcut.auxiliaryfunctions.edit_config(pose_config_path, edits)
    print("Pose config edited.")

    print("Training network...")
    deeplabcut.train_network(config_path, maxiters=N_ITER)
    print("Network trained.")

    print("Evaluating network...")
    deeplabcut.evaluate_network(
        config_path, plotting=True, per_keypoint_evaluation=True
    )

    print("Network evaluated....")

    print("Extracting maps...")
    deeplabcut.extract_save_all_maps(config_path, Indices=[0, 1, 2])

    new_video_path = deeplabcut.ShortenVideo(
        video_path,
        start="00:00:00",
        stop="00:00:01",
        outsuffix="short",
        outpath=os.path.join(cfg["project_path"], "videos"),
    )

    print("Analyzing video...")
    try:
        deeplabcut.analyze_videos(
            config_path,
            [new_video_path],
            "mp4",
            robust_nframes=True,
            allow_growth=True,
            use_shelve=USE_SHELVE,
        )
        print("Video analyzed.")
    except ValueError:
        pass

    print("Create video with all detections...")
    scorer, _ = auxiliaryfunctions.get_scorer_name(cfg, 1, TRAIN_SIZE)

    deeplabcut.create_video_with_all_detections(
        config_path, [new_video_path], shuffle=1, displayedbodyparts=["bodypart1"]
    )

    print("Video created.")

    print("Convert detections to tracklets...")
    deeplabcut.convert_detections2tracklets(
        config_path, [new_video_path], "mp4", track_method=TESTTRACKER
    )
    print("Tracklets created...")
    h5path = os.path.splitext(new_video_path)[0] + scorer + "_el.h5"
    try:
        deeplabcut.stitch_tracklets(
            config_path,
            [new_video_path],
            "mp4",
            output_name=h5path,
            track_method=TESTTRACKER,
        )
    except ValueError:
        # Sometimes tracks cannot be reconstructed as test data are randomly
        # created; when this happens, we generate a fake h5 data file.
        individuals = [map_[ind] for ind in animals_id for _ in range(3)]
        scorer = [SCORER] * len(individuals)
        coords = ["x", "y", "likelihood"] * len(animals_id)
        bodyparts = [
            bp for _ in range(n_animals) for bp in bodyparts_multi for _ in range(3)
        ]
        bodyparts += [bp for bp in bodyparts_single for _ in range(3)]
        columns = pd.MultiIndex.from_arrays(
            [scorer, individuals, bodyparts, coords],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        vid = VideoReader(new_video_path)
        fake_data = np.ones((len(vid), columns.shape[0]))
        df = pd.DataFrame(fake_data, columns=columns)
        df.to_hdf(h5path, key="data")

    print("Plotting trajectories...")
    deeplabcut.plot_trajectories(
        config_path, [new_video_path], "mp4", track_method=TESTTRACKER
    )
    print("Trajectory plotted.")

    print("Creating labeled video...")
    deeplabcut.create_labeled_video(
        config_path,
        [new_video_path],
        "mp4",
        save_frames=False,
        color_by="individual",
        track_method=TESTTRACKER,
    )
    print("Labeled video created.")

    print("Filtering predictions...")
    deeplabcut.filterpredictions(
        config_path, [new_video_path], "mp4", track_method=TESTTRACKER
    )
    print("Predictions filtered.")

    print("Extracting outlier frames...")
    deeplabcut.extract_outlier_frames(
        config_path, [new_video_path], "mp4", automatic=True, track_method=TESTTRACKER
    )
    print("Outlier frames extracted.")

    vname = Path(new_video_path).stem

    file = os.path.join(
        cfg["project_path"],
        "labeled-data",
        vname,
        "machinelabels-iter" + str(cfg["iteration"]) + ".h5",
    )

    print("RELABELING")
    DF = pd.read_hdf(file, "df_with_missing")
    DLCscorer = np.unique(DF.columns.get_level_values(0))[0]
    DF.columns = DF.columns.set_levels([scorer.replace(DLCscorer, SCORER)], level=0)
    DF = DF.drop("likelihood", axis=1, level=3)
    DF.to_csv(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            vname,
            "CollectedData_" + SCORER + ".csv",
        )
    )
    DF.to_hdf(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            vname,
            "CollectedData_" + SCORER + ".h5",
        ),
        "df_with_missing",
    )

    print("MERGING")
    deeplabcut.merge_datasets(config_path)  # iteration + 1

    print("CREATING TRAININGSET updated training set")
    deeplabcut.create_training_dataset(config_path, net_type=NET, engine=ENGINE)

    print("Training network...")
    deeplabcut.train_network(config_path, maxiters=N_ITER)
    print("Network trained.")

    print("Evaluating network...")
    deeplabcut.evaluate_network(
        config_path, plotting=True, per_keypoint_evaluation=True
    )

    print("Network evaluated....")

    print("Analyzing video with auto_track....")
    deeplabcut.analyze_videos(
        config_path,
        [new_video_path],
        save_as_csv=True,
        destfolder=DESTFOLDER,
        cropping=[0, 50, 0, 50],
        allow_growth=True,
        use_shelve=USE_SHELVE,
        auto_track=False,
    )

    print("Export model...")
    deeplabcut.export_model(config_path, shuffle=1, make_tar=False)

    print("Merging datasets...")
    trainIndices, testIndices = deeplabcut.mergeandsplit(
        config_path, trainindex=0, uniform=True
    )

    print("Creating two identical splits...")
    deeplabcut.create_multianimaltraining_dataset(
        config_path,
        Shuffles=[4, 5],
        net_type=NET,
        trainIndices=[trainIndices, trainIndices],
        testIndices=[testIndices, testIndices],
        engine=ENGINE,
    )

    print("ALL DONE!!! - default multianimal cases are functional.")
