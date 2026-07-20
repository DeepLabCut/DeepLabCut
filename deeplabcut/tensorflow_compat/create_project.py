#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Frozen TensorFlow-specific create_project helpers.

NOT actively maintained. Exists only as a reference for legacy TF support.
The functions in this module are called from the ``if engine == Engine.TF:``
branches in ``create_project``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import yaml
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    MODELOPTIONS,
    download_huggingface_model,
)

import deeplabcut
from deeplabcut.core.config import ProjectConfig, write_config
from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset.metadata import (
    DataSplit,
    ShuffleMetadata,
    TrainingDatasetMetadata,
)
from deeplabcut.utils import auxiliaryfunctions


def _MakeTrain_pose_yaml(itemstochange, saveasconfigfile, defaultconfigfile):
    raw = Path(defaultconfigfile).open().read()
    docs = []
    for raw_doc in raw.split("\n---"):
        try:
            docs.append(yaml.load(raw_doc, Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]
    docs[0]["max_input_size"] = 1500
    write_config(saveasconfigfile, docs[0])
    return docs[0]


def _UpdateTrain_pose_yaml(dict_train, dict2change, saveasfile):
    for key in dict2change.keys():
        dict_train[key] = dict2change[key]
    auxiliaryfunctions.write_plainconfig(saveasfile, dict_train)


def _MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]
    dict_test["scoremap_dir"] = "test"
    dict_test["global_scale"] = 1.0
    auxiliaryfunctions.write_plainconfig(saveasfile, dict_test)


def _tf_create_pretrained_project(
    project: str,
    experimenter: str,
    videos: list[str] | None,
    model: str | None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    video_extensions: str | Sequence[str] | None = None,
    analyzevideo: bool = True,
    filtered: bool = True,
    createlabeledvideo: bool = True,
    trainFraction: float | None = None,
    *,
    engine=Engine.TF,
) -> tuple[str, str]:
    """Create a pretrained project using the TensorFlow engine.

    This is the frozen TF-specific backing implementation originally defined
    as ``create_pretrained_project_tensorflow`` in
    ``deeplabcut.create_project.modelzoo``.
    """
    if not model:
        model = "full_human"

    if model not in MODELOPTIONS:
        return "N/A", "N/A"

    cwd = Path.cwd()

    cfg = deeplabcut.create_new_project(
        project,
        experimenter,
        videos,
        working_directory,
        copy_videos,
        video_extensions=video_extensions,
    )
    if trainFraction is not None:
        ProjectConfig.from_yaml(cfg).update(
            TrainingFraction=[trainFraction],
        ).to_yaml(cfg, log_changes=True, mark_clean=True)

    config = auxiliaryfunctions.read_config(cfg)
    if model == "full_human":
        config["bodyparts"] = [
            "ankle1",
            "knee1",
            "hip1",
            "hip2",
            "knee2",
            "ankle2",
            "wrist1",
            "elbow1",
            "shoulder1",
            "shoulder2",
            "elbow2",
            "wrist2",
            "chin",
            "forehead",
        ]
        config["skeleton"] = [
            ["ankle1", "knee1"],
            ["ankle2", "knee2"],
            ["knee1", "hip1"],
            ["knee2", "hip2"],
            ["hip1", "hip2"],
            ["shoulder1", "shoulder2"],
            ["shoulder1", "hip1"],
            ["shoulder2", "hip2"],
            ["shoulder1", "elbow1"],
            ["shoulder2", "elbow2"],
            ["chin", "forehead"],
            ["elbow1", "wrist1"],
            ["elbow2", "wrist2"],
        ]
        config["default_net_type"] = "resnet_101"

    auxiliaryfunctions.write_config(cfg, config)
    config = auxiliaryfunctions.read_config(cfg)

    train_dir = (
        Path(config["project_path"])
        / str(
            auxiliaryfunctions.get_model_folder(
                trainFraction=config["TrainingFraction"][0],
                shuffle=1,
                cfg=config,
            )
        )
        / "train"
    )
    test_dir = (
        Path(config["project_path"])
        / str(
            auxiliaryfunctions.get_model_folder(
                trainFraction=config["TrainingFraction"][0],
                shuffle=1,
                cfg=config,
            )
        )
        / "test"
    )

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    modelfoldername = auxiliaryfunctions.get_model_folder(
        trainFraction=config["TrainingFraction"][0],
        shuffle=1,
        cfg=config,
    )
    path_train_config = str(Path(config["project_path"]) / Path(modelfoldername) / "train" / "pose_cfg.yaml")
    path_test_config = str(Path(config["project_path"]) / Path(modelfoldername) / "test" / "pose_cfg.yaml")

    print("Downloading weights...")
    download_huggingface_model(model, train_dir)

    pose_cfg = deeplabcut.auxiliaryfunctions.read_plainconfig(path_train_config)
    pose_cfg["dataset_type"] = "imgaug"
    print(path_train_config)

    dict_ = {
        "default_net_type": pose_cfg["net_type"],
        "default_augmenter": pose_cfg["dataset_type"],
        "bodyparts": pose_cfg["all_joints_names"],
        "dotsize": 6,
    }
    ProjectConfig.from_yaml(cfg).update(dict_).to_yaml(cfg, log_changes=True, mark_clean=True)

    snapshotname = [p.name for p in Path(train_dir).iterdir() if ".meta" in p.name][0].split(".meta")[0]
    dict2change = {
        "init_weights": str(Path(train_dir) / snapshotname),
        "project_path": str(config["project_path"]),
    }

    _UpdateTrain_pose_yaml(pose_cfg, dict2change, path_train_config)
    keys2save = [
        "dataset",
        "dataset_type",
        "num_joints",
        "all_joints",
        "all_joints_names",
        "net_type",
        "init_weights",
        "global_scale",
        "location_refinement",
        "locref_stdev",
    ]
    _MakeTest_pose_yaml(pose_cfg, keys2save, path_test_config)

    # Create metadata
    metadata = TrainingDatasetMetadata.create(config)
    new_shuffle = ShuffleMetadata(
        name=modelfoldername.name,
        train_fraction=config["TrainingFraction"][0],
        index=1,
        engine=Engine.TF,
        split=DataSplit(train_indices=(), test_indices=()),
    )
    metadata = metadata.add(new_shuffle)
    metadata.save()

    # Process videos
    cfg_path = str(cfg)
    video_dir = Path(cfg_path).parent / "videos"

    if analyzevideo:
        print("Analyzing video...")
        deeplabcut.analyze_videos(
            cfg_path,
            [video_dir],
            video_extensions=video_extensions,
            save_as_csv=True,
        )

    if createlabeledvideo:
        if filtered:
            deeplabcut.filterpredictions(cfg_path, [video_dir], video_extensions)

        print("Plotting results...")
        deeplabcut.create_labeled_video(
            cfg_path,
            [video_dir],
            video_extensions,
            draw_skeleton=True,
            filtered=filtered,
        )
        deeplabcut.plot_trajectories(
            cfg_path,
            [video_dir],
            video_extensions,
            filtered=filtered,
        )

    os.chdir(cwd)
    return cfg, path_train_config


# ---------------------------------------------------------------------------
# Utility: TF scorer name builder (extracted from auxiliaryfunctions)
# ---------------------------------------------------------------------------


def _tf_get_scorer_name(
    cfg: dict,
    shuffle: int,
    trainFraction: float,
    engine,
    modelprefix: str = "",
    trainingsiterations: str = "unknown",
) -> tuple[str, str]:
    """Build scorer name strings for TensorFlow projects.

    Returns:
        A tuple of ``(scorer, scorer_legacy)``.
    """
    from pathlib import Path

    from deeplabcut.utils.auxiliaryfunctions import (
        get_model_folder,
        get_snapshot_index_for_scorer,
        get_snapshots_from_folder,
        read_plainconfig,
    )

    Task = cfg["Task"]
    date = cfg["date"]

    if trainingsiterations == "unknown":
        snapshotindex = get_snapshot_index_for_scorer("snapshotindex", cfg["snapshotindex"])
        model_folder = get_model_folder(
            trainFraction,
            shuffle,
            cfg,
            engine=engine,
            modelprefix=modelprefix,
        )
        train_folder = Path(cfg["project_path"]) / model_folder / "train"
        snapshot_names = get_snapshots_from_folder(train_folder)
        snapshot_name = snapshot_names[snapshotindex]
        trainingsiterations = Path(snapshot_name).parts[-1].split("-")[-1]

    dlc_cfg = read_plainconfig(
        str(
            Path(cfg["project_path"])
            / get_model_folder(
                trainFraction,
                shuffle,
                cfg,
                engine=engine,
                modelprefix=modelprefix,
            )
            / "train"
            / engine.pose_cfg_name
        )
    )
    if "resnet" in dlc_cfg["net_type"]:
        if dlc_cfg.get("multi_stage", False):
            netname = "dlcrnetms5"
        else:
            netname = dlc_cfg["net_type"].replace("_", "")
    elif "mobilenet" in dlc_cfg["net_type"]:
        netname = "mobnet_" + str(int(float(dlc_cfg["net_type"].split("_")[-1]) * 100))
    elif "efficientnet" in dlc_cfg["net_type"]:
        netname = "effnet_" + dlc_cfg["net_type"].split("-")[1]
    else:
        raise ValueError(f"Failed to abbreviate network name: {dlc_cfg['net_type']}")

    scorer = "DLC_" + netname + "_" + Task + str(date) + "shuffle" + str(shuffle) + "_" + str(trainingsiterations)
    scorer_legacy = scorer.replace("DLC", "DeepCut")
    return scorer, scorer_legacy
