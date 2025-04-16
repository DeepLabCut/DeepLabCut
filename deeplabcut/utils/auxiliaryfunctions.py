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
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
from __future__ import annotations

import os
import typing
import pickle
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import ruamel.yaml.representer
import yaml
from ruamel.yaml import YAML

from deeplabcut.core.engine import Engine
from deeplabcut.core.trackingutils import TRACK_METHODS
from deeplabcut.utils import auxfun_videos, auxfun_multianimal


def create_config_template(multianimal=False):
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    if multianimal:
        yaml_str = """\
# Project definitions (do not edit)
Task:
scorer:
date:
multianimalproject:
identity:
\n
# Project path (change when moving around)
project_path:
\n
# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
engine: pytorch
\n
# Annotation data set configuration (and individual video cropping parameters)
video_sets:
individuals:
uniquebodyparts:
multianimalbodyparts:
bodyparts:
\n
# Fraction of video to start/stop when extracting frames for labeling/refinement
start:
stop:
numframes2pick:
\n
# Plotting configuration
skeleton:
skeleton_color:
pcutoff:
dotsize:
alphavalue:
colormap:
\n
# Training,Evaluation and Analysis configuration
TrainingFraction:
iteration:
default_net_type:
default_augmenter:
default_track_method:
snapshotindex:
detector_snapshotindex:
batch_size:
\n
# Cropping Parameters (for analysis and outlier frame detection)
cropping:
#if cropping is true for analysis, then set the values here:
x1:
x2:
y1:
y2:
\n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
corner2move2:
move2corner:
\n
# Conversion tables to fine-tune SuperAnimal weights
SuperAnimalConversionTables:
        """
    else:
        yaml_str = """\
# Project definitions (do not edit)
Task:
scorer:
date:
multianimalproject:
identity:
\n
# Project path (change when moving around)
project_path:
\n
# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
engine: pytorch
\n
# Annotation data set configuration (and individual video cropping parameters)
video_sets:
bodyparts:
\n
# Fraction of video to start/stop when extracting frames for labeling/refinement
start:
stop:
numframes2pick:
\n
# Plotting configuration
skeleton:
skeleton_color:
pcutoff:
dotsize:
alphavalue:
colormap:
\n
# Training,Evaluation and Analysis configuration
TrainingFraction:
iteration:
default_net_type:
default_augmenter:
snapshotindex:
detector_snapshotindex:
batch_size:
detector_batch_size:
\n
# Cropping Parameters (for analysis and outlier frame detection)
cropping:
#if cropping is true for analysis, then set the values here:
x1:
x2:
y1:
y2:
\n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
corner2move2:
move2corner:
\n
# Conversion tables to fine-tune SuperAnimal weights
SuperAnimalConversionTables:
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile


def create_config_template_3d():
    """
    Creates a template for config.yaml file for 3d project. This specific order is preserved while saving as yaml file.
    """
    yaml_str = """\
# Project definitions (do not edit)
Task:
scorer:
date:
\n
# Project path (change when moving around)
project_path:
\n
# Plotting configuration
skeleton: # Note that the pairs must be defined, as you want them linked!
skeleton_color:
pcutoff:
colormap:
dotsize:
alphaValue:
markerType:
markerColor:
\n
# Number of cameras, camera names, path of the config files, shuffle index and trainingsetindex used to analyze videos:
num_cameras:
camera_names:
scorername_3d: # Enter the scorer name for the 3D output
    """
    ruamelFile_3d = YAML()
    cfg_file_3d = ruamelFile_3d.load(yaml_str)
    return cfg_file_3d, ruamelFile_3d


def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = str(Path(configname).parent.resolve())

                if cfg.get("engine") is None:
                    cfg["engine"] = Engine.TF.aliases[0]
                    write_config(configname, cfg)

                if cfg.get("detector_snapshotindex") is None:
                    cfg["detector_snapshotindex"] = -1

                if cfg.get("detector_batch_size") is None:
                    cfg["detector_batch_size"] = 1

                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                    err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            f"Config file at {path} not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg


def write_config(configname, cfg):
    """
    Write structured config file.
    """
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template(
            cfg.get("multianimalproject", False)
        )
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if not "skeleton" in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        ruamelFile.dump(cfg_file, cf)


def edit_config(configname, edits, output_name=""):
    """
    Convenience function to edit and save a config file from a dictionary.

    Parameters
    ----------
    configname : string
        String containing the full path of the config file in the project.
    edits : dict
        Key–value pairs to edit in config
    output_name : string, optional (default='')
        Overwrite the original config.yaml by default.
        If passed in though, new filename of the edited config.

    Examples
    --------
    config_path = 'my_stellar_lab/dlc/config.yaml'

    edits = {'numframes2pick': 5,
             'trainingFraction': [0.5, 0.8],
             'skeleton': [['a', 'b'], ['b', 'c']]}

    deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)
    """
    cfg = read_plainconfig(configname)
    for key, value in edits.items():
        cfg[key] = value
    if not output_name:
        output_name = configname
    try:
        write_plainconfig(output_name, cfg)
    except ruamel.yaml.representer.RepresenterError:
        warnings.warn(
            "Some edits could not be written. "
            "The configuration file will be left unchanged."
        )
        for key in edits:
            cfg.pop(key)
        write_plainconfig(output_name, cfg)
    return cfg


def get_bodyparts(cfg: dict) -> typing.List[str]:
    """
    Args:
        cfg: a project configuration file

    Returns: bodyparts listed in the project (does not include the unique_bodyparts entry)
    """
    if cfg.get("multianimalproject", False):
        (
            _,
            _,
            multianimal_bodyparts,
        ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)
        return multianimal_bodyparts

    return cfg["bodyparts"]


def get_unique_bodyparts(cfg: dict) -> typing.List[str]:
    """
    Args:
        cfg: a project configuration file

    Returns: all unique bodyparts listed in the project
    """
    if cfg.get("multianimalproject", False):
        (
            _,
            unique_bodyparts,
            _,
        ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)
        return unique_bodyparts

    return []


def write_config_3d(configname, cfg):
    """
    Write structured 3D config file.
    """
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template_3d()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]
        ruamelFile.dump(cfg_file, cf)


def write_config_3d_template(projconfigfile, cfg_file_3d, ruamelFile_3d):
    with open(projconfigfile, "w") as cf:
        ruamelFile_3d.dump(cfg_file_3d, cf)


def read_plainconfig(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(
            f"Config {configname} is not found. Please make sure that the file exists."
        )
    with open(configname) as file:
        return YAML().load(file)


def write_plainconfig(configname, cfg):
    with open(configname, "w") as file:
        YAML().dump(cfg, file)


def attempt_to_make_folder(foldername, recursive=False):
    """Attempts to create a folder with specified name. Does nothing if it already exists."""
    try:
        os.path.isdir(foldername)
    except TypeError:  # https://www.python.org/dev/peps/pep-0519/
        foldername = os.fspath(
            foldername
        )  # https://github.com/DeepLabCut/DeepLabCut/issues/105 (windows)

    if os.path.isdir(foldername):
        pass
    else:
        if recursive:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)


def read_pickle(filename):
    """Read the pickle file"""
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def write_pickle(filename, data):
    """Write the pickle file"""
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_list_of_videos(
    videos: typing.Union[typing.List[str], str],
    videotype: typing.Union[typing.List[str], str] = "",
    in_random_order: bool = True,
) -> typing.List[str]:
    """Returns list of videos of videotype "videotype" in
    folder videos or for list of videos.

    NOTE: excludes keyword videos of the form:

    *_labeled.videotype
    *_full.videotype

    Args:
        videos (list[str], str): List of video paths or a single path string. If string (or len() == 1 list of strings) is a directory,
            finds all videos whose extension matches  ``videotype`` in the directory

        videotype (list[str], str): File extension used to filter videos. Optional if ``videos`` is a list of video files,
            and filters with common video extensions if a directory is passed in.

        in_random_order (bool): Whether or not to return a shuffled list of videos.
    """
    if isinstance(videos, str):
        videos = [videos]

    if [os.path.isdir(i) for i in videos] == [True]:  # checks if input is a directory
        """
        Returns all the videos in the directory.
        """
        if not videotype:
            videotype = auxfun_videos.SUPPORTED_VIDEOS

        print("Analyzing all the videos in the directory...")
        videofolder = videos[0]

        # make list of full paths
        videos = [os.path.join(videofolder, fn) for fn in os.listdir(videofolder)]

        if in_random_order:
            from random import shuffle

            shuffle(
                videos
            )  # this is useful so multiple nets can be used to analyze simultaneously
        else:
            videos.sort()

    if isinstance(videotype, str):
        videotype = [videotype]
    if not videotype:
        videotype = auxfun_videos.SUPPORTED_VIDEOS
    # filter list of videos
    videos = [
        v
        for v in videos
        if os.path.isfile(v)
        and any(v.endswith(ext) for ext in videotype)
        and "_labeled." not in v
        and "_full." not in v
    ]

    return videos


def save_data(PredicteData, metadata, dataname, pdindex, imagenames, save_as_csv):
    """Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py"""
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    if save_as_csv:
        print("Saving csv poses!")
        DataMachine.to_csv(dataname.split(".h5")[0] + ".csv")
    DataMachine.to_hdf(dataname, key="df_with_missing", format="table", mode="w")
    with open(dataname.split(".h5")[0] + "_meta.pickle", "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def save_metadata(metadatafilename, data, trainIndices, testIndices, trainFraction):
    with open(metadatafilename, "wb") as f:
        # Pickle the 'labeled-data' dictionary using the highest protocol available.
        pickle.dump(
            [data, trainIndices, testIndices, trainFraction], f, pickle.HIGHEST_PROTOCOL
        )


def load_metadata(metadatafile):
    with open(metadatafile, "rb") as f:
        [
            trainingdata_details,
            trainIndices,
            testIndices,
            testFraction_data,
        ] = pickle.load(f)
        return trainingdata_details, trainIndices, testIndices, testFraction_data


def get_immediate_subdirectories(a_dir):
    """Get list of immediate subdirectories"""
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def grab_files_in_folder(folder, ext="", relative=True):
    """Return the paths of files with extension *ext* present in *folder*."""
    for file in os.listdir(folder):
        if file.endswith(ext):
            yield file if relative else os.path.join(folder, file)


def filter_files_by_patterns(
    folder: str | Path,
    start_patterns: set[str] | None = None,
    contain_patterns: set[str] | None = None,
    end_patterns: set[str] | None = None,
) -> List[Path]:
    """
    Filters files in a folder based on start, contain, and end patterns.

    Args:
        folder (str | Path): The folder to search for files.

        start_patterns (Set[str] | None): Patterns the filenames should start with.
            If None or empty, this pattern is not taken into account.

        contain_patterns (set[str]): Patterns the filenames should contain.
            If None or empty, this pattern is not taken into account.

        end_patterns (set[str]): Patterns the filenames should end with.
            If None or empty, this pattern is not taken into account.

    Returns:
        List[Path]: List of files that match the criteria.
    """
    folder = Path(folder)  # Ensure the folder is a Path object
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a valid directory.")

    # Filter files based on the given patterns
    matching_files = [
        file
        for file in folder.iterdir()
        if file.is_file()
        and (
            not start_patterns
            or any(file.name.startswith(start) for start in start_patterns)
        )
        and (
            not contain_patterns
            or any(contain in file.name for contain in contain_patterns)
        )
        and (not end_patterns or any(file.name.endswith(end) for end in end_patterns))
    ]

    return matching_files


def get_video_list(filename, videopath, videtype):
    """Get list of videos in a path (if filetype == all), otherwise just a specific file."""
    videos = list(grab_files_in_folder(videopath, videtype))
    if filename == "all":
        return videos
    else:
        if filename in videos:
            videos = [filename]
        else:
            videos = []
            print("Video not found!", filename)
    return videos


## Various functions to get filenames, foldernames etc. based on configuration parameters.
def get_training_set_folder(cfg: dict) -> Path:
    """Training Set folder for config file based on parameters"""
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        os.path.join("training-datasets", iterate, "UnaugmentedDataSet_" + Task + date)
    )


def get_data_and_metadata_filenames(trainingsetfolder, trainFraction, shuffle, cfg):
    # Filename for metadata and data relative to project path for corresponding parameters
    metadatafn = os.path.join(
        str(trainingsetfolder),
        "Documentation_data-"
        + cfg["Task"]
        + "_"
        + str(int(trainFraction * 100))
        + "shuffle"
        + str(shuffle)
        + ".pickle",
    )
    datafn = os.path.join(
        str(trainingsetfolder),
        cfg["Task"]
        + "_"
        + cfg["scorer"]
        + str(int(100 * trainFraction))
        + "shuffle"
        + str(shuffle)
        + ".mat",
    )

    return datafn, metadatafn


def get_model_folder(
    trainFraction: float,
    shuffle: int,
    cfg: dict,
    modelprefix: str = "",
    engine: Engine = Engine.TF,
) -> Path:
    """
    Args:
        trainFraction: the training fraction (as defined in the project configuration)
            for which to get the model folder
        shuffle: the index of the shuffle for which to get the model folder
        cfg: the project configuration
        modelprefix: The name of the folder
        engine: The engine for which we want the model folder. Defaults to `tensorflow`
            for backwards compatibility with DeepLabCut 2.X

    Returns:
        the relative path from the project root to the folder containing the model files
        for a shuffle (configuration files, snapshots, training logs, ...)
    """
    proj_id = f"{cfg['Task']}{cfg['date']}"
    return Path(
        modelprefix,
        engine.model_folder_name,
        f"iteration-{cfg['iteration']}",
        f"{proj_id}-trainset{int(trainFraction * 100)}shuffle{shuffle}",
    )


def get_evaluation_folder(
    trainFraction: float,
    shuffle: int,
    cfg: dict,
    engine: Engine | None = None,
    modelprefix: str = "",
) -> Path:
    """
    Args:
        trainFraction: the training fraction (as defined in the project configuration)
            for which to get the evaluation folder
        shuffle: the index of the shuffle for which to get the evaluation folder
        cfg: the project configuration
        engine: The engine for which we want the model folder. Defaults to None,
            which automatically gets the engine for the shuffle from the training
            dataset metadata file.
        modelprefix: The name of the folder

    Returns:
        the relative path from the project root to the folder containing the model files
        for a shuffle (configuration files, snapshots, training logs, ...)
    """
    if engine is None:
        from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine

        engine = get_shuffle_engine(
            cfg=cfg,
            trainingsetindex=cfg["TrainingFraction"].index(trainFraction),
            shuffle=shuffle,
            modelprefix=modelprefix,
        )

    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    if "eval_prefix" in cfg:
        eval_prefix = cfg["eval_prefix"]
    else:
        eval_prefix = engine.results_folder_name
    return Path(
        modelprefix,
        eval_prefix,
        iterate,
        Task
        + date
        + "-trainset"
        + str(int(trainFraction * 100))
        + "shuffle"
        + str(shuffle),
    )


def get_snapshots_from_folder(train_folder: Path) -> List[str]:
    """
    Returns an ordered list of existing snapshot names in the train folder, sorted by
    increasing training iterations.

    Raises:
        FileNotFoundError: if no snapshot_names are found in the train_folder.
    """
    snapshot_names = [
        file.stem for file in train_folder.iterdir() if "index" in file.name
    ]

    if len(snapshot_names) == 0:
        raise FileNotFoundError(
            f"No snapshots were found in {train_folder}! Please ensure the network has "
            f"been trained and verify the iteration, shuffle and trainFraction are "
            f"correct."
        )

    # sort in ascending order of iteration number
    return sorted(snapshot_names, key=lambda name: int(name.split("-")[1]))


def get_deeplabcut_path():
    """Get path of where deeplabcut is currently running"""
    import importlib.util

    return os.path.split(importlib.util.find_spec("deeplabcut").origin)[0]


def intersection_of_body_parts_and_ones_given_by_user(cfg, comparisonbodyparts):
    """Returns all body parts when comparisonbodyparts=='all', otherwise all bpts that are in the intersection of comparisonbodyparts and the actual bodyparts"""
    # if "MULTI!" in allbpts:
    if cfg["multianimalproject"]:
        allbpts = cfg["multianimalbodyparts"] + cfg["uniquebodyparts"]
    else:
        allbpts = cfg["bodyparts"]

    if comparisonbodyparts == "all":
        return list(allbpts)
    else:  # take only items in list that are actually bodyparts...
        cpbpts = [bp for bp in allbpts if bp in comparisonbodyparts]
        # Ensure same order as in config.yaml
        return cpbpts


def get_labeled_data_folder(cfg, video):
    videoname = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(cfg["project_path"], "labeled-data", videoname)


def form_data_containers(df, bodyparts):
    mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
    df_masked = df.loc[:, mask]
    df_likelihood = df_masked.xs("likelihood", level=-1, axis=1).values.T
    df_x = df_masked.xs("x", level=-1, axis=1).values.T
    df_y = df_masked.xs("y", level=-1, axis=1).values.T
    return df_x, df_y, df_likelihood


def get_scorer_name(
    cfg: dict,
    shuffle: int,
    trainFraction: float,
    trainingsiterations: str | int = "unknown",
    modelprefix: str = "",
    engine: Engine | None = None,
):
    """Extract the scorer/network name for a particular shuffle, training fraction, etc.
    If the engine is not specified, determines which to use from
    Returns tuple of DLCscorer, DLCscorerlegacy (old naming convention)
    """
    if engine is None:
        from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine

        engine = get_shuffle_engine(
            cfg=cfg,
            trainingsetindex=cfg["TrainingFraction"].index(trainFraction),
            shuffle=shuffle,
            modelprefix=modelprefix,
        )

    if engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_scorer_name

        snapshot_index = None
        if isinstance(trainingsiterations, int):
            snapshot_index = trainingsiterations

        dlc3_scorer = get_scorer_name(
            cfg=cfg,
            shuffle=shuffle,
            train_fraction=trainFraction,
            snapshot_index=snapshot_index,
            detector_index=None,
            modelprefix=modelprefix,
        )
        return dlc3_scorer, dlc3_scorer

    Task = cfg["Task"]
    date = cfg["date"]

    if trainingsiterations == "unknown":
        snapshotindex = get_snapshot_index_for_scorer(
            "snapshotindex", cfg["snapshotindex"]
        )
        model_folder = get_model_folder(
            trainFraction, shuffle, cfg, engine=engine, modelprefix=modelprefix
        )
        train_folder = Path(cfg["project_path"]) / model_folder / "train"
        snapshot_names = get_snapshots_from_folder(train_folder)
        snapshot_name = snapshot_names[snapshotindex]
        trainingsiterations = (snapshot_name.split(os.sep)[-1]).split("-")[-1]

    dlc_cfg = read_plainconfig(
        os.path.join(
            cfg["project_path"],
            str(
                get_model_folder(
                    trainFraction, shuffle, cfg, engine=engine, modelprefix=modelprefix
                )
            ),
            "train",
            engine.pose_cfg_name,
        )
    )
    # ABBREVIATE NETWORK NAMES -- esp. for mobilenet!
    if "resnet" in dlc_cfg["net_type"]:
        if dlc_cfg.get("multi_stage", False):
            netname = "dlcrnetms5"
        else:
            netname = dlc_cfg["net_type"].replace("_", "")
    elif "mobilenet" in dlc_cfg["net_type"]:  # mobilenet >> mobnet_100; mobnet_35 etc.
        netname = "mobnet_" + str(int(float(dlc_cfg["net_type"].split("_")[-1]) * 100))
    elif "efficientnet" in dlc_cfg["net_type"]:
        netname = "effnet_" + dlc_cfg["net_type"].split("-")[1]
    else:
        raise ValueError(f"Failed to abbreviate network name: {dlc_cfg['net_type']}")

    scorer = (
        "DLC_"
        + netname
        + "_"
        + Task
        + str(date)
        + "shuffle"
        + str(shuffle)
        + "_"
        + str(trainingsiterations)
    )
    # legacy scorername until DLC 2.1. (cfg['resnet'] is deprecated / which is why we get the resnet_xyz name from dlc_cfg!
    # scorer_legacy = 'DeepCut' + "_resnet" + str(cfg['resnet']) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
    scorer_legacy = scorer.replace("DLC", "DeepCut")
    return scorer, scorer_legacy


def check_if_post_processing(
    folder, vname, DLCscorer, DLCscorerlegacy, suffix="filtered"
):
    """Checks if filtered/bone lengths were already calculated. If not, figures
    out if data was already analyzed (either with legacy scorer name or new one!)"""
    outdataname = os.path.join(folder, vname + DLCscorer + suffix + ".h5")
    sourcedataname = os.path.join(folder, vname + DLCscorer + ".h5")
    if os.path.isfile(outdataname):  # was data already processed?
        if suffix == "filtered":
            print("Video already filtered...", outdataname)
        elif suffix == "_skeleton":
            print("Skeleton in video already processed...", outdataname)

        return False, outdataname, sourcedataname, DLCscorer
    else:
        odn = os.path.join(folder, vname + DLCscorerlegacy + suffix + ".h5")
        if os.path.isfile(odn):  # was it processed by DLC <2.1 project?
            if suffix == "filtered":
                print("Video already filtered...(with DLC<2.1)!", odn)
            elif suffix == "_skeleton":
                print("Skeleton in video already processed... (with DLC<2.1)!", odn)
            return False, odn, odn, DLCscorerlegacy
        else:
            sdn = os.path.join(folder, vname + DLCscorerlegacy + ".h5")
            tracks = sourcedataname.replace(".h5", "tracks.h5")
            if os.path.isfile(sourcedataname):  # Was the video already analyzed?
                return True, outdataname, sourcedataname, DLCscorer
            elif os.path.isfile(sdn):  # was it analyzed with DLC<2.1?
                return True, odn, sdn, DLCscorerlegacy
            elif os.path.isfile(tracks):  # May be a MA project with tracklets
                return True, tracks.replace(".h5", f"{suffix}.h5"), tracks, DLCscorer
            else:
                print("Video not analyzed -- Run analyze_videos first.")
                return False, outdataname, sourcedataname, DLCscorer


def check_if_not_analyzed(destfolder, vname, DLCscorer, DLCscorerlegacy, flag="video"):
    h5files = list(grab_files_in_folder(destfolder, "h5", relative=False))
    if not len(h5files):
        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")
        return True, dataname, DLCscorer

    # Iterate over data files and stop as soon as one matching the scorer is found
    for h5file in h5files:
        if vname + DLCscorer in Path(h5file).stem:
            if flag == "video":
                print("Video already analyzed!", h5file)
            elif flag == "framestack":
                print("Frames already analyzed!", h5file)
            return False, h5file, DLCscorer
        elif vname + DLCscorerlegacy in Path(h5file).stem:
            if flag == "video":
                print("Video already analyzed!", h5file)
            elif flag == "framestack":
                print("Frames already analyzed!", h5file)
            return False, h5file, DLCscorerlegacy

    # If there was no match...
    dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")
    return True, dataname, DLCscorer


def check_if_not_evaluated(folder, DLCscorer, DLCscorerlegacy, snapshot):
    dataname = os.path.join(folder, DLCscorer + "-" + str(snapshot) + ".h5")
    if os.path.isfile(dataname):
        print("This net has already been evaluated!")
        return False, dataname, DLCscorer
    else:
        dn = os.path.join(folder, DLCscorerlegacy + "-" + str(snapshot) + ".h5")
        if os.path.isfile(dn):
            print("This net has already been evaluated (with DLC<2.1)!")
            return False, dn, DLCscorerlegacy
        else:
            return True, dataname, DLCscorer


def find_video_full_data(folder, videoname, scorer):
    scorer_legacy = scorer.replace("DLC", "DeepCut")
    full_files = filter_files_by_patterns(
        folder=folder,
        start_patterns={videoname + scorer, videoname + scorer_legacy},
        contain_patterns={"full"},
        end_patterns={"pickle"},
    )
    if not full_files:
        raise FileNotFoundError(
            f"No full data found in {folder} "
            f"for video {videoname} and scorer {scorer}."
        )
    return full_files[0]


def find_video_metadata(folder, videoname, scorer):
    """For backward compatibility, let us search the substring 'meta'"""
    scorer_legacy = scorer.replace("DLC", "DeepCut")
    meta_files = filter_files_by_patterns(
        folder=folder,
        start_patterns={videoname + scorer, videoname + scorer_legacy},
        contain_patterns={"meta"},
        end_patterns={"pickle"},
    )
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata found in {folder} "
            f"for video {videoname} and scorer {scorer}."
        )
    return meta_files[0]


def load_video_metadata(folder, videoname, scorer):
    return read_pickle(find_video_metadata(folder, videoname, scorer))


def load_video_full_data(folder, videoname, scorer):
    return read_pickle(find_video_full_data(folder, videoname, scorer))


def find_analyzed_data(folder, videoname, scorer, filtered=False, track_method=""):
    """Find potential data files from the hints given to the function."""
    scorer_legacy = scorer.replace("DLC", "DeepCut")
    suffix = "_filtered" if filtered else ""
    tracker = TRACK_METHODS.get(track_method, "")

    candidates = []
    for file in grab_files_in_folder(folder, "h5"):
        stem = Path(file).stem.replace("_filtered", "")
        starts_by_scorer = file.startswith(videoname + scorer) or file.startswith(
            videoname + scorer_legacy
        )
        if tracker:
            matches_tracker = stem.endswith(tracker)
        else:
            matches_tracker = not any(stem.endswith(s) for s in TRACK_METHODS.values())
        if all(
            (
                starts_by_scorer,
                "skeleton" not in file,
                matches_tracker,
                (filtered and "filtered" in file)
                or (not filtered and "filtered" not in file),
            )
        ):
            candidates.append(file)

    if not len(candidates):
        msg = (
            f'No {"un" if not filtered else ""}filtered data file found in {folder} '
            f"for video {videoname} and scorer {scorer}"
        )
        if track_method:
            msg += f" and {track_method} tracker"
        msg += "."
        raise FileNotFoundError(msg)

    n_candidates = len(candidates)
    if n_candidates > 1:  # This should not be happening anyway...
        print(
            f"{n_candidates} possible data files were found: {candidates}.\n"
            f"Picking the first by default..."
        )
    filepath = os.path.join(folder, candidates[0])
    scorer = scorer if scorer in filepath else scorer_legacy
    return filepath, scorer, suffix


def load_analyzed_data(folder, videoname, scorer, filtered=False, track_method=""):
    filepath, scorer, suffix = find_analyzed_data(
        folder, videoname, scorer, filtered, track_method
    )
    df = pd.read_hdf(filepath)
    return df, filepath, scorer, suffix


def load_detection_data(video, scorer, track_method):
    folder = os.path.dirname(video)
    videoname = os.path.splitext(os.path.basename(video))[0]
    if track_method == "skeleton":
        tracker = "sk"
    elif track_method == "box":
        tracker = "bx"
    elif track_method == "ellipse":
        tracker = "el"
    else:
        raise ValueError(f"Unrecognized track_method={track_method}")

    filepath = os.path.splitext(video)[0] + scorer + f"_{tracker}.pickle"
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"No detection data found in {folder} for video {videoname}, "
            f"scorer {scorer}, and tracker {track_method}"
        )
    return read_pickle(filepath)


def find_next_unlabeled_folder(config_path, verbose=False):
    cfg = read_config(config_path)
    base_folder = Path(os.path.join(cfg["project_path"], "labeled-data"))
    h5files = sorted(
        base_folder.rglob("*.h5"),
        key=lambda p: p.lstat().st_mtime,
        reverse=True,
    )
    folders = sorted(f for f in base_folder.iterdir() if f.is_dir())
    most_recent_folder = h5files[0].parent
    ind = folders.index(most_recent_folder)
    next_folder = folders[min(ind + 1, len(folders) - 1)]
    if verbose:  # Print some stats about data completeness
        print("Data completeness\n-----------------")
        for folder in folders:
            dfs = []
            for file in folder.rglob("*.h5"):
                dfs.append(pd.read_hdf(file))
            if dfs:
                df = pd.concat(dfs)
                frac = (~df.isna()).sum().sum() / df.size
                print(f"{folder.name} | {int(100 * frac)} %")
    return next_folder


def get_snapshot_index_for_scorer(name: str, index: int | str) -> int:
    if index == "all":
        print(
            f"Changing {name} to the last one -- plotting, videomaking, etc. should "
            "not be performed for all indices. For more selectivity enter the ordinal "
            "number of the snapshot you want (ie. 4 for the fifth) in the config file."
        )
        return -1

    return index


# aliases for backwards-compatibility.
SaveData = save_data
SaveMetadata = save_metadata
LoadMetadata = load_metadata
GetVideoList = get_video_list
GetTrainingSetFolder = get_training_set_folder
GetDataandMetaDataFilenames = get_data_and_metadata_filenames
IntersectionofBodyPartsandOnesGivenbyUser = (
    intersection_of_body_parts_and_ones_given_by_user
)
GetScorerName = get_scorer_name
CheckifPostProcessing = check_if_post_processing
CheckifNotAnalyzed = check_if_not_analyzed
CheckifNotEvaluated = check_if_not_evaluated
GetEvaluationFolder = get_evaluation_folder
GetModelFolder = get_model_folder
