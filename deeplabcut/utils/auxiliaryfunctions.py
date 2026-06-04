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
import pickle
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import ruamel.yaml.representer
import yaml
from ruamel.yaml import YAML

from deeplabcut.core.engine import Engine
from deeplabcut.core.trackingutils import TRACK_METHODS
from deeplabcut.utils import auxfun_multianimal
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS, collect_video_paths
from deeplabcut.utils.deprecation import deprecated


def as_path(path: str | Path) -> Path:
    """Coerce a filesystem path argument to :class:`~pathlib.Path`."""
    return Path(path)


def as_optional_path(path: str | Path | None) -> Path | None:
    """Coerce an optional filesystem path argument to :class:`~pathlib.Path`."""
    return None if path is None else Path(path)


def as_path_list(paths: Sequence[str | Path]) -> list[Path]:
    """Coerce a sequence of filesystem path arguments to :class:`~pathlib.Path`."""
    return [Path(p) for p in paths]


def create_config_template(multianimal=False):
    """Creates a template for config.yaml file.

    This specific order is preserved while saving as yaml file.
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
    """Creates a template for config.yaml file for 3d project.

    This specific order is preserved while saving as yaml file.
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


def safe_resolve(path: Path) -> Path:
    """Return a resolved Path that is safe to use with str-based I/O.

    Prefers Path.resolve() so that symlinks are followed (useful on Linux).
    Falls back to Path.absolute() when the resolved path cannot be opened
    as a plain string — e.g. on Windows 11 + SMB network drives where
    resolve() may return an unusable \\\\?\\Volume{GUID}\\... form.

    See https://github.com/DeepLabCut/DeepLabCut/issues/3348
    """
    resolved = path.resolve()
    try:
        resolved.open().close()
        return resolved
    except OSError:
        return path.absolute()


def read_config(configname: str | Path):
    """Reads structured config file defining a project."""
    ruamelFile = YAML()
    path = Path(configname)
    if path.exists():
        try:
            with path.open() as f:
                cfg = ruamelFile.load(f)
                curr_dir = Path(configname).parent.absolute()

                if cfg.get("engine") is None:
                    cfg["engine"] = Engine.TF.aliases[0]
                    write_config(configname, cfg)

                if cfg.get("detector_snapshotindex") is None:
                    cfg["detector_snapshotindex"] = -1

                if cfg.get("detector_batch_size") is None:
                    cfg["detector_batch_size"] = 1

                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = str(curr_dir)
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                    with path.open() as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            f"Config file at {path} not found. Please make sure that the file exists and/or that you passed the path of"
            f"the config file correctly!"
        )
    return cfg


def write_config(configname, cfg):
    """Write structured config file."""
    with Path(configname).open("w") as cf:
        cfg_file, ruamelFile = create_config_template(cfg.get("multianimalproject", False))
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if "skeleton" not in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        # Use a very large width so long strings (e.g., file paths or keys with spaces)
        # are kept on a single line instead of being wrapped, which can otherwise cause
        # them to be emitted as complex keys. See also:
        # https://stackoverflow.com/questions/31197268/pyyaml-yaml-dump-produces-complex-key-for-string-key-122-chars/31199123#31199123
        ruamelFile.width = 1_000_000
        ruamelFile.dump(cfg_file, cf)


def edit_config(configname, edits, output_name=""):
    """Convenience function to edit and save a config file from a dictionary.

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
        warnings.warn("Some edits could not be written. The configuration file will be left unchanged.", stacklevel=2)
        for key in edits:
            cfg.pop(key)
        write_plainconfig(output_name, cfg)
    return cfg


def get_bodyparts(cfg: dict) -> list[str]:
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


def get_unique_bodyparts(cfg: dict) -> list[str]:
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
    """Write structured 3D config file."""
    with Path(configname).open("w") as cf:
        cfg_file, ruamelFile = create_config_template_3d()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]
        ruamelFile.dump(cfg_file, cf)


def write_config_3d_template(projconfigfile, cfg_file_3d, ruamelFile_3d):
    with Path(projconfigfile).open("w") as cf:
        ruamelFile_3d.dump(cfg_file_3d, cf)


def read_plainconfig(configname):
    if not Path(configname).exists():
        raise FileNotFoundError(f"Config {configname} is not found. Please make sure that the file exists.")
    with Path(configname).open() as file:
        return YAML().load(file)


def write_plainconfig(configname, cfg):
    with Path(configname).open("w") as file:
        YAML().dump(cfg, file)


def attempt_to_make_folder(foldername, recursive=False):
    """Attempts to create a folder with specified name.

    Does nothing if it already exists.
    """
    try:
        Path(foldername).is_dir()
    except TypeError:  # https://www.python.org/dev/peps/pep-0519/
        foldername = os.fspath(foldername)  # https://github.com/DeepLabCut/DeepLabCut/issues/105 (windows)

    if Path(foldername).is_dir():
        pass
    else:
        if recursive:
            Path(foldername).mkdir(parents=True)
        else:
            Path(foldername).mkdir()


def read_pickle(filename):
    """Read the pickle file."""
    with Path(filename).open("rb") as handle:
        return pickle.load(handle)


def write_pickle(filename, data):
    """Write the pickle file."""
    with Path(filename).open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


@deprecated(replacement="deeplabcut.collect_video_paths", since="3.0.0")
def get_list_of_videos(
    videos: list[str] | str,
    videotype: str | Sequence[str] | None = SUPPORTED_VIDEOS,
    in_random_order: bool = True,
) -> list[str]:
    video_paths = collect_video_paths(
        data_path=videos,
        extensions=videotype,
        shuffle=in_random_order,
    )
    return [str(path) for path in video_paths]


def save_data(PredicteData, metadata, dataname, pdindex, imagenames, save_as_csv):
    """Save predicted data as h5 file and metadata as pickle file; created by
    predict_videos.py."""
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    if save_as_csv:
        print("Saving csv poses!")
        DataMachine.to_csv(dataname.split(".h5")[0] + ".csv")
    DataMachine.to_hdf(dataname, key="df_with_missing", format="table", mode="w")
    with Path(dataname.split(".h5")[0] + "_meta.pickle").open("wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def save_metadata(
    metadatafilename: str | Path,
    data,
    trainIndices,
    testIndices,
    trainFraction,
):
    with Path(metadatafilename).open("wb") as f:
        # Pickle the 'labeled-data' dictionary using the highest protocol available.
        pickle.dump([data, trainIndices, testIndices, trainFraction], f, pickle.HIGHEST_PROTOCOL)


def load_metadata(metadatafile: str | Path):
    with Path(metadatafile).open("rb") as f:
        [
            trainingdata_details,
            trainIndices,
            testIndices,
            testFraction_data,
        ] = pickle.load(f)
        return trainingdata_details, trainIndices, testIndices, testFraction_data


def get_immediate_subdirectories(a_dir):
    """Get list of immediate subdirectories."""
    return [p.name for p in Path(a_dir).iterdir() if p.is_dir()]


# TODO: @deruyter92 2026-05-20: this function could be updated to match the
# signature of collect_video_paths, allowing for multiple extensions.
def grab_files_in_folder(folder, ext="", relative=True):
    """Return the paths of files with extension *ext* present in *folder*."""
    for file in Path(folder).iterdir():
        if file.name.endswith(ext):
            yield file.name if relative else str(file)


def filter_files_by_patterns(
    folder: str | Path,
    start_patterns: set[str] | None = None,
    contain_patterns: set[str] | None = None,
    end_patterns: set[str] | None = None,
) -> list[Path]:
    """Filters files in a folder based on start, contain, and end patterns.

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
        and (not start_patterns or any(file.name.startswith(start) for start in start_patterns))
        and (not contain_patterns or any(contain in file.name for contain in contain_patterns))
        and (not end_patterns or any(file.name.endswith(end) for end in end_patterns))
    ]

    return matching_files


@deprecated(replacement="deeplabcut.collect_video_paths", since="3.0.0")
def get_video_list(filename, videopath, videtype):
    """Get list of videos in a path (if filetype == all), otherwise just a specific
    file."""
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
    """Training Set folder for config file based on parameters."""
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path("training-datasets") / iterate / ("UnaugmentedDataSet_" + Task + date)


def get_data_and_metadata_filenames(
    trainingsetfolder: str | Path,
    trainFraction: float,
    shuffle: int,
    cfg: dict,
) -> tuple[Path, Path]:
    """Paths to data and metadata files relative to the project root."""
    base = Path(trainingsetfolder)
    datafn = base / (
        cfg["Task"] + "_" + cfg["scorer"] + str(int(100 * trainFraction)) + "shuffle" + str(shuffle) + ".mat"
    )
    metadatafn = base / (
        "Documentation_data-" + cfg["Task"] + "_" + str(int(trainFraction * 100)) + "shuffle" + str(shuffle) + ".pickle"
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
        Task + date + "-trainset" + str(int(trainFraction * 100)) + "shuffle" + str(shuffle),
    )


def get_snapshots_from_folder(train_folder: Path) -> list[str]:
    """Returns an ordered list of existing snapshot names in the train folder, sorted by
    increasing training iterations.

    Raises:
        FileNotFoundError: if no snapshot_names are found in the train_folder.
    """
    snapshot_names = [file.stem for file in train_folder.iterdir() if "index" in file.name]

    if len(snapshot_names) == 0:
        raise FileNotFoundError(
            f"No snapshots were found in {train_folder}! Please ensure the network has "
            f"been trained and verify the iteration, shuffle and trainFraction are "
            f"correct."
        )

    # sort in ascending order of iteration number
    return sorted(snapshot_names, key=lambda name: int(name.split("-")[1]))


def get_deeplabcut_path() -> Path:
    """Get path of where deeplabcut is currently running."""
    import importlib.util

    return Path(importlib.util.find_spec("deeplabcut").origin).parent


def intersection_of_body_parts_and_ones_given_by_user(cfg, comparisonbodyparts):
    """Returns all body parts when comparisonbodyparts=='all', otherwise all bpts that
    are in the intersection of comparisonbodyparts and the actual bodyparts."""
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


def get_labeled_data_folder(cfg: dict, video: str | Path) -> Path:
    videoname = Path(video).stem
    return Path(cfg["project_path"]) / "labeled-data" / videoname


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
    **kwargs,
):
    """Extract the scorer/network name for a particular shuffle, training fraction, etc.
    If the engine is not specified, determines which to use from
    kwargs: additional arguments.
        For torch-based shuffles, can be used to specify:
            - snapshot_index
            - detector_snapshot_index

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

        snapshot_index = kwargs.get("snapshot_index", None)
        detector_snapshot_index = kwargs.get("detector_snapshot_index", None)
        dlc3_scorer = get_scorer_name(
            cfg=cfg,
            shuffle=shuffle,
            train_fraction=trainFraction,
            snapshot_index=snapshot_index,
            detector_index=detector_snapshot_index,
            modelprefix=modelprefix,
        )
        return dlc3_scorer, dlc3_scorer

    Task = cfg["Task"]
    date = cfg["date"]

    if trainingsiterations == "unknown":
        snapshotindex = get_snapshot_index_for_scorer("snapshotindex", cfg["snapshotindex"])
        model_folder = get_model_folder(trainFraction, shuffle, cfg, engine=engine, modelprefix=modelprefix)
        train_folder = Path(cfg["project_path"]) / model_folder / "train"
        snapshot_names = get_snapshots_from_folder(train_folder)
        snapshot_name = snapshot_names[snapshotindex]
        trainingsiterations = Path(snapshot_name).parts[-1].split("-")[-1]

    dlc_cfg = read_plainconfig(
        str(
            Path(cfg["project_path"])
            / get_model_folder(trainFraction, shuffle, cfg, engine=engine, modelprefix=modelprefix)
            / "train"
            / engine.pose_cfg_name
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

    scorer = "DLC_" + netname + "_" + Task + str(date) + "shuffle" + str(shuffle) + "_" + str(trainingsiterations)
    # legacy scorername until DLC 2.1. (cfg['resnet'] is deprecated / which is why we get the resnet_xyz name from
    # dlc_cfg!
    # scorer_legacy = 'DeepCut' + "_resnet" + str(cfg['resnet']) + "_" + Task + str(date) + 'shuffle' + str(shuffle) +
    # '_' + str(trainingsiterations)
    scorer_legacy = scorer.replace("DLC", "DeepCut")
    return scorer, scorer_legacy


def check_if_post_processing(folder, vname, DLCscorer, DLCscorerlegacy, suffix="filtered"):
    """Checks if filtered/bone lengths were already calculated.

    If not, figures out if data was already analyzed (either with legacy scorer name or
    new one!)
    """
    folder = Path(folder)
    outdataname = str(folder / (vname + DLCscorer + suffix + ".h5"))
    sourcedataname = str(folder / (vname + DLCscorer + ".h5"))
    if Path(outdataname).is_file():  # was data already processed?
        if suffix == "filtered":
            print("Video already filtered...", outdataname)
        elif suffix == "_skeleton":
            print("Skeleton in video already processed...", outdataname)

        return False, outdataname, sourcedataname, DLCscorer
    else:
        odn = str(folder / (vname + DLCscorerlegacy + suffix + ".h5"))
        if Path(odn).is_file():  # was it processed by DLC <2.1 project?
            if suffix == "filtered":
                print("Video already filtered...(with DLC<2.1)!", odn)
            elif suffix == "_skeleton":
                print("Skeleton in video already processed... (with DLC<2.1)!", odn)
            return False, odn, odn, DLCscorerlegacy
        else:
            sdn = str(folder / (vname + DLCscorerlegacy + ".h5"))
            tracks = sourcedataname.replace(".h5", "tracks.h5")
            if Path(sourcedataname).is_file():  # Was the video already analyzed?
                return True, outdataname, sourcedataname, DLCscorer
            elif Path(sdn).is_file():  # was it analyzed with DLC<2.1?
                return True, odn, sdn, DLCscorerlegacy
            elif Path(tracks).is_file():  # May be a MA project with tracklets
                return True, tracks.replace(".h5", f"{suffix}.h5"), tracks, DLCscorer
            else:
                print("Video not analyzed -- Run analyze_videos first.")
                return False, outdataname, sourcedataname, DLCscorer


def check_if_not_analyzed(destfolder, vname, DLCscorer, DLCscorerlegacy, flag="video"):
    h5files = list(grab_files_in_folder(destfolder, "h5", relative=False))
    if not len(h5files):
        dataname = str(Path(destfolder) / (vname + DLCscorer + ".h5"))
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
    dataname = str(Path(destfolder) / (vname + DLCscorer + ".h5"))
    return True, dataname, DLCscorer


def check_if_not_evaluated(folder, DLCscorer, DLCscorerlegacy, snapshot):
    dataname = str(Path(folder) / (DLCscorer + "-" + str(snapshot) + ".h5"))
    if Path(dataname).is_file():
        print("This net has already been evaluated!")
        return False, dataname, DLCscorer
    else:
        dn = str(Path(folder) / (DLCscorerlegacy + "-" + str(snapshot) + ".h5"))
        if Path(dn).is_file():
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
        raise FileNotFoundError(f"No full data found in {folder} for video {videoname} and scorer {scorer}.")
    return full_files[0]


def find_video_metadata(folder, videoname: str, scorer: str):
    """For backward compatibility, let us search the substring 'meta'."""

    scorer_legacy = scorer.replace("DLC", "DeepCut")
    meta_files = filter_files_by_patterns(
        folder=folder,
        start_patterns={videoname + scorer, videoname + scorer_legacy},
        contain_patterns={"meta"},
        end_patterns={"pickle"},
    )
    if not meta_files:
        raise FileNotFoundError(f"No metadata found in {folder} for video {videoname} and scorer {scorer}.")
    return meta_files[0]


def load_video_metadata(folder, videoname, scorer):
    return read_pickle(find_video_metadata(folder, videoname, scorer))


def load_video_full_data(folder, videoname, scorer):
    return read_pickle(find_video_full_data(folder, videoname, scorer))


def find_analyzed_data(folder, videoname: str, scorer: str, filtered=False, track_method=""):
    """Find potential data files from the hints given to the function."""

    scorer_legacy = scorer.replace("DLC", "DeepCut")
    suffix = "_filtered" if filtered else ""
    tracker = TRACK_METHODS.get(track_method, "")

    candidates = []
    for file in grab_files_in_folder(folder, "h5"):
        stem = Path(file).stem.replace("_filtered", "")
        starts_by_scorer = file.startswith(videoname + scorer) or file.startswith(videoname + scorer_legacy)
        if tracker:
            matches_tracker = stem.endswith(tracker)
        else:
            matches_tracker = not any(stem.endswith(s) for s in TRACK_METHODS.values())
        if all(
            (
                starts_by_scorer,
                "skeleton" not in file,
                matches_tracker,
                (filtered and "filtered" in file) or (not filtered and "filtered" not in file),
            )
        ):
            candidates.append(file)

    if not len(candidates):
        msg = (
            f"No {'un' if not filtered else ''}filtered data file found in {folder} "
            f"for video {videoname} and scorer {scorer}"
        )
        if track_method:
            msg += f" and {track_method} tracker"
        msg += "."
        raise FileNotFoundError(msg)

    n_candidates = len(candidates)
    if n_candidates > 1:  # This should not be happening anyway...
        print(f"{n_candidates} possible data files were found: {candidates}.\nPicking the first by default...")
    filepath = str(Path(folder) / candidates[0])
    scorer = scorer if scorer in filepath else scorer_legacy
    return filepath, scorer, suffix


def load_analyzed_data(folder, videoname, scorer, filtered=False, track_method=""):
    filepath, scorer, suffix = find_analyzed_data(folder, videoname, scorer, filtered, track_method)
    df = pd.read_hdf(filepath)
    return df, filepath, scorer, suffix


def load_detection_data(video, scorer, track_method):
    video = Path(video)
    folder = video.parent
    videoname = video.stem
    if track_method == "skeleton":
        tracker = "sk"
    elif track_method == "box":
        tracker = "bx"
    elif track_method == "ellipse":
        tracker = "el"
    else:
        raise ValueError(f"Unrecognized track_method={track_method}")

    filepath = str(video.with_suffix("")) + scorer + f"_{tracker}.pickle"
    if not Path(filepath).is_file():
        raise FileNotFoundError(
            f"No detection data found in {folder} for video {videoname}, scorer {scorer}, and tracker {track_method}"
        )
    return read_pickle(filepath)


def find_next_unlabeled_folder(config_path, verbose=False):
    cfg = read_config(config_path)
    base_folder = Path(cfg["project_path"]) / "labeled-data"
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
IntersectionofBodyPartsandOnesGivenbyUser = intersection_of_body_parts_and_ones_given_by_user
GetScorerName = get_scorer_name
CheckifPostProcessing = check_if_post_processing
CheckifNotAnalyzed = check_if_not_analyzed
CheckifNotEvaluated = check_if_not_evaluated
GetEvaluationFolder = get_evaluation_folder
GetModelFolder = get_model_folder
