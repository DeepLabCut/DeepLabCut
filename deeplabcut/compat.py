"""Compatibility file for methods available with either PyTorch or Tensorflow"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
from ruamel.yaml import YAML


@dataclass(frozen=True)
class EngineDataMixin:
    aliases: tuple[str]
    model_folder_name: str
    pose_cfg_name: str
    results_folder_name: str


class Engine(EngineDataMixin, Enum):
    PYTORCH = (
        ("pytorch", "torch"),
        "dlc-models-pytorch",
        "pytorch_config.yaml",
        "evaluation-results-pytorch",
    )
    TF = (
        ("tensorflow", "tf"),
        "dlc-models",
        "pose_cfg.yaml",
        "evaluation-results",
    )

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if value.lower() in member.aliases:
                    return member
        return None


DEFAULT_ENGINE = Engine.PYTORCH


def get_project_engine(cfg: dict) -> Engine:
    """
    Args:
        cfg: the project configuration file

    Returns:
        the engine specified for the project, or the default engine if none is specified
    """
    if cfg.get("engine") is not None:
        return Engine(cfg["engine"])

    return DEFAULT_ENGINE


def get_shuffle_engine(
    cfg: dict,
    trainingsetindex: int,
    shuffle: int,
    modelprefix: str = "",
) -> Engine:
    """
    Args:
        cfg: the project configuration file
        trainingsetindex: the training set index used
        shuffle: the shuffle for which to get the engine
        modelprefix: the added prefix

    Returns:
        the engine that the shuffle was created with

    Raises:
        ValueError if the engine for the shuffle cannot be determined or the shuffle
        doesn't exist
    """
    project_path = Path(cfg["project_path"])
    train_frac = int(100 * cfg["TrainingFraction"][trainingsetindex])
    shuffle_name = f"{cfg['Task']}{cfg['date']}-trainset{train_frac}shuffle{shuffle}"

    found_engines = set()
    for engine in Engine:
        models_root = project_path / modelprefix / engine.model_folder_name
        train_folder = models_root / f"iteration-{cfg['iteration']}" / shuffle_name
        if train_folder.exists():
            found_engines.add(engine)

    if len(found_engines) == 1:
        return found_engines.pop()
    elif len(found_engines) > 1:
        logging.warning(
            "There are multiple engines with model configurations defined for "
            f"train_frac={train_frac} and shuffle={shuffle}: {found_engines}"
        )
        if DEFAULT_ENGINE in found_engines:
            logging.warning(f" -> using the default engine: {DEFAULT_ENGINE}")
            return DEFAULT_ENGINE
        else:
            selected_engine = found_engines.pop()
            logging.warning(f" -> using a random engine: {selected_engine}")
            return selected_engine

    raise ValueError(
        f"Could not get the engine for the shuffle {shuffle_name}. Could not find a "
        f"folder for any engine."
    )


def train_network(
    config: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    max_snapshots_to_keep: int = 5,
    displayiters: int | None = None,
    saveiters: int | None = None,
    maxiters: int | None = None,
    allow_growth: bool = True,
    gputouse: str | None = None,
    autotune: bool = False,
    keepdeconvweights: bool = True,
    modelprefix: str = "",
    **torch_kwargs,
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import train_network
        return train_network(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            max_snapshots_to_keep=max_snapshots_to_keep,
            displayiters=displayiters,
            saveiters=saveiters,
            maxiters=maxiters,
            allow_growth=allow_growth,
            gputouse=gputouse,
            autotune=autotune,
            keepdeconvweights=keepdeconvweights,
            modelprefix=modelprefix,
        )
    elif engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis import train_network
        _update_device(gputouse, torch_kwargs)
        return train_network(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            modelprefix=modelprefix,
            **torch_kwargs,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def return_train_network_path(
    config,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    engine: Engine | None = None,
) -> tuple[Path, Path, Path]:
    if engine is None:
        engine = get_shuffle_engine(
            _load_config(config),
            trainingsetindex=trainingsetindex,
            shuffle=shuffle,
            modelprefix=modelprefix,
        )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import return_train_network_path
        return return_train_network_path(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            modelprefix=modelprefix,
        )
    elif engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis.utils import return_train_network_path
        return return_train_network_path(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            modelprefix=modelprefix,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def evaluate_network(
    config,
    Shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    plotting: bool | str = False,
    show_errors: bool = True,
    comparisonbodyparts: str | list[str] = "all",
    gputouse: str | None = None,
    rescale: bool = False,
    modelprefix: str = "",
    per_keypoint_evaluation: bool = False,
    **torch_kwargs,
):
    cfg = _load_config(config)
    engines = set()
    for shuffle in Shuffles:
        engines.add(
            get_shuffle_engine(
                cfg,
                trainingsetindex=trainingsetindex,
                shuffle=shuffle,
                modelprefix=modelprefix,
            )
        )
    if len(engines) == 0:
        raise ValueError(
            f"You must pass at least one shuffle to evaluate (had {list(Shuffles)})"
        )
    elif len(engines) > 1:
        raise ValueError(
            f"All shuffles must have the same engine (found {list(engines)})"
        )

    engine = engines.pop()
    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import evaluate_network
        return evaluate_network(
            config,
            Shuffles=Shuffles,
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=show_errors,
            comparisonbodyparts=comparisonbodyparts,
            gputouse=gputouse,
            rescale=rescale,
            modelprefix=modelprefix,
            per_keypoint_evaluation=per_keypoint_evaluation,
        )
    elif engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis import evaluate_network
        _update_device(gputouse, torch_kwargs)
        return evaluate_network(
            config,
            shuffles=Shuffles,
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=show_errors,
            modelprefix=modelprefix,
            **torch_kwargs,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def return_evaluate_network_data(
    config: str,
    shuffle: int = 0,
    trainingsetindex: int = 0,
    comparisonbodyparts: str | list[str] = "all",
    Snapindex: str | int | None = None,
    rescale: bool = False,
    fulldata: bool = False,
    show_errors: bool = True,
    modelprefix: str = "",
    returnjustfns: bool = True,
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import return_evaluate_network_data
        return return_evaluate_network_data(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            comparisonbodyparts=comparisonbodyparts,
            Snapindex=Snapindex,
            rescale=rescale,
            fulldata=fulldata,
            show_errors=show_errors,
            modelprefix=modelprefix,
            returnjustfns=returnjustfns,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def analyze_videos(
    config: str,
    videos: list[str],
    videotype: str = "",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: str | None = None,
    save_as_csv: bool = False,
    in_random_order: bool = True,
    destfolder: str | None = None,
    batchsize: int = None,
    cropping: list[int] | None = None,
    TFGPUinference: bool = True,
    dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    modelprefix: str = "",
    robust_nframes: bool = False,
    allow_growth: bool = False,
    use_shelve: bool = False,
    auto_track: bool = True,
    n_tracks: int | None = None,
    calibrate: bool = False,
    identity_only: bool = False,
    use_openvino: str | None = None,
    **torch_kwargs,
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import analyze_videos
        kwargs = {}
        if use_openvino is not None:  # otherwise default comes from tensorflow API
            kwargs["use_openvino"] = use_openvino

        return analyze_videos(
            config,
            videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=gputouse,
            save_as_csv=save_as_csv,
            in_random_order=in_random_order,
            destfolder=destfolder,
            batchsize=batchsize,
            cropping=cropping,
            TFGPUinference=TFGPUinference,
            dynamic=dynamic,
            modelprefix=modelprefix,
            robust_nframes=robust_nframes,
            allow_growth=allow_growth,
            use_shelve=use_shelve,
            auto_track=auto_track,
            n_tracks=n_tracks,
            calibrate=calibrate,
            identity_only=identity_only,
            **kwargs,
        )
    elif engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis import analyze_videos
        _update_device(gputouse, torch_kwargs)

        if use_shelve:
            raise NotImplementedError(
                f"The 'use_shelve' option is not yet implemented with {engine}"
            )

        return analyze_videos(
            config,
            videos=videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            destfolder=destfolder,
            batchsize=batchsize,
            modelprefix=modelprefix,
            auto_track=auto_track,
            identity_only=identity_only,
            overwrite=False,
            **torch_kwargs,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def create_tracking_dataset(
    config: str,
    videos: list[str],
    track_method: str,
    videotype: str = "",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: int | None = None,
    destfolder: str | None = None,
    batchsize: int | None = None,
    cropping: list[int] | None = None,
    TFGPUinference: bool = True,
    dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    modelprefix: str = "",
    robust_nframes: bool = False,
    n_triplets: int = 1000,
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import create_tracking_dataset
        return create_tracking_dataset(
            config,
            videos,
            track_method,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=gputouse,
            save_as_csv=False,  # not used in method
            destfolder=destfolder,
            batchsize=batchsize,
            cropping=cropping,
            TFGPUinference=TFGPUinference,
            dynamic=dynamic,
            modelprefix=modelprefix,
            robust_nframes=robust_nframes,
            n_triplets=n_triplets,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def analyze_time_lapse_frames(
    config: str,
    directory: str,
    frametype: str = ".png",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: int | None = None,
    save_as_csv: bool = False,
    modelprefix: str = "",
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import analyze_time_lapse_frames
        return analyze_time_lapse_frames(
            config,
            directory,
            frametype=frametype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=gputouse,
            save_as_csv=save_as_csv,
            modelprefix=modelprefix,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def convert_detections2tracklets(
    config: str,
    videos: list[str],
    videotype: str = "",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    overwrite: bool = False,
    destfolder: str | None = None,
    ignore_bodyparts: list[str] | None = None,
    inferencecfg: dict | None = None,
    modelprefix: str = "",
    greedy: bool = False,
    calibrate: bool = False,
    window_size: int = 0,
    identity_only: int = False,
    track_method: str = "",
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import convert_detections2tracklets
        return convert_detections2tracklets(
            config,
            videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            overwrite=overwrite,
            destfolder=destfolder,
            ignore_bodyparts=ignore_bodyparts,
            inferencecfg=inferencecfg,
            modelprefix=modelprefix,
            greedy=greedy,
            calibrate=calibrate,
            window_size=window_size,
            identity_only=identity_only,
            track_method=track_method,
        )

    elif engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis import convert_detections2tracklets

        if greedy or calibrate or window_size:
            raise NotImplementedError(
                f"The 'greedy', 'calibrate' and 'window_size' option are not yet "
                f"implemented with {engine}"
            )

        return convert_detections2tracklets(
            config,
            videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            overwrite=overwrite,
            destfolder=destfolder,
            ignore_bodyparts=ignore_bodyparts,
            inferencecfg=inferencecfg,
            modelprefix=modelprefix,
            identity_only=identity_only,
            track_method=track_method,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def extract_maps(
    config,
    shuffle: int = 0,
    trainingsetindex: int = 0,
    gputouse: int | None = None,
    rescale: bool = False,
    Indices: list[int] | None = None,
    modelprefix: str = "",
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import extract_maps
        return extract_maps(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=gputouse,
            rescale=rescale,
            Indices=Indices,
            modelprefix=modelprefix,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def visualize_scoremaps(
    image: np.ndarray, scmap: np.ndarray, engine: Engine = DEFAULT_ENGINE,
):
    if engine == Engine.TF:
        # TODO: also works for Pytorch, but should not import as then requires TF
        from deeplabcut.pose_estimation_tensorflow import visualize_scoremaps
        return visualize_scoremaps(image, scmap)

    raise NotImplementedError(f"This function is not implemented for {engine}")


def visualize_locrefs(
    image: np.ndarray,
    scmap: np.ndarray,
    locref_x: np.ndarray,
    locref_y: np.ndarray,
    step: int = 5,
    zoom_width: int = 0,
    engine: Engine = DEFAULT_ENGINE,
):
    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import visualize_locrefs
        return visualize_locrefs(image, scmap, locref_x, locref_y, step=step, zoom_width=zoom_width)

    raise NotImplementedError(f"This function is not implemented for {engine}")


def visualize_paf(
    image: np.ndarray,
    paf: np.ndarray,
    step: int = 5,
    colors: list | None = None,
    engine: Engine = DEFAULT_ENGINE,
):
    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import visualize_paf
        return visualize_paf(image, paf, step=step, colors=colors)

    raise NotImplementedError(f"This function is not implemented for {engine}")


def extract_save_all_maps(
    config,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    comparisonbodyparts: str | list[str] = "all",
    extract_paf: bool = True,
    all_paf_in_one: bool = True,
    gputouse: int = None,
    rescale: bool = False,
    Indices: list[int] | None = None,
    modelprefix: str = "",
    dest_folder: str = None,
):
    engine = get_shuffle_engine(
        _load_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import extract_save_all_maps
        return extract_save_all_maps(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            comparisonbodyparts=comparisonbodyparts,
            extract_paf=extract_paf,
            all_paf_in_one=all_paf_in_one,
            gputouse=gputouse,
            rescale=rescale,
            Indices=Indices,
            modelprefix=modelprefix,
            dest_folder=dest_folder,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def export_model(
    cfg_path: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshotindex: int | None = None,
    iteration: int = None,
    TFGPUinference: bool = True,
    overwrite: bool = False,
    make_tar: bool = True,
    wipepaths: bool = False,
    modelprefix: str = "",
):
    engine = get_shuffle_engine(
        _load_config(cfg_path),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import export_model
        return export_model(
            cfg_path=cfg_path,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            snapshotindex=snapshotindex,
            iteration=iteration,
            TFGPUinference=TFGPUinference,
            overwrite=overwrite,
            make_tar=make_tar,
            wipepaths=wipepaths,
            modelprefix=modelprefix,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def _update_device(gpu_to_use: int | None, torch_kwargs: dict) -> None:
    if "device" not in torch_kwargs and gpu_to_use is not None:
        if isinstance(gpu_to_use, int):
            torch_kwargs["device"] = f"cuda:{gpu_to_use}"
        else:
            torch_kwargs["device"] = gpu_to_use


def _load_config(config: str) -> dict:
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config {config} is not found. Please make sure that the file exists."
        )

    with open(config, "r") as f:
        project_config = YAML(typ="safe", pure=True).load(f)

    return project_config
