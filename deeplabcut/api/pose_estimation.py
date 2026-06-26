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
"""DeepLabCut pose estimation API"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import albumentations as A

    from deeplabcut.pose_estimation_pytorch.data.ctd import CondFromModel
    from deeplabcut.pose_estimation_pytorch.runners import CTDTrackingConfig
    from deeplabcut.pose_estimation_pytorch.runners.inference import InferenceConfig


from deeplabcut.api._tf_routing import with_tensorflow_fallback
from deeplabcut.pose_estimation_pytorch.runners.inference import InferenceConfig
from deeplabcut.utils.deprecation import deprecated, renamed_parameter


@with_tensorflow_fallback(
    normalize_gputouse=True,
    dropped_params=[
        "allow_growth",
        "gputouse",
        "autotune",
        "superanimal_name",
        "superanimal_transfer_learning",
        "save_iters",
        "max_iters",
    ],
    renamed_params={"keepdeconvweights": "load_head_weights"},
)
@renamed_parameter(old="displayiters", new="display_iters", since="3.0.0")
def train_network(
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    device: str | None = None,
    snapshot_path: str | Path | None = None,
    detector_path: str | Path | None = None,
    load_head_weights: bool = True,
    batch_size: int | None = None,
    epochs: int | None = None,
    save_epochs: int | None = None,
    detector_batch_size: int | None = None,
    detector_epochs: int | None = None,
    detector_save_epochs: int | None = None,
    display_iters: int | None = None,
    max_snapshots_to_keep: int | None = None,
    pose_threshold: float | None = 0.1,
    pytorch_cfg_updates: dict | None = None,
):
    from deeplabcut.pose_estimation_pytorch.apis import train_network as _train_network

    return _train_network(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        device=device,
        snapshot_path=snapshot_path,
        detector_path=detector_path,
        load_head_weights=load_head_weights,
        batch_size=batch_size,
        epochs=epochs,
        save_epochs=save_epochs,
        detector_batch_size=detector_batch_size,
        detector_epochs=detector_epochs,
        detector_save_epochs=detector_save_epochs,
        display_iters=display_iters,
        max_snapshots_to_keep=max_snapshots_to_keep,
        pose_threshold=pose_threshold,
        pytorch_cfg_updates=pytorch_cfg_updates,
    )


@with_tensorflow_fallback
def return_train_network_path(
    config: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
) -> tuple[Path, Path, Path]:
    from deeplabcut.pose_estimation_pytorch.apis.utils import (
        return_train_network_path as _return_train_network_path,
    )

    return _return_train_network_path(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )


@renamed_parameter(old="Shuffles", new="shuffles", since="3.0.0")
@with_tensorflow_fallback(normalize_gputouse=True, dropped_params=["rescale"])
@renamed_parameter(old="comparisonbodyparts", new="comparison_bodyparts", since="3.0.0")
def evaluate_network(
    config: str | Path,
    shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    snapshotindex: int | str | None = None,
    device: str | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    transform: A.Compose = None,
    snapshots_to_evaluate: list[str] | None = None,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    modelprefix: str = "",
    detector_snapshot_index: int | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
):
    from deeplabcut.pose_estimation_pytorch.apis import evaluate_network as _evaluate_network

    return _evaluate_network(
        config=config,
        shuffles=shuffles,
        trainingsetindex=trainingsetindex,
        snapshotindex=snapshotindex,
        device=device,
        plotting=plotting,
        show_errors=show_errors,
        transform=transform,
        snapshots_to_evaluate=snapshots_to_evaluate,
        comparison_bodyparts=comparison_bodyparts,
        per_keypoint_evaluation=per_keypoint_evaluation,
        modelprefix=modelprefix,
        detector_snapshot_index=detector_snapshot_index,
        pcutoff=pcutoff,
    )


@with_tensorflow_fallback
@renamed_parameter(old="comparisonbodyparts", new="comparison_bodyparts", since="3.0.0")
@renamed_parameter(old="Snapindex", new="snapshotindex", since="3.0.0")
def return_evaluate_network_data(
    config: str,
    shuffle: int = 0,
    trainingsetindex: int = 0,
    comparison_bodyparts: str | list[str] = "all",
    snapshotindex: str | int | None = None,
    rescale: bool = False,
    fulldata: bool = False,
    show_errors: bool = True,
    modelprefix: str = "",
    returnjustfns: bool = True,
):
    """deprecated TensorFlow-only function"""
    raise NotImplementedError("This function is not implemented for PyTorch")


@with_tensorflow_fallback(
    normalize_gputouse=True,
    dropped_params=[
        "allow_growth",
        "TFGPUinference",
        "use_openvino",
    ],
)
@renamed_parameter(old="batchsize", new="batch_size", since="3.0.0")
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def analyze_videos(
    config: str,
    videos: str | list[str],
    video_extensions: str | Sequence[str] | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    save_as_csv: bool = False,
    in_random_order: bool = False,
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
    device: str | None = None,
    destfolder: str | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    ctd_conditions: dict | CondFromModel | None = None,
    ctd_tracking: bool | dict | CTDTrackingConfig = False,
    top_down_dynamic: dict | None = None,
    modelprefix: str = "",
    use_shelve: bool = False,
    robust_nframes: bool = False,
    transform: A.Compose | None = None,
    auto_track: bool | None = True,
    n_tracks: int | None = None,
    animal_names: list[str] | None = None,
    calibrate: bool = False,
    identity_only: bool | None = False,
    overwrite: bool = False,
    cropping: list[int] | None = None,
    save_as_df: bool = False,
    show_gpu_memory: bool = False,
    inference_cfg: InferenceConfig | dict | None = None,
):
    from deeplabcut.pose_estimation_pytorch.apis import analyze_videos as _analyze_videos

    return _analyze_videos(
        config=config,
        videos=videos,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        save_as_csv=save_as_csv,
        in_random_order=in_random_order,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        device=device,
        destfolder=destfolder,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        dynamic=dynamic,
        ctd_conditions=ctd_conditions,
        ctd_tracking=ctd_tracking,
        top_down_dynamic=top_down_dynamic,
        modelprefix=modelprefix,
        use_shelve=use_shelve,
        robust_nframes=robust_nframes,
        transform=transform,
        auto_track=auto_track,
        n_tracks=n_tracks,
        animal_names=animal_names,
        calibrate=calibrate,
        identity_only=identity_only,
        overwrite=overwrite,
        cropping=cropping,
        save_as_df=save_as_df,
        show_gpu_memory=show_gpu_memory,
        inference_cfg=inference_cfg,
    )


@with_tensorflow_fallback(normalize_gputouse=True, dropped_params=["TFGPUinference"])
@renamed_parameter(old="batchsize", new="batch_size", since="3.0.0")
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def create_tracking_dataset(
    config: str,
    videos: list[str] | list[Path],
    track_method: str,
    video_extensions: str | Sequence[str] | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    destfolder: str | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    cropping: list[int] | None = None,
    modelprefix: str = "",
    robust_nframes: bool = False,
    n_triplets: int = 1000,
) -> str:
    from deeplabcut.pose_estimation_pytorch.apis import create_tracking_dataset as _create_tracking_dataset

    return _create_tracking_dataset(
        config=config,
        videos=videos,
        track_method=track_method,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        destfolder=destfolder,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        cropping=cropping,
        modelprefix=modelprefix,
        robust_nframes=robust_nframes,
        n_triplets=n_triplets,
    )


@with_tensorflow_fallback(normalize_gputouse=True, dropped_params=["destfolder"])
def analyze_images(
    config: str | Path,
    images: str | Path | list[str] | list[Path],
    frame_type: str | None = None,
    output_dir: str | Path | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    modelprefix: str = "",
    device: str | None = None,
    max_individuals: int | None = None,
    save_as_csv: bool = False,
    progress_bar: bool = True,
    plotting: bool | str = False,
    pcutoff: float | None = None,
    bbox_pcutoff: float | None = None,
    plot_skeleton: bool = True,
    ctd_conditions: dict | CondFromModel | None = None,
) -> dict[str, dict[str, np.ndarray | np.ndarray]]:

    from deeplabcut.pose_estimation_pytorch import analyze_images as _analyze_images

    return _analyze_images(
        config=config,
        images=images,
        frame_type=frame_type,
        output_dir=output_dir,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        modelprefix=modelprefix,
        device=device,
        max_individuals=max_individuals,
        save_as_csv=save_as_csv,
        progress_bar=progress_bar,
        plotting=plotting,
        pcutoff=pcutoff,
        bbox_pcutoff=bbox_pcutoff,
        plot_skeleton=plot_skeleton,
        ctd_conditions=ctd_conditions,
    )


@deprecated(replacement="deeplabcut.analyze_images", since="3.1")
@with_tensorflow_fallback(normalize_gputouse=True, dropped_params=["frametype"])
def analyze_time_lapse_frames(
    config: str,
    directory: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    device: str | None = None,
    save_as_csv: bool = False,
    modelprefix: str = "",
):
    """Deprecated shim for deeplabcut.analyze_images"""
    from deeplabcut.pose_estimation_pytorch import analyze_images as _analyze_images

    return _analyze_images(
        config=config,
        images=directory,
        output_dir=directory,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        device=device,
        save_as_csv=save_as_csv,
        modelprefix=modelprefix,
    )


@with_tensorflow_fallback(normalize_gputouse=True, dropped_params=["destfolder"])
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def convert_detections2tracklets(
    config: str,
    videos: str | list[str],
    video_extensions: str | Sequence[str] | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    overwrite: bool = False,
    destfolder: str | None = None,
    ignore_bodyparts: list[str] | None = None,
    inferencecfg: dict | None = None,
    modelprefix="",
    # greedy: bool = False,  # TODO(niels): implement greedy assembly during video analysis
    # calibrate: bool = False,  # TODO(niels): implement assembly calibration during video analysis
    # window_size: int = 0,  # TODO(niels): implement window size selection for assembly during video analysis
    identity_only=False,
    track_method="",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
):
    from deeplabcut.pose_estimation_pytorch.apis import convert_detections2tracklets as _convert_detections2tracklets

    return _convert_detections2tracklets(
        config=config,
        videos=videos,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        overwrite=overwrite,
        destfolder=destfolder,
        ignore_bodyparts=ignore_bodyparts,
        inferencecfg=inferencecfg,
        modelprefix=modelprefix,
        # greedy=greedy,
        # calibrate=calibrate,
        # window_size=window_size,
        identity_only=identity_only,
        track_method=track_method,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )


@with_tensorflow_fallback(normalize_gputouse=True, renamed_params={"Indices": "indices"})
def extract_maps(
    config,
    shuffle: int = 0,
    trainingsetindex: int | str = 0,
    device: str | None = None,
    rescale: bool = False,
    indices: list[int] | None = None,
    extract_paf: bool = True,
    modelprefix: str | None = "",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
):
    from deeplabcut.pose_estimation_pytorch import extract_maps as _extract_maps

    return _extract_maps(
        config=config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        device=device,
        rescale=rescale,
        indices=indices,
        extract_paf=extract_paf,
        modelprefix=modelprefix,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )


def visualize_scoremaps(image: np.ndarray, scmap: np.ndarray):
    import deeplabcut.core.visualization as _visualization

    return _visualization.visualize_scoremaps(image, scmap)


def visualize_locrefs(
    image: np.ndarray,
    scmap: np.ndarray,
    locref_x: np.ndarray,
    locref_y: np.ndarray,
    step: int = 5,
    zoom_width: int = 0,
):
    import deeplabcut.core.visualization as _visualization

    return _visualization.visualize_locrefs(
        image=image,
        scmap=scmap,
        locref_x=locref_x,
        locref_y=locref_y,
        step=step,
        zoom_width=zoom_width,
    )


def visualize_paf(
    image: np.ndarray,
    paf: np.ndarray,
    step: int = 5,
    colors: list | None = None,
):
    import deeplabcut.core.visualization as _visualization

    return _visualization.visualize_paf(
        image=image,
        paf=paf,
        step=step,
        colors=colors,
    )


@with_tensorflow_fallback(normalize_gputouse=True, renamed_params={"Indices": "indices"})
@renamed_parameter(old="comparisonbodyparts", new="comparison_bodyparts", since="3.0.0")
def extract_save_all_maps(
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    comparison_bodyparts: str | list[str] = "all",
    extract_paf: bool = True,
    all_paf_in_one: bool = True,
    device: str | None = None,
    rescale: bool = False,
    indices: list[int] | None = None,
    modelprefix: str | None = "",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
    dest_folder: str | Path | None = None,
):
    from deeplabcut.pose_estimation_pytorch import extract_save_all_maps as _extract_save_all_maps

    return _extract_save_all_maps(
        config=config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        comparison_bodyparts=comparison_bodyparts,
        extract_paf=extract_paf,
        all_paf_in_one=all_paf_in_one,
        device=device,
        rescale=rescale,
        indices=indices,
        modelprefix=modelprefix,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        dest_folder=dest_folder,
    )


@with_tensorflow_fallback(
    normalize_gputouse=True,
    dropped_params=["TFGPUinference", "make_tar"],
    renamed_params={"wipepaths": "wipe_paths", "cfg_path": "config"},
)
def export_model(
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshotindex: int | None = None,
    detector_snapshot_index: int | None = None,
    iteration: int | None = None,
    overwrite: bool = False,
    wipe_paths: bool = False,
    without_detector: bool = False,
    modelprefix: str | None = None,
) -> None:
    from deeplabcut.pose_estimation_pytorch.apis.export import export_model

    return export_model(
        config=config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        snapshotindex=snapshotindex,
        detector_snapshot_index=detector_snapshot_index,
        iteration=iteration,
        overwrite=overwrite,
        wipe_paths=wipe_paths,
        without_detector=without_detector,
        modelprefix=modelprefix,
    )
