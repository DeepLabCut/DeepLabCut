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

from pathlib import Path

import numpy as np

from deeplabcut.utils.deprecation import renamed_parameter



@renamed_parameter(old="maxiters", new="max_iters", since="3.0.0")
@renamed_parameter(old="saveiters", new="save_iters", since="3.0.0")
@renamed_parameter(old="displayiters", new="display_iters", since="3.0.0")
def train_network(
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    max_snapshots_to_keep: int | None = None,
    display_iters: int | None = None,
    save_iters: int | None = None,
    max_iters: int | None = None,
    epochs: int | None = None,
    save_epochs: int | None = None,
    allow_growth: bool = True,
    gputouse: str | None = None,
    autotune: bool = False,
    keepdeconvweights: bool = True,
    modelprefix: str = "",
    superanimal_name: str = "",
    superanimal_transfer_learning: bool = False,
    engine: Engine | None = None,
    device: str | None = None,
    snapshot_path: str | Path | None = None,
    detector_path: str | Path | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    detector_epochs: int | None = None,
    detector_save_epochs: int | None = None,
    pose_threshold: float | None = 0.1,
    pytorch_cfg_updates: dict | None = None,
):
    from deeplabcut.pose_estimation_pytorch.apis import train_network

    return train_network(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        device=device,
        snapshot_path=snapshot_path,
        detector_path=detector_path,
        load_head_weights=keepdeconvweights,
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


def return_train_network_path(
    config,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    engine: Engine | None = None,
) -> tuple[Path, Path, Path]:
    from deeplabcut.pose_estimation_pytorch.apis.utils import (
        return_train_network_path,
    )

    return return_train_network_path(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )


@renamed_parameter(old="comparisonbodyparts", new="comparison_bodyparts", since="3.0.0")
@renamed_parameter(old="Shuffles", new="shuffles", since="3.0.0")
def evaluate_network(
    config: str | Path,
    shuffles: Sequence[int] = (1,),
    trainingsetindex: int | str = 0,
    plotting: bool | str = False,
    show_errors: bool = True,
    comparison_bodyparts: str | list[str] = "all",
    gputouse: str | None = None,
    rescale: bool = False,
    modelprefix: str = "",
    per_keypoint_evaluation: bool = False,
    snapshots_to_evaluate: list[str] | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
    engine: Engine | None = None,
    **torch_kwargs,
):
   from deeplabcut.pose_estimation_pytorch.apis import evaluate_network
    return evaluate_network(
        config,
        shuffles=shuffles,
        trainingsetindex=trainingsetindex,
        plotting=plotting,
        show_errors=show_errors,
        comparison_bodyparts=comparison_bodyparts,
        snapshots_to_evaluate=snapshots_to_evaluate,
        per_keypoint_evaluation=per_keypoint_evaluation,
        modelprefix=modelprefix,
        pcutoff=pcutoff,
        **torch_kwargs,
    )


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
    engine: Engine | None = None,
):
    raise NotImplementedError(f"This function is not implemented for {engine}")


@renamed_parameter(old="batchsize", new="batch_size", since="3.0.0")
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def analyze_videos(
    config: str,
    videos: list[str],
    video_extensions: str | Sequence[str] | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: str | None = None,
    save_as_csv: bool = False,
    in_random_order: bool = True,
    destfolder: str | None = None,
    batch_size: int | None = None,
    cropping: list[int] | None = None,
    TFGPUinference: bool = True,
    dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    modelprefix: str = "",
    robust_nframes: bool = False,
    allow_growth: bool = False,
    use_shelve: bool = False,
    auto_track: bool = True,
    n_tracks: int | None = None,
    animal_names: list[str] | None = None,
    calibrate: bool = False,
    identity_only: bool = False,
    use_openvino: str | None = None,
    engine: Engine | None = None,
    **torch_kwargs,
):

    from deeplabcut.pose_estimation_pytorch.apis import analyze_videos

    return analyze_videos(
        config,
        videos=videos,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        save_as_csv=save_as_csv,
        in_random_order=in_random_order,
        destfolder=destfolder,
        dynamic=dynamic,
        modelprefix=modelprefix,
        use_shelve=use_shelve,
        robust_nframes=robust_nframes,
        auto_track=auto_track,
        n_tracks=n_tracks,
        animal_names=animal_names,
        calibrate=calibrate,
        identity_only=identity_only,
        overwrite=False,
        cropping=cropping,
        **torch_kwargs,
    )


@renamed_parameter(old="batchsize", new="batch_size", since="3.0.0")
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def create_tracking_dataset(
    config: str,
    videos: list[str],
    track_method: str,
    video_extensions: str | Sequence[str] | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: int | None = None,
    destfolder: str | None = None,
    batch_size: int | None = None,
    cropping: list[int] | None = None,
    TFGPUinference: bool = True,
    modelprefix: str = "",
    robust_nframes: bool = False,
    n_triplets: int = 1000,
    engine: Engine | None = None,
) -> str:


    from deeplabcut.pose_estimation_pytorch.apis import create_tracking_dataset

    return create_tracking_dataset(
        config,
        videos,
        track_method,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        destfolder=destfolder,
        batch_size=batch_size,
        cropping=cropping,
        modelprefix=modelprefix,
        robust_nframes=robust_nframes,
        n_triplets=n_triplets,
    )


def analyze_images(
    config: str | Path,
    images: str | Path | list[str] | list[Path],
    frame_type: str | None = None,
    destfolder: str | Path | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    max_individuals: int | None = None,
    device: str | None = None,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    save_as_csv: bool = False,
    modelprefix: str = "",
    plotting: bool | str = False,
    pcutoff: float | None = None,
    bbox_pcutoff: float | None = None,
    plot_skeleton: bool = False,
    **torch_kwargs,
) -> dict[str, dict[str, np.ndarray | np.ndarray]]:

    from deeplabcut.pose_estimation_pytorch import analyze_images

    return analyze_images(
        config=config,
        images=images,
        frame_type=frame_type,
        output_dir=destfolder,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        modelprefix=modelprefix,
        device=device,
        save_as_csv=save_as_csv,
        max_individuals=max_individuals,
        plotting=plotting,
        pcutoff=pcutoff,
        bbox_pcutoff=bbox_pcutoff,
        plot_skeleton=plot_skeleton,
        **torch_kwargs,
    )


def analyze_time_lapse_frames(
    config: str,
    directory: str,
    frametype: str = ".png",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: int | None = None,
    device: str | None = None,
    save_as_csv: bool = False,
    modelprefix: str = "",
    engine: Engine | None = None,
):


    from deeplabcut.pose_estimation_pytorch import analyze_images

    return analyze_images(
        config=config,
        images=directory,
        output_dir=directory,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        device=_gpu_to_use_to_device(gputouse, device),
        save_as_csv=save_as_csv,
        modelprefix=modelprefix,
    )


@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def convert_detections2tracklets(
    config: str,
    videos: list[str],
    video_extensions: str | Sequence[str] | None = None,
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
    engine: Engine | None = None,
):

    from deeplabcut.pose_estimation_pytorch.apis import convert_detections2tracklets
    return convert_detections2tracklets(
        config,
        videos,
        video_extensions=video_extensions,
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


def extract_maps(
    config,
    shuffle: int = 0,
    trainingsetindex: int = 0,
    gputouse: int | None = None,
    device: str | None = None,
    rescale: bool = False,
    Indices: list[int] | None = None,
    modelprefix: str = "",
    engine: Engine | None = None,
):
    from deeplabcut.pose_estimation_pytorch import extract_maps

    return extract_maps(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        device=_gpu_to_use_to_device(gputouse, device),
        rescale=rescale,
        indices=Indices,
        modelprefix=modelprefix,
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
    return _visualization.visualize_locrefs(image, scmap, locref_x, locref_y, step=step, zoom_width=zoom_width)


def visualize_paf(
    image: np.ndarray,
    paf: np.ndarray,
    step: int = 5,
    colors: list | None = None,
):
    import deeplabcut.core.visualization as _visualization
    return _visualization.visualize_paf(image, paf, step=step, colors=colors)


@renamed_parameter(old="comparisonbodyparts", new="comparison_bodyparts", since="3.0.0")
def extract_save_all_maps(
    config,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    comparison_bodyparts: str | list[str] = "all",
    extract_paf: bool = True,
    all_paf_in_one: bool = True,
    gputouse: int | None = None,
    device: str | None = None,
    rescale: bool = False,
    Indices: list[int] | None = None,
    modelprefix: str = "",
    dest_folder: str = None,
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
    engine: Engine | None = None,
):
    from deeplabcut.pose_estimation_pytorch import extract_save_all_maps

    return extract_save_all_maps(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        comparison_bodyparts=comparison_bodyparts,
        extract_paf=extract_paf,
        all_paf_in_one=all_paf_in_one,
        device=_gpu_to_use_to_device(gputouse, device),
        rescale=rescale,
        indices=Indices,
        modelprefix=modelprefix,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        dest_folder=dest_folder,
    )


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
    without_detector: bool = False,
    modelprefix: str = "",
    engine: Engine | None = None,
) -> None:
    from deeplabcut.pose_estimation_pytorch.apis.export import export_model

    return export_model(
        config=cfg_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        snapshotindex=snapshotindex,
        iteration=iteration,
        overwrite=overwrite,
        wipe_paths=wipepaths,
        without_detector=without_detector,
        modelprefix=modelprefix,
    )