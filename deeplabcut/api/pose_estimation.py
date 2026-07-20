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

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import albumentations as A

    from deeplabcut.core.config import ProjectConfig
    from deeplabcut.pose_estimation_pytorch.data.ctd import CondFromModel
    from deeplabcut.pose_estimation_pytorch.runners import CTDTrackingConfig
    from deeplabcut.pose_estimation_pytorch.runners.inference import InferenceConfig


from deeplabcut.api._tf_routing import with_tensorflow_fallback
from deeplabcut.core.deprecation import deprecated, renamed_parameter


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
    config: ProjectConfig | dict | Path | str,
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
) -> None:
    """Trains a network for a project.

    Args:
        config: Path to the yaml config file of the project, or a ``ProjectConfig``
            instance.
        shuffle: Index of the shuffle to train on.
        trainingsetindex: Training set index.
        modelprefix: Directory containing the deeplabcut configuration files to use
            to train the network (and where snapshots will be saved). By default, they
            are assumed to exist in the project folder.
        device: The torch device to train on (such as ``"cpu"``, ``"cuda"``, ``"mps"``).
        snapshot_path: If resuming training, the snapshot from which to resume.
        detector_path: If resuming training of a top-down model, used to specify the
            detector snapshot from which to resume.
        load_head_weights: If resuming training of a pose estimation model (either
            through the ``snapshot_path`` attribute or the ``resume_training_from`` key
            in the ``pytorch_config.yaml`` file), setting this to ``True`` also loads
            the weights for the model head. Note that if you change the number of
            bodyparts, you need to set this to ``False`` for re-training.
        batch_size: Overrides the batch size to train with.
        epochs: Overrides the maximum number of epochs to train the model for.
        save_epochs: Overrides the number of epochs between each snapshot save.
        detector_batch_size: Only for top-down models. Overrides the batch size with
            which to train the detector.
        detector_epochs: Only for top-down models. Overrides the maximum number of
            epochs to train the model for. Setting to ``0`` means the detector will not
            be trained.
        detector_save_epochs: Only for top-down models. Overrides the number of epochs
            between each snapshot of the detector is saved.
        display_iters: Overrides the number of iterations between each log of the loss
            within an epoch.
        max_snapshots_to_keep: The maximum number of snapshots to save for each model.
        pose_threshold: Used for memory-replay. Pseudo-predictions with confidence
            lower than this threshold are discarded for memory-replay.
        pytorch_cfg_updates: A dictionary of updates to the pytorch config. The keys
            are the dot-separated paths to the values to update in the config. For
            example, to update the GPUs to run the training on::

                pytorch_cfg_updates = {"runner.gpus": [0, 1, 2, 3]}

            To see the full list, check the ``pytorch_cfg.yaml`` file in your project
            folder.
    """
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
    config: ProjectConfig | dict | Path | str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
) -> tuple[Path, Path, Path]:
    """Returns the paths to the training config, test config and snapshot folder.

    Args:
        config: Full path of the config.yaml file as a string, or a ``ProjectConfig``
            instance.
        shuffle: The shuffle index to select for training.
        trainingsetindex: Which ``TrainingsetFraction`` to use (note that
            ``TrainingFraction`` is a list in ``config.yaml``).
        modelprefix: The modelprefix for the model.

    Returns:
        A 3-tuple of:
            - Path to the training ``pytorch_config.yaml`` file.
            - Path to the test ``pose_cfg.yaml`` file.
            - Path to the folder containing the snapshots.
    """
    from deeplabcut.pose_estimation_pytorch.apis.utils import (
        return_train_network_path as _return_train_network_path,
    )

    return _return_train_network_path(
        config,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
    )


@with_tensorflow_fallback(normalize_gputouse=True, dropped_params=["rescale"])
@renamed_parameter(old="Shuffles", new="shuffles", since="3.0.0")
@renamed_parameter(old="comparisonbodyparts", new="comparison_bodyparts", since="3.0.0")
def evaluate_network(
    config: ProjectConfig | dict | Path | str,
    shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    snapshotindex: int | str | None = None,
    device: str | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    transform: A.Compose | None = None,
    snapshots_to_evaluate: list[str] | None = None,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    modelprefix: str = "",
    detector_snapshot_index: int | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
) -> None:
    """Evaluates a snapshot.

    The evaluation results are stored in the ``.h5`` and ``.csv`` file under the
    subdirectory ``evaluation_results``.

    Args:
        config: Path to the project's config file, or a ``ProjectConfig`` instance.
        shuffles: Iterable of integers specifying the shuffle indices to evaluate.
        trainingsetindex: Integer specifying which training set fraction to use.
            Evaluates all fractions if set to ``"all"``.
        snapshotindex: Index (starting at 0) of the snapshot to load. To evaluate the
            last one, use ``-1``. To evaluate all snapshots, use ``"all"``. If ``None``,
            the snapshotindex is loaded from the project configuration.
        device: The device to run evaluation on.
        plotting: Plots the predictions on the train and test images. If provided it
            must be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``.
            Setting to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
        show_errors: Display train and test errors.
        transform: Transformation pipeline for evaluation. Should normalise the data
            the same way it was normalised during training.
        snapshots_to_evaluate: List of snapshot names to evaluate (e.g.
            ``["snapshot-50", "snapshot-75"]``). If defined, ``snapshotindex`` will be
            ignored.
        comparison_bodyparts: A subset of the bodyparts for which to compute the
            evaluation metrics.
        per_keypoint_evaluation: Compute the train and test RMSE for each keypoint and
            save the results to a ``{model_name}-keypoint-results.csv`` in the
            ``evaluation-results-pytorch`` folder.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        detector_snapshot_index: Only for TD models. If defined, uses the detector with
            the given index for pose estimation.
        pcutoff: The cutoff to use for computing evaluation metrics. When ``None``, the
            cutoff will be loaded from the project config. If a list is provided, there
            should be one value for each bodypart and one for each unique bodypart (if
            any). If a dict is provided, the keys should be bodyparts mapping to pcutoff
            values; bodyparts not in the dict default to ``0.6``.

    Examples:
        Evaluate shuffle 1 without plotting predictions:

        >>> import deeplabcut
        >>> deeplabcut.evaluate_network(
        ...     "/analysis/project/reaching-task/config.yaml",
        ...     shuffles=[1],
        ... )

        Evaluate shuffles 0 and 1 and plot the predictions:

        >>> deeplabcut.evaluate_network(
        ...     "/analysis/project/reaching-task/config.yaml",
        ...     shuffles=[0, 1],
        ...     plotting=True,
        ... )

        Plot assemblies for a maDLC project:

        >>> deeplabcut.evaluate_network(
        ...     "/analysis/project/reaching-task/config.yaml",
        ...     shuffles=[1],
        ...     plotting="individual",
        ... )
    """
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
    config: ProjectConfig | dict | Path | str,
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
    """Deprecated TensorFlow-only function."""
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
    config: ProjectConfig | dict | Path | str,
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
) -> str:
    """Makes predictions based on a trained network.

    The index of the trained network is specified by parameters in the config file
    (in particular the variable ``snapshot_index``).

    Args:
        config: Full path of the ``config.yaml`` file for the project, or a
            ``ProjectConfig`` instance.
        videos: A str (or list of strings) containing the full paths to videos for
            analysis or a path to the directory where all the videos with the same
            extension are stored.
        video_extensions: Controls how ``videos`` are filtered, based on file extension.
            File paths and directory contents are treated differently:

            - ``None`` (default): file paths are accepted as-is; directories are
              scanned for files with a recognized video extension.
            - ``str`` or ``Sequence[str]`` (e.g. ``"mp4"`` or ``["mp4", "avi"]``):
              both file paths and directory contents are filtered by the given
              extension(s).
        shuffle: An integer specifying the shuffle index of the training dataset used
            for training the network.
        trainingsetindex: Integer specifying which ``TrainingsetFraction`` to use.
        save_as_csv: For multi-animal projects and when ``auto_track=True``, passed
            along to the ``stitch_tracklets`` method to save tracks as CSV.
        in_random_order: Whether to analyze videos in a random order. Only relevant
            when specifying a video directory in ``videos``.
        device: The device to use for video analysis.
        destfolder: Specifies the destination folder for analysis data. If ``None``,
            the path of the video is used. Note that for subsequent analysis this
            folder also needs to be passed.
        snapshot_index: Index (starting at 0) of the snapshot to use to analyze the
            videos. Use ``-1`` for the last snapshot. If ``None``, the snapshot index is
            loaded from the project configuration.
        detector_snapshot_index: Only for top-down models. Index of the detector
            snapshot to use, used in the same way as ``snapshot_index``.
        dynamic: ``(state, detection_threshold, margin)`` triplet. If the state is
            ``True``, dynamic cropping will be performed.
        ctd_conditions: Only for CTD models. If ``None``, the configuration for the
            condition provider will be loaded from the pytorch config file. If given as
            a dict, a ``CondFromModel`` is created from it. Example::

                ctd_conditions = {"shuffle": 17, "snapshot": "snapshot-best-190.pt"}

        ctd_tracking: Only for CTD models. When ``True`` (or a ``CTDTrackingConfig``),
            poses from frame T are used as conditions for frame T+1, enabling
            direct tracking without a separate BU model per frame.
        top_down_dynamic: Configuration for a top-down dynamic cropper. If ``None``,
            top-down dynamic cropping is not used.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        batch_size: The batch size to use for inference. Defaults to the value from the
            project config.
        detector_batch_size: The batch size to use for detector inference. Defaults to
            the value from the project config.
        transform: Optional custom transforms to apply to the video.
        overwrite: Overwrite any existing video analysis results.
        use_shelve: When ``True``, data are written to disk on the fly using a
            pickle-based persistent database. Default is to dump all data at the end.
        robust_nframes: Evaluate a video's number of frames in a robust manner. Slower
            (reads the full video frame-by-frame) but robust against file corruption.
        auto_track: By default, tracking and stitching are automatically performed,
            producing the final h5 data file. If ``False``, one must run
            ``convert_detections2tracklets`` and ``stitch_tracklets`` afterwards.
        n_tracks: Number of tracks to reconstruct. Defaults to the number of
            individuals defined in the ``config.yaml``.
        animal_names: Names to give to individuals in the labeled data file.
        calibrate: Whether to calibrate the assembly step.
        identity_only: If ``True`` and animal identity was learned by the model,
            assembly and tracking rely exclusively on identity prediction.
        cropping: List of cropping coordinates as ``[x1, x2, y1, y2]``. The same
            cropping parameters will be used for all videos.
        save_as_df: Saves the video predictions (before tracking) to an H5 file
            containing a pandas DataFrame. Cannot be used with ``use_shelve=True``.
        show_gpu_memory: When ``True``, the tqdm progress bar shows GPU memory usage.
        inference_cfg: ``InferenceConfig`` to use. If ``None``, the configuration from
            the ``pytorch_cfg.yaml`` will be used.

    Returns:
        The scorer name used to analyze the videos.
    """
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
    config: ProjectConfig | dict | Path | str,
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
    """Creates a tracking dataset to train a ReID tracklet stitcher.

    Args:
        config: Full path of the ``config.yaml`` file for the project, or a
            ``ProjectConfig`` instance.
        videos: A list of full paths to videos from which to create the tracking
            dataset, or a path to a directory where all videos with the same extension
            are stored.
        track_method: Specifies the tracker used to generate the pose estimation data.
            Must be either ``'box'``, ``'skeleton'``, or ``'ellipse'``.
        video_extensions: Controls how ``videos`` are filtered, based on file extension.
        shuffle: An integer specifying the shuffle index of the training dataset used
            for training the network.
        trainingsetindex: Integer specifying which ``TrainingsetFraction`` to use.
        destfolder: Specifies the destination folder for the tracking data. If
            ``None``, the path of the video is used.
        batch_size: The batch size to use for inference. Defaults to the value from the
            project config.
        detector_batch_size: The batch size to use for detector inference. Defaults to
            the value from the project config.
        cropping: List of cropping coordinates as ``[x1, x2, y1, y2]``.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        robust_nframes: Evaluate a video's number of frames in a robust manner.
        n_triplets: The number of triplets to extract for the dataset.

    Returns:
        The scorer name used to analyze the videos.
    """
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
    config: ProjectConfig | dict | Path | str,
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
    """Runs analysis on images using a pose model.

    Args:
        config: The project configuration file path or a ``ProjectConfig`` instance.
        images: The image(s) to run inference on. Can be the path to an image, the path
            to a directory containing images, or a list of image paths or directories
            containing images.
        frame_type: Filters the images to analyze to only those with the given suffix
            (e.g. ``".png"`` analyzes only PNG images). By default, all ``".jpg"``,
            ``".jpeg"`` and ``".png"`` images are analyzed.
        output_dir: The directory where the predictions will be stored.
        shuffle: The shuffle for which to run image analysis.
        trainingsetindex: The trainingsetindex for which to run image analysis.
        snapshot_index: The index of the snapshot to use. Loaded from the project
            configuration file if ``None``.
        detector_snapshot_index: For top-down models only. The index of the detector
            snapshot to use. Loaded from the project configuration file if ``None``.
        modelprefix: The model prefix used for the shuffle.
        device: The device to use to run image analysis.
        max_individuals: The maximum number of individuals to detect in each image.
            Defaults to the number of individuals in the project.
        save_as_csv: Whether to also save the predictions as a CSV file.
        progress_bar: Whether to display a progress bar when running inference.
        plotting: Whether to plot predictions on images.
        pcutoff: The cutoff score when plotting pose predictions. Must be in ``(0, 1)``
            or ``None``. If ``None``, the pcutoff is read from the project configuration
            file.
        bbox_pcutoff: The cutoff score when plotting bounding box predictions. Must be
            in ``(0, 1)`` or ``None``. If ``None``, read from the project configuration
            file.
        plot_skeleton: If a skeleton is defined in the model configuration file, whether
            to plot the skeleton connecting the predicted bodyparts on the images.
        ctd_conditions: Only for CTD models. If ``None``, the configuration for the
            condition provider will be loaded from the pytorch config file. Example::

                ctd_conditions = {"shuffle": 17, "snapshot": "snapshot-best-190.pt"}

    Returns:
        A dictionary mapping each image filename to the different types of predictions
        for it (e.g. ``"bodyparts"``, ``"unique_bodyparts"``, ``"bboxes"``,
        ``"bbox_scores"``).
    """
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
    config: ProjectConfig | dict | Path | str,
    directory: str,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    device: str | None = None,
    save_as_csv: bool = False,
    modelprefix: str = "",
):
    """Deprecated shim for ``deeplabcut.analyze_images``.

    Args:
        config: Full path of the ``config.yaml`` file for the project, or a
            ``ProjectConfig`` instance.
        directory: Directory containing the frames to analyze.
        shuffle: The shuffle index of the model to use.
        trainingsetindex: Which ``TrainingsetFraction`` to use.
        device: The device to use for inference.
        save_as_csv: Whether to save predictions as a CSV file.
        modelprefix: The model prefix for the shuffle.
    """
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


@with_tensorflow_fallback(normalize_gputouse=True)
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def convert_detections2tracklets(
    config: ProjectConfig | dict | Path | str,
    videos: str | list[str],
    video_extensions: str | Sequence[str] | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    overwrite: bool = False,
    destfolder: str | None = None,
    ignore_bodyparts: list[str] | None = None,
    inferencecfg: dict | None = None,
    modelprefix="",
    identity_only=False,
    track_method="",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
):
    """Converts detections to tracklets for multi-animal projects.

    Args:
        config: Full path of the ``config.yaml`` file for the project, or a
            ``ProjectConfig`` instance.
        videos: A str (or list of strings) containing the full paths to videos for
            processing.
        video_extensions: Controls how ``videos`` are filtered, based on file extension.
        shuffle: An integer specifying the shuffle index of the training dataset.
        trainingsetindex: Integer specifying which ``TrainingsetFraction`` to use.
        overwrite: Whether to overwrite existing tracklet files.
        destfolder: Destination folder for tracklet data.
        ignore_bodyparts: Body parts to ignore during assembly.
        inferencecfg: Inference configuration dictionary.
        modelprefix: Directory containing the deeplabcut models to use.
        identity_only: If ``True`` and animal identity was learned by the model,
            assembly and tracking rely exclusively on identity prediction.
        track_method: The tracking method to use (``'box'``, ``'skeleton'``, or
            ``'ellipse'``).
        snapshot_index: Index of the snapshot to use. If ``None``, loaded from the
            project configuration.
        detector_snapshot_index: Only for TD models. Index of the detector snapshot to
            use. If ``None``, loaded from the project configuration.
    """
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
        identity_only=identity_only,
        track_method=track_method,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )


@with_tensorflow_fallback(normalize_gputouse=True, renamed_params={"Indices": "indices"})
def extract_maps(
    config: ProjectConfig | dict | Path | str,
    shuffle: int = 0,
    trainingsetindex: int | str = 0,
    device: str | None = None,
    rescale: bool = False,
    indices: list[int] | None = None,
    extract_paf: bool = True,
    modelprefix: str | None = "",
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
) -> dict:
    """Extracts the different maps output by DeepLabCut models.

    Extracts scoremaps, location refinement fields and part-affinity fields.

    Args:
        config: The project configuration, or a path to ``config.yaml``.
        shuffle: Index of the shuffle for which to extract maps.
        trainingsetindex: Integer specifying which ``TrainingsetFraction`` to use.
            Can also be set to ``"all"``.
        device: The device to use for inference.
        rescale: Evaluate the model at the ``global_scale`` variable (as set in the
            ``test/pose_config.yaml`` file for a particular project).
        indices: Optionally, only obtain maps for a subset of images in the dataset.
            The indices given here are the indices of the images for which maps will be
            extracted.
        extract_paf: Extract part affinity fields. Note that turning it off will make
            the function much faster.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        snapshot_index: Index (starting at 0) of the snapshot to extract maps from.
            Use ``-1`` for the last snapshot, ``"all"`` for all snapshots.
        detector_snapshot_index: Only for TD models. If defined, uses the detector
            with the given index for pose estimation.

    Returns:
        A dict indexed by ``(trainingset_fraction, snapshot_index, image_index)``. For
        each key the value is a tuple of
        ``(img, scmap, locref, paf, bpt_names, paf_graph, img_name, is_train)``.

    Examples:
        Extract maps for images 0 and 103 of the training set for shuffle 0:

        >>> deeplabcut.extract_maps(config, 0, indices=[0, 103])
    """
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


def visualize_scoremaps(
    image: np.ndarray,
    scmap: np.ndarray,
) -> tuple:
    """Plots scoremaps as an image overlay.

    Args:
        image: An image as a numpy array of shape ``(h, w, channels)``.
        scmap: A scoremap of shape ``(h, w)``.

    Returns:
        The figure and axis on which the image scoremap was plotted.
    """
    import deeplabcut.core.visualization as _visualization

    return _visualization.visualize_scoremaps(image, scmap)


def visualize_locrefs(
    image: np.ndarray,
    scmap: np.ndarray,
    locref_x: np.ndarray,
    locref_y: np.ndarray,
    step: int = 5,
    zoom_width: int = 0,
) -> tuple:
    """Plots a scoremap and the corresponding location refinement field on an image.

    Args:
        image: An image as a numpy array of shape ``(h, w, channels)``.
        scmap: A scoremap of shape ``(h, w)``.
        locref_x: The x-coordinate of the location refinement field, of shape ``(h, w)``.
        locref_y: The y-coordinate of the location refinement field, of shape ``(h, w)``.
        step: The step with which to plot the location refinement field.
        zoom_width: The zoom width with which to plot the scoremaps.

    Returns:
        The figure and axis on which the image scoremap and locref field were plotted.
    """
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
) -> tuple:
    """Plots the PAF on top of the image.

    Args:
        image: Shape ``(height, width, channels)``. The image on which the model was
            run.
        paf: Shape ``(height, width, 2 * len(paf_graph))``. The PAF output by the
            model.
        step: The step with which to plot the scoremaps.
        colors: The colormap to use.

    Returns:
        The figure and axis on which the image PAF was plotted.
    """
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
    config: ProjectConfig | dict | Path | str,
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
) -> None:
    """Extracts and saves scoremaps, locref fields and PAF predictions to disk.

    The maps are rescaled to the size of the input image and stored in the
    corresponding model folder in ``/evaluation-results-pytorch``.

    Args:
        config: The project configuration, or a path to ``config.yaml``.
        shuffle: Index of the shuffle for which to extract maps.
        trainingsetindex: Integer specifying which ``TrainingsetFraction`` to use.
        comparison_bodyparts: The body parts for which to compute the average error.
            Must be a subset of the body parts.
        extract_paf: Extract part affinity fields by default. Turning it off makes
            the function much faster.
        all_paf_in_one: By default, all part affinity fields are displayed on a single
            frame. If ``False``, individual fields are shown on separate frames.
        device: The device to use for inference.
        rescale: Evaluate the model at the ``global_scale`` variable.
        indices: Optionally, only obtain maps for a subset of images in the dataset.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        snapshot_index: Index (starting at 0) of the snapshot to extract maps from.
            Use ``-1`` for the last snapshot, ``"all"`` for all snapshots.
        detector_snapshot_index: Only for TD models. If defined, uses the detector
            with the given index for pose estimation.
        dest_folder: Destination folder for the saved maps. If ``None``, maps are
            saved in the default evaluation results folder.

    Examples:
        Calculate maps for images 0, 1 and 33:

        >>> deeplabcut.extract_save_all_maps(
        ...     "/analysis/project/reaching-task/config.yaml",
        ...     shuffle=1,
        ...     indices=[0, 1, 33],
        ... )
    """
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
    config: ProjectConfig | dict | Path | str,
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
    """Exports DeepLabCut models for live inference.

    Saves the ``pytorch_config.yaml`` configuration and snapshot files of the model to
    a directory named ``exported-models-pytorch`` within the project directory.

    Args:
        config: Path to the project configuration file, or a ``ProjectConfig``
            instance.
        shuffle: The shuffle of the model to export.
        trainingsetindex: The index of the training fraction for the model to export.
        snapshotindex: The snapshot index for the weights to export. If ``None``, uses
            the snapshotindex as defined in ``config.yaml``.
        detector_snapshot_index: Only for TD models. The detector snapshot index. If
            ``None``, uses the snapshotindex as defined in the project ``config.yaml``.
        iteration: The project iteration (active learning loop) to export. If ``None``,
            the iteration listed in the project config file is used.
        overwrite: Whether to overwrite if the model has already been exported.
        wipe_paths: Removes the actual path of your project and the ``init_weights``
            from the ``pytorch_config.yaml``.
        without_detector: Exports top-down models without the detector.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, the models are assumed to exist in the project
            folder.

    Raises:
        ValueError: If no snapshots could be found for the shuffle.
        ValueError: If a top-down model is exported but no detector snapshots are found.

    Examples:
        Export the last stored snapshot for the model trained with shuffle 3:

        >>> import deeplabcut
        >>> deeplabcut.export_model(
        ...     "/analysis/project/reaching-task/config.yaml",
        ...     shuffle=3,
        ...     snapshotindex=-1,
        ... )
    """
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
