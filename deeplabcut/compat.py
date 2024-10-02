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
"""Compatibility file for methods available with either PyTorch or Tensorflow"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from ruamel.yaml import YAML

from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine

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


def get_available_aug_methods(engine: Engine) -> tuple[str, ...]:
    """
    Args:
        engine: the engine for which augmentation methods should be returned

    Returns:
        the augmentations available for the given engine, where the first one is the
        default method to use

    Raises:
        RuntimeError: if no augmentations methods are defined for the given engine
    """
    if engine == Engine.TF:
        return "imgaug", "default", "deterministic", "scalecrop", "tensorpack"
    elif engine == Engine.PYTORCH:
        return ("albumentations", )

    raise RuntimeError(f"Unknown augmentation for engine: {engine}")


def train_network(
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    max_snapshots_to_keep: int | None = None,
    displayiters: int | None = None,
    saveiters: int | None = None,
    maxiters: int | None = None,
    allow_growth: bool = True,
    gputouse: str | None = None,
    autotune: bool = False,
    keepdeconvweights: bool = True,
    modelprefix: str = "",
    superanimal_name: str = "",
    superanimal_transfer_learning: bool = False,
    engine: Engine | None = None,
    **torch_kwargs,
):
    """
    Trains the network with the labels in the training dataset.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: int, optional, default=1
        Integer value specifying the shuffle index to select for training.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        Note that TrainingFraction is a list in config.yaml.

    max_snapshots_to_keep: int or None
        Sets how many snapshots are kept, i.e. states of the trained network. Every
        saving iteration many times a snapshot is stored, however only the last
        ``max_snapshots_to_keep`` many are kept! If you change this to None, then all
        are kept.
        See: https://github.com/DeepLabCut/DeepLabCut/issues/8#issuecomment-387404835

    displayiters: optional, default=None
        This variable is actually set in ``pose_config.yaml``. However, you can
        overwrite it with this hack. Don't use this regularly, just if you are too lazy
        to dig out the ``pose_config.yaml`` file for the corresponding project. If
        ``None``, the value from there is used, otherwise it is overwritten!

    saveiters: optional, default=None
        Only for the TensorFlow engine (for the PyTorch engine see the ``torch_kwargs``:
        you can use ``save_epochs``).
        This variable is actually set in ``pose_config.yaml``. However, you can
        overwrite it with this hack. Don't use this regularly, just if you are too lazy
        to dig out the ``pose_config.yaml`` file for the corresponding project.
        If ``None``, the value from there is used, otherwise it is overwritten!

    maxiters: optional, default=None
        Only for the TensorFlow engine (for the PyTorch engine see the ``torch_kwargs``:
        you can use ``epochs``).
        This variable is actually set in ``pose_config.yaml``. However, you can
        overwrite it with this hack. Don't use this regularly, just if you are too lazy
        to dig out the ``pose_config.yaml`` file for the corresponding project.
        If ``None``, the value from there is used, otherwise it is overwritten!

    allow_growth: bool, optional, default=True.
        Only for the TensorFlow engine.
        For some smaller GPUs the memory issues happen. If ``True``, the memory
        allocator does not pre-allocate the entire specified GPU memory region, instead
        starting small and growing as needed.
        See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2

    gputouse: optional, default=None
        Only for the TensorFlow engine (for the PyTorch engine see the ``torch_kwargs``:
        you can use ``device``).
        Natural number indicating the number of your GPU (see number in nvidia-smi).
        If you do not have a GPU put None.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    autotune: bool, optional, default=False
        Only for the TensorFlow engine.
        Property of TensorFlow, somehow faster if ``False``
        (as Eldar found out, see https://github.com/tensorflow/tensorflow/issues/13317).

    keepdeconvweights: bool, optional, default=True
        Only for the TensorFlow engine.
        Also restores the weights of the deconvolution layers (and the backbone) when
        training from a snapshot. Note that if you change the number of bodyparts, you
        need to set this to false for re-training.

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    superanimal_name: str, optional, default =""
        Only for the TensorFlow engine. For the PyTorch engine, you need to specify
        this through the ``weight_init`` when creating the training dataset.
        Specified if transfer learning with superanimal is desired

    superanimal_transfer_learning: bool, optional, default = False.
        Only for the TensorFlow engine. For the PyTorch engine, you need to specify
        this through the ``weight_init`` when creating the training dataset.
        If set true, the training is transfer learning (new decoding layer). If set
        false, and superanimal_name is True, then the training is fine-tuning (reusing
        the decoding layer)

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    torch_kwargs:
        You can add any keyword arguments for the deeplabcut.pose_estimation_pytorch
        train_network method here. These arguments are passed to the downstream method.
        Some of the parameters that can be passed are
            * ``device`` (the CUDA device to use for training)
            * ``epochs`` (maximum number of epochs to train the network for)
            * ``save_epochs`` (the number of epochs between each snapshot saved)
        `   * ``batch_size`` (the batch size to use while training).

        When training a top-down model, these parameters are also available for the
        detector, with the parameters ``detector_batch_size``, ``detector_epochs`` and
        ``detector_save_epochs``.

    Returns
    -------
    None

    Examples
    --------
    To train the network for first shuffle of the training dataset

    >>> deeplabcut.train_network('/analysis/project/reaching-task/config.yaml')

    To train the network for second shuffle of the training dataset

    >>> deeplabcut.train_network(
            '/analysis/project/reaching-task/config.yaml',
            shuffle=2,
            keepdeconvweights=True,
        )

    To train the network for shuffle created with a PyTorch engine, while overriding the
    number of epochs, batch size and other parameters.

    >>> deeplabcut.train_network(
            '/analysis/project/reaching-task/config.yaml',
            shuffle=1,
            batch_size=8,
            epochs=100,
            save_epochs=10,
            display_iters=50,
        )
    """
    if engine is None:
        engine = get_shuffle_engine(
            _load_config(config),
            trainingsetindex=trainingsetindex,
            shuffle=shuffle,
        )

    if engine == Engine.TF:
        from deeplabcut.pose_estimation_tensorflow import train_network
        if max_snapshots_to_keep is None:
            max_snapshots_to_keep = 5

        return train_network(
            str(config),
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
            superanimal_name=superanimal_name,
            superanimal_transfer_learning=superanimal_transfer_learning,
            modelprefix=modelprefix,
        )
    elif engine == Engine.PYTORCH:
        from deeplabcut.pose_estimation_pytorch.apis import train_network
        _update_device(gputouse, torch_kwargs)
        if "display_iters" not in torch_kwargs:
            torch_kwargs["display_iters"] = displayiters

        return train_network(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            modelprefix=modelprefix,
            max_snapshots_to_keep=max_snapshots_to_keep,
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
    """
    Returns the training and test pose config file names as well as the folder where the
    snapshot is

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: int
        Integer value specifying the shuffle index to select for training.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note
        that TrainingFraction is a list in config.yaml).

    modelprefix: str, optional
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    Returns the triple: trainposeconfigfile, testposeconfigfile, snapshotfolder
    """
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
    config: str | Path,
    Shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    plotting: bool | str = False,
    show_errors: bool = True,
    comparisonbodyparts: str | list[str] = "all",
    gputouse: str | None = None,
    rescale: bool = False,
    modelprefix: str = "",
    per_keypoint_evaluation: bool = False,
    snapshots_to_evaluate: list[str] | None = None,
    engine: Engine | None = None,
    **torch_kwargs,
):
    """Evaluates the network.

    Evaluates the network based on the saved models at different stages of the training
    network. The evaluation results are stored in the .h5 and .csv file under the
    subdirectory 'evaluation_results'. Change the snapshotindex parameter in the config
    file to 'all' in order to evaluate all the saved models.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file.

    Shuffles: list, optional, default=[1]
        List of integers specifying the shuffle indices of the training dataset.

    trainingsetindex: int or str, optional, default=0
        Integer specifying which "TrainingsetFraction" to use.
        Note that "TrainingFraction" is a list in config.yaml. This variable can also
        be set to "all".

    plotting: bool or str, optional, default=False
        Plots the predictions on the train and test images.
        If provided it must be either ``True``, ``False``, ``"bodypart"``, or
        ``"individual"``. Setting to ``True`` defaults as ``"bodypart"`` for
        multi-animal projects.

    show_errors: bool, optional, default=True
        Display train and test errors.

    comparisonbodyparts: str or list, optional, default="all"
        The average error will be computed for those body parts only.
        The provided list has to be a subset of the defined body parts.

    gputouse: int or None, optional, default=None
        Indicates the GPU to use (see number in ``nvidia-smi``). If you do not have a
        GPU put `None``.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    rescale: bool, optional, default=False
        Evaluate the model at the ``'global_scale'`` variable (as set in the
        ``pose_config.yaml`` file for a particular project). I.e. every image will be
        resized according to that scale and prediction will be compared to the resized
        ground truth. The error will be reported in pixels at rescaled to the
        *original* size. I.e. For a [200,200] pixel image evaluated at
        ``global_scale=.5``, the predictions are calculated on [100,100] pixel images,
        compared to 1/2*ground truth and this error is then multiplied by 2!.
        The evaluation images are also shown for the original size!

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    per_keypoint_evaluation: bool, default=False
        Compute the train and test RMSE for each keypoint, and save the results to
        a {model_name}-keypoint-results.csv in the evalution-results folder

    snapshots_to_evaluate: List[str], optional, default=None
        List of snapshot names to evaluate (e.g. ["snapshot-5000", "snapshot-7500"]).

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    torch_kwargs:
        You can add any keyword arguments for the deeplabcut.pose_estimation_pytorch
        evaluate_network function here. These arguments are passed to the downstream
        function. Available parameters are `snapshotindex`, which overrides the
        `snapshotindex` parameter in the project configuration file. For top-down models
        the `detector_snapshot_index` parameter can override the index of the detector
        to use for evaluation in the project configuration file.

    Returns
    -------
    None

    Examples
    --------
    If you do not want to plot and evaluate with shuffle set to 1.

    >>> deeplabcut.evaluate_network(
            '/analysis/project/reaching-task/config.yaml', Shuffles=[1],
        )

    If you want to plot and evaluate with shuffle set to 0 and 1.

    >>> deeplabcut.evaluate_network(
            '/analysis/project/reaching-task/config.yaml',
            Shuffles=[0, 1],
            plotting=True,
        )

    If you want to plot assemblies for a maDLC project

    >>> deeplabcut.evaluate_network(
            '/analysis/project/reaching-task/config.yaml',
            Shuffles=[1],
            plotting="individual",
        )

    Note: This defaults to standard plotting for single-animal projects.
    """
    if engine is None:
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
            str(config),
            Shuffles=Shuffles,
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=show_errors,
            comparisonbodyparts=comparisonbodyparts,
            gputouse=gputouse,
            rescale=rescale,
            modelprefix=modelprefix,
            per_keypoint_evaluation=per_keypoint_evaluation,
            snapshots_to_evaluate=snapshots_to_evaluate,
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
    engine: Engine | None = None,
):
    """
    Returns the results for (previously evaluated) network. deeplabcut.evaluate_network(..)
    Returns list of (per model): [trainingsiterations,trainfraction,shuffle,trainerror,testerror,pcutoff,trainerrorpcutoff,testerrorpcutoff,Snapshots[snapindex],scale,net_type]

    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.

    If fulldata=True, also returns (the complete annotation and prediction array)
    Returns list of: (DataMachine, Data, data, trainIndices, testIndices, trainFraction, DLCscorer,comparisonbodyparts, cfg, Snapshots[snapindex])
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: integer
        integers specifying shuffle index of the training dataset. The default is 0.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml). This
        variable can also be set to "all".

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    rescale: bool, default False
        Evaluate the model at the 'global_scale' variable (as set in the test/pose_config.yaml file for a particular project). I.e. every
        image will be resized according to that scale and prediction will be compared to the resized ground truth. The error will be reported
        in pixels at rescaled to the *original* size. I.e. For a [200,200] pixel image evaluated at global_scale=.5, the predictions are calculated
        on [100,100] pixel images, compared to 1/2*ground truth and this error is then multiplied by 2!. The evaluation images are also shown for the
        original size!

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    Examples
    --------
    If you do not want to plot
    >>> deeplabcut._evaluate_network_data('/analysis/project/reaching-task/config.yaml', shuffle=[1])
    --------
    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',shuffle=[1],plotting=True)
    """
    if engine is None:
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
    engine: Engine | None = None,
    **torch_kwargs,
):
    """Makes prediction based on a trained network.

    The index of the trained network is specified by parameters in the config file
    (in particular the variable 'snapshotindex').

    The labels are stored as MultiIndex Pandas Array, which contains the name of
    the network, body part name, (x, y) label position in pixels, and the
    likelihood for each frame per body part. These arrays are stored in an
    efficient Hierarchical Data Format (HDF) in the same directory where the video
    is stored. However, if the flag save_as_csv is set to True, the data can also
    be exported in comma-separated values format (.csv), which in turn can be
    imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    videos: list[str]
        A list of strings containing the full paths to videos for analysis or a path to
        the directory, where all the videos with same extension are stored.

    videotype: str, optional, default=""
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed. If left unspecified,
        videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional, default=1
        An integer specifying the shuffle index of the training dataset used for
        training the network.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int or None, optional, default=None
        Only for the TensorFlow engine (for the PyTorch engine see the ``torch_kwargs``:
        you can use ``device``).
        Indicates the GPU to use (see number in ``nvidia-smi``). If you do not have a
        GPU put ``None``.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional, default=False
        Saves the predictions in a .csv file.

    in_random_order: bool, optional (default=True)
        Only for the TensorFlow engine.
        Whether or not to analyze videos in a random order.
        This is only relevant when specifying a video directory in `videos`.

    destfolder: string or None, optional, default=None
        Specifies the destination folder for analysis data. If ``None``, the path of
        the video is used. Note that for subsequent analysis this folder also needs to
        be passed.

    batchsize: int or None, optional, default=None
        Currently not supported by the PyTorch engine.
        Change batch size for inference; if given overwrites value in ``pose_cfg.yaml``.

    cropping: list or None, optional, default=None
        Currently not supported by the PyTorch engine.
        List of cropping coordinates as [x1, x2, y1, y2].
        Note that the same cropping parameters will then be used for all videos.
        If different video crops are desired, run ``analyze_videos`` on individual
        videos with the corresponding cropping coordinates.

    TFGPUinference: bool, optional, default=True
        Only for the TensorFlow engine.
        Perform inference on GPU with TensorFlow code. Introduced in "Pretraining
        boosts out-of-domain robustness for pose estimation" by Alexander Mathis,
        Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis.
        Source: https://arxiv.org/abs/1909.11229

    dynamic: tuple(bool, float, int) triple containing (state, detectiontreshold, margin)
        Currently not supported by the PyTorch engine.
        If the state is true, then dynamic cropping will be performed. That means that
        if an object is detected (i.e. any body part > detectiontreshold), then object
        boundaries are computed according to the smallest/largest x position and
        smallest/largest y position of all body parts. This  window is expanded by the
        margin and from then on only the posture within this crop is analyzed (until the
        object is lost, i.e. <detectiontreshold). The current position is utilized for
        updating the crop window for the next frame (this is why the margin is important
        and should be set large enough given the movement of the animal).

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    robust_nframes: bool, optional, default=False
        Currently not supported by the PyTorch engine.
        Evaluate a video's number of frames in a robust manner.
        This option is slower (as the whole video is read frame-by-frame),
        but does not rely on metadata, hence its robustness against file corruption.

    allow_growth: bool, optional, default=False.
        Only for the TensorFlow engine.
        For some smaller GPUs the memory issues happen. If ``True``, the memory
        allocator does not pre-allocate the entire specified GPU memory region, instead
        starting small and growing as needed.
        See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2

    use_shelve: bool, optional, default=False
        Currently not supported by the PyTorch engine.
        By default, data are dumped in a pickle file at the end of the video analysis.
        Otherwise, data are written to disk on the fly using a "shelf"; i.e., a
        pickle-based, persistent, database-like object by default, resulting in
        constant memory footprint.

    The following parameters are only relevant for multi-animal projects:

    auto_track: bool, optional, default=True
        By default, tracking and stitching are automatically performed, producing the
        final h5 data file. This is equivalent to the behavior for single-animal
        projects.

        If ``False``, one must run ``convert_detections2tracklets`` and
        ``stitch_tracklets`` afterwards, in order to obtain the h5 file.

    This function has 3 related sub-calls:

    identity_only: bool, optional, default=False
        If ``True`` and animal identity was learned by the model, assembly and tracking
        rely exclusively on identity prediction.

    calibrate: bool, optional, default=False
        Currently not supported by the PyTorch engine.
        If ``True``, use training data to calibrate the animal assembly procedure. This
        improves its robustness to wrong body part links, but requires very little
        missing data.

    n_tracks: int or None, optional, default=None
        Number of tracks to reconstruct. By default, taken as the number of individuals
        defined in the config.yaml. Another number can be passed if the number of
        animals in the video is different from the number of animals the model was
        trained on.

    use_openvino: str, optional
        Only for the TensorFlow engine.
        Use "CPU" for inference if OpenVINO is available in the Python environment.

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    torch_kwargs:
        Any extra parameters to pass to the PyTorch API, such as ``device`` which can
        be used to specify the CUDA device to use for training.

    Returns
    -------
    DLCScorer: str
        the scorer used to analyze the videos

    Examples
    --------

    Analyzing a single video on Windows

    >>> deeplabcut.analyze_videos(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'],
        )

    Analyzing a single video on Linux/MacOS

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/reachingvideo1.avi'],
        )

    Analyze all videos of type ``avi`` in a folder

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos'],
            videotype='.avi',
        )

    Analyze multiple videos

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
        )

    Analyze multiple videos with ``shuffle=2``

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
            shuffle=2,
        )

    Analyze multiple videos with ``shuffle=2``, save results as an additional csv file

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
            shuffle=2,
            save_as_csv=True,
        )
    """
    if engine is None:
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

        if batchsize is not None:
            if "batch_size" in torch_kwargs:
                print(
                    f"You called analyze_videos with parameters ``batchsize={batchsize}"
                    f"`` and batch_size={torch_kwargs['batch_size']}. Only one is "
                    f"needed/used. Using batch size {torch_kwargs['batch_size']}")
            else:
                torch_kwargs["batch_size"] = batchsize

        return analyze_videos(
            config,
            videos=videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            save_as_csv=save_as_csv,
            destfolder=destfolder,
            modelprefix=modelprefix,
            auto_track=auto_track,
            identity_only=identity_only,
            overwrite=False,
            cropping=cropping,
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
    engine: Engine | None = None,
):
    if engine is None:
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
    engine: Engine | None = None,
):
    """
    Analyzed all images (of type = frametype) in a folder and stores the output in one file.

    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png
    """
    if engine is None:
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
    engine: Engine | None = None,
):
    """
    This should be called at the end of deeplabcut.analyze_videos for multianimal projects!

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    overwrite: bool, optional.
        Overwrite tracks file i.e. recompute tracks from full detections and overwrite.

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    ignore_bodyparts: optional
        List of body part names that should be ignored during tracking (advanced).
        By default, all the body parts are used.

    inferencecfg: Default is None.
        Configuration file for inference (assembly of individuals). Ideally
        should be obtained from cross validation (during evaluation). By default
        the parameters are loaded from inference_cfg.yaml, but these get_level_values
        can be overwritten.

    calibrate: bool, optional (default=False)
        If True, use training data to calibrate the animal assembly procedure.
        This improves its robustness to wrong body part links,
        but requires very little missing data.

    window_size: int, optional (default=0)
        Recurrent connections in the past `window_size` frames are
        prioritized during assembly. By default, no temporal coherence cost
        is added, and assembly is driven mainly by part affinity costs.

    identity_only: bool, optional (default=False)
        If True and animal identity was learned by the model,
        assembly and tracking rely exclusively on identity prediction.

    track_method: string, optional
         Specifies the tracker used to generate the pose estimation data.
         For multiple animals, must be either 'box', 'skeleton', or 'ellipse'
         and will be taken from the config.yaml file if none is given.

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    Examples
    --------
    If you want to convert detections to tracklets:
    >>> deeplabcut.convert_detections2tracklets('/analysis/project/reaching-task/config.yaml',[]'/analysis/project/video1.mp4'], videotype='.mp4')

    If you want to convert detections to tracklets based on box_tracker:
    >>> deeplabcut.convert_detections2tracklets('/analysis/project/reaching-task/config.yaml',[]'/analysis/project/video1.mp4'], videotype='.mp4',track_method='box')

    --------

    """
    if engine is None:
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
    engine: Engine | None = None,
):
    """
    Extracts the scoremap, locref, partaffinityfields (if available).

    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.

    Returns a dictionary indexed by: trainingsetfraction, snapshotindex, and imageindex
    for those keys, each item contains: (image,scmap,locref,paf,bpt names,partaffinity graph, imagename, True/False if this image was in trainingset)
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: integer
        integers specifying shuffle index of the training dataset. The default is 0.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml). This
        variable can also be set to "all".

    rescale: bool, default False
        Evaluate the model at the 'global_scale' variable (as set in the test/pose_config.yaml file for a particular project). I.e. every
        image will be resized according to that scale and prediction will be compared to the resized ground truth. The error will be reported
        in pixels at rescaled to the *original* size. I.e. For a [200,200] pixel image evaluated at global_scale=.5, the predictions are calculated
        on [100,100] pixel images, compared to 1/2*ground truth and this error is then multiplied by 2!. The evaluation images are also shown for the
        original size!

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    Examples
    --------
    If you want to extract the data for image 0 and 103 (of the training set) for model trained with shuffle 0.
    >>> deeplabcut.extract_maps(configfile,0,Indices=[0,103])

    """
    if engine is None:
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
    """
    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.
    """
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
    """
    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.
    """
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
    """
    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.
    """
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
    engine: Engine | None = None,
):
    """
    Extracts the scoremap, location refinement field and part affinity field prediction of the model. The maps
    will be rescaled to the size of the input image and stored in the corresponding model folder in /evaluation-results.

    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.

    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: integer
        integers specifying shuffle index of the training dataset. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml). This
        variable can also be set to "all".

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    extract_paf : bool
        Extract part affinity fields by default.
        Note that turning it off will make the function much faster.

    all_paf_in_one : bool
        By default, all part affinity fields are displayed on a single frame.
        If false, individual fields are shown on separate frames.

    Indices: default None
        For which images shall the scmap/locref and paf be computed? Give a list of images

    nplots_per_row: int, optional (default=None)
        Number of plots per row in grid plots. By default, calculated to approximate a squared grid of plots

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    Examples
    --------
    Calculated maps for images 0, 1 and 33.
    >>> deeplabcut.extract_save_all_maps('/analysis/project/reaching-task/config.yaml', shuffle=1,Indices=[0,1,33])

    """
    if engine is None:
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
    engine: Engine | None = None,
):
    """Export DeepLabCut models for the model zoo or for live inference.

    Saves the pose configuration, snapshot files, and frozen TF graph of the model to
    directory named exported-models within the project directory

    This function is only implemented for tensorflow models/shuffles, and will throw
    an error if called with a PyTorch shuffle.

    Parameters
    -----------

    cfg_path : string
        path to the DLC Project config.yaml file

    shuffle : int, optional
        the shuffle of the model to export. default = 1

    trainingsetindex : int, optional
        the index of the training fraction for the model you wish to export. default = 1

    snapshotindex : int, optional
        the snapshot index for the weights you wish to export. If None,
        uses the snapshotindex as defined in 'config.yaml'. Default = None

    iteration : int, optional
        The model iteration (active learning loop) you wish to export. If None,
        the iteration listed in the config file is used.

    TFGPUinference : bool, optional
        use the tensorflow inference model? Default = True
        For inference using DeepLabCut-live, it is recommended to set TFGPIinference=False

    overwrite : bool, optional
        if the model you wish to export has already been exported, whether to overwrite. default = False

    make_tar : bool, optional
        Do you want to compress the exported directory to a tar file? Default = True
        This is necessary to export to the model zoo, but not for live inference.

    wipepaths : bool, optional
        Removes the actual path of your project and the init_weights from pose_cfg.

    engine: Engine, optional, default = None.
        The default behavior loads the engine for the shuffle from the metadata. You can
        overwrite this by passing the engine as an argument, but this should generally
        not be done.

    Example:
    --------
    Export the first stored snapshot for model trained with shuffle 3:
    >>> deeplabcut.export_model('/analysis/project/reaching-task/config.yaml',shuffle=3, snapshotindex=-1)
    --------
    """
    if engine is None:
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
