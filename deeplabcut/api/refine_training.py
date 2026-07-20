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
"""DeepLabCut refine-training-dataset API"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from deeplabcut.core.deprecation import renamed_parameter


def find_outliers_in_raw_data(
    config: str | Path,
    pickle_file: str | Path,
    video_file: str | Path,
    pcutoff=0.1,
    percentiles=(5, 95),
    with_annotations=True,
    extraction_algo="kmeans",
    copy_videos=False,
):
    """Extract outlier frames from either raw detections or assemblies of multiple
    animals.

    Args:
        config (str | Path): Absolute path to the project config.yaml.
        pickle_file (str | Path): Path to a *_full.pickle or *_assemblies.pickle.
        video_file (str | Path): Path to the corresponding video file for frame extraction.
        pcutoff (float, optional): Detection confidence threshold below which frames are
            flagged as containing outliers. Only considered if raw detections are
            passed in. Defaults to 0.1.
        percentiles (tuple, optional): Assemblies are considered outliers if their areas
            are beyond the 5th and 95th percentiles. Must contain a lower and upper
            bound. Defaults to (5, 95).
        with_annotations (bool, optional): If true, extract frames and the corresponding
            network predictions. Otherwise, only the frames are extracted. Defaults to
            True.
        extraction_algo (string, optional): Outlier detection algorithm. Must be either
            ``uniform`` or ``kmeans``. Defaults to "kmeans".
        copy_videos (bool, optional): If True, newly-added videos (from which outlier
            frames are extracted) are copied to the project folder. By default, symbolic
            links are created instead. Defaults to False.
    """
    from deeplabcut.refine_training_dataset.outlier_frames import (
        find_outliers_in_raw_data as _find_outliers_in_raw_data,
    )

    return _find_outliers_in_raw_data(
        config=config,
        pickle_file=pickle_file,
        video_file=video_file,
        pcutoff=pcutoff,
        percentiles=percentiles,
        with_annotations=with_annotations,
        extraction_algo=extraction_algo,
        copy_videos=copy_videos,
    )


@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def extract_outlier_frames(
    config: str | Path,
    videos: list[str | Path],
    video_extensions: str | Sequence[str] | None = None,
    shuffle=1,
    trainingsetindex=0,
    outlieralgorithm="jump",
    frames2use=None,
    comparisonbodyparts="all",
    epsilon=20,
    p_bound=0.01,
    ARdegree=3,
    MAdegree=1,
    alpha=0.01,
    extractionalgorithm="kmeans",
    automatic=False,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=True,
    savelabeled=False,
    copy_videos=False,
    destfolder=None,
    modelprefix="",
    track_method="",
    **kwargs,
):
    """Extracts the outlier frames.

    Extracts the outlier frames if the predictions are not correct for a certain video
    from the cropped video running from start to stop as defined in config.yaml.

    Another crucial parameter in config.yaml is how many frames to extract
    ``numframes2extract``.

    Args:
        config (str | Path): Full path of the config.yaml file.
        videos (list[str | Path]): The full paths to videos for analysis or a path to the
            directory, where all the videos with same extension are stored.
        video_extensions (str | Sequence[str] | None, optional): Controls how ``videos`` are
            filtered, based on file extension. File paths and directory contents are
            treated differently:
            - ``None`` (default): file paths are accepted as-is; directories are
              scanned for files with a recognized video extension.
            - ``str`` or ``Sequence[str]`` (e.g. ``"mp4"`` or ``["mp4", "avi"]``):
              both file paths and directory contents are filtered by the given
              extension(s). Defaults to None.
        shuffle (int, optional): The shuffle index of training dataset. The extracted
            frames will be stored in the labeled-dataset for the corresponding shuffle
            of training dataset. Defaults to 1.
        trainingsetindex (int, optional): Integer specifying which TrainingsetFraction
            to use. Note that TrainingFraction is a list in config.yaml. Defaults to 0.
        outlieralgorithm (str, optional): String specifying the algorithm used to detect
            the outliers.

            * ``'fitting'`` fits an Auto Regressive Integrated Moving Average model to
              the data and computes the distance to the estimated data. Larger distances
              than epsilon are then potentially identified as outliers
            * ``'jump'`` identifies larger jumps than 'epsilon' in any body part
            * ``'uncertain'`` looks for frames with confidence below p_bound
            * ``'manual'`` launches a GUI from which the user can choose the frames
            * ``'list'`` looks for user to provide a list of frame numbers to use,
              'frames2use'. In this case, ``'extractionalgorithm'`` is forced to be
              ``'uniform.'``

            Defaults to "jump".
        frames2use (list[str], optional): If ``'outlieralgorithm'`` is ``'list'``,
            provide the list of frames here. Defaults to None.
        comparisonbodyparts (list[str] or str, optional): This selects the body parts for
            which the comparisons with the outliers are carried out. If ``"all"``, then
            all body parts from config.yaml are used. If a list of strings that are a
            subset of the full list E.g. ['hand','Joystick'] for the demo
            Reaching-Mackenzie-2018-08-30/config.yaml to select only these body parts.
            Defaults to "all".
        p_bound (float, optional): For outlieralgorithm ``'uncertain'`` this parameter
            defines the likelihood below which a body part will be flagged as a putative
            outlier. Defaults to 0.01.
        epsilon (float, optional): If ``'outlieralgorithm'`` is ``'fitting'``, this is
            the float bound according to which frames are picked when the (average) body
            part estimate deviates from model fit. If ``'outlieralgorithm'`` is
            ``'jump'``, this is the float bound specifying the distance by which body
            points jump from one frame to next (Euclidean distance). Defaults to 20.
        ARdegree (int, optional): For outlieralgorithm ``'fitting'``: Autoregressive
            degree of ARIMA model degree. (Note we use SARIMAX without exogeneous and
            seasonal part) See
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
            Defaults to 3.
        MAdegree (int, optional): For outlieralgorithm ``'fitting'``: Moving Average
            degree of ARIMA model degree. (Note we use SARIMAX without exogeneous and
            seasonal part) See
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
            Defaults to 1.
        alpha (float, optional): Significance level for detecting outliers based on
            confidence interval of fitted ARIMA model. Only the distance is used
            however. Defaults to 0.01.
        extractionalgorithm (str, optional): String specifying the algorithm to use for
            selecting the frames from the identified putatative outlier frames.
            Currently, deeplabcut supports either ``kmeans`` or ``uniform`` based
            selection (same logic as for extract_frames). Defaults to "kmeans".
        automatic (bool, optional): If ``True``, extract outliers without being asked
            for user feedback. Defaults to False.
        cluster_resizewidth (number, optional): If ``"extractionalgorithm"`` is
            ``"kmeans"``, one can change the width to which the images are downsampled
            (aspect ratio is fixed). Defaults to 30.
        cluster_color (bool, optional): If ``False``, each downsampled image is treated
            as a grayscale vector (discarding color information). If ``True``, then the
            color channels are considered. This increases the computational complexity.
            Defaults to False.
        opencv (bool, optional): Uses openCV for loading & extractiong (otherwise moviepy
            (legacy)). Defaults to True.
        savelabeled (bool, optional): If ``True``, frame are saved with predicted labels
            in each folder. Defaults to False.
        copy_videos (bool, optional): If True, newly-added videos (from which outlier
            frames are extracted) are copied to the project folder. By default, symbolic
            links are created instead. Defaults to False.
        destfolder (str or None, optional): Specifies the destination folder that was
            used for storing analysis data. If ``None``, the path of the video is used.
            Defaults to None.
        modelprefix (str, optional): Directory containing the deeplabcut models to use
            when evaluating the network. By default, the models are assumed to exist in
            the project folder. Defaults to "".
        track_method (str, optional): Specifies the tracker used to generate the data.
            Empty by default (corresponding to a single animal project). For multiple
            animals, must be either 'box', 'skeleton', or 'ellipse' and will be taken
            from the config.yaml file if none is given. Defaults to "".
        **kwargs: Additional arguments. For torch-based shuffles, can be used to specify:
            - snapshot_index
            - detector_snapshot_index

    Returns:
        None

    Examples:
        Extract the frames with default settings on Windows.

            deeplabcut.extract_outlier_frames(
                'C:\\myproject\\reaching-task\\config.yaml',
                ['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'],
            )

        Extract the frames with default settings on Linux/MacOS.

            deeplabcut.extract_outlier_frames(
                '/analysis/project/reaching-task/config.yaml',
                ['/analysis/project/video/reachinvideo1.avi'],
            )

        Extract the frames using the "kmeans" algorithm.

            deeplabcut.extract_outlier_frames(
                '/analysis/project/reaching-task/config.yaml',
                ['/analysis/project/video/reachinvideo1.avi'],
                extractionalgorithm='kmeans',
            )

        Extract the frames using the "kmeans" algorithm and ``"epsilon=5"`` pixels.

            deeplabcut.extract_outlier_frames(
                '/analysis/project/reaching-task/config.yaml',
                ['/analysis/project/video/reachinvideo1.avi'],
                epsilon=5,
                extractionalgorithm='kmeans',
            )
    """
    from deeplabcut.refine_training_dataset.outlier_frames import (
        extract_outlier_frames as _extract_outlier_frames,
    )

    return _extract_outlier_frames(
        config=config,
        videos=videos,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        outlieralgorithm=outlieralgorithm,
        frames2use=frames2use,
        comparisonbodyparts=comparisonbodyparts,
        epsilon=epsilon,
        p_bound=p_bound,
        ARdegree=ARdegree,
        MAdegree=MAdegree,
        alpha=alpha,
        extractionalgorithm=extractionalgorithm,
        automatic=automatic,
        cluster_resizewidth=cluster_resizewidth,
        cluster_color=cluster_color,
        opencv=opencv,
        savelabeled=savelabeled,
        copy_videos=copy_videos,
        destfolder=destfolder,
        modelprefix=modelprefix,
        track_method=track_method,
        **kwargs,
    )


def merge_datasets(config: str | Path, forceiterate=None):
    """Merge the original training dataset with the newly refined data.

    Checks if the original training dataset can be merged with the newly refined
    training dataset. To do so it will check if the frames in all extracted video sets
    were relabeled.

    If this is the case then the ``"iteration"`` variable is advanced by 1.

    Args:
        config (str | Path): Full path of the config.yaml file.
        forceiterate (int or None, optional): If an integer is given the iteration
            variable is set to this value. This is only done if all datasets were
            labeled or refined. Defaults to None.

    Examples:

            deeplabcut.merge_datasets("/analysis/project/reaching-task/config.yaml")
    """
    from deeplabcut.refine_training_dataset.outlier_frames import (
        merge_datasets as _merge_datasets,
    )

    return _merge_datasets(config=config, forceiterate=forceiterate)


@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def stitch_tracklets(
    config_path: str | Path,
    videos: list[str | Path],
    video_extensions: str | Sequence[str] | None = None,
    shuffle=1,
    trainingsetindex=0,
    n_tracks=None,
    animal_names: list[str] | None = None,
    min_length=10,
    split_tracklets=True,
    prestitch_residuals=True,
    max_gap=None,
    weight_func=None,
    destfolder=None,
    modelprefix="",
    track_method="",
    output_name="",
    transformer_checkpoint="",
    save_as_csv=False,
    **kwargs,
):
    """Stitch sparse tracklets into full tracks via a graph-based, minimum-cost flow
    optimization problem.

    Args:
        config_path (str | Path): Path to the main project config.yaml file.
        videos (list[str | Path]): Full paths to videos for analysis, or a directory where all videos
            with the same extension are stored.
        video_extensions (str | Sequence[str] | None, optional): Controls how ``videos`` are
            filtered, based on file extension. File paths and directory contents are
            treated differently:
            - ``None`` (default): file paths are accepted as-is; directories are
              scanned for files with a recognized video extension.
            - ``str`` or ``Sequence[str]`` (e.g. ``"mp4"`` or ``["mp4", "avi"]``):
              both file paths and directory contents are filtered by the given
              extension(s). Defaults to None.
        shuffle (int, optional): Shuffle index of the training dataset used for training the network. Defaults to 1.
        trainingsetindex (int, optional): Which TrainingsetFraction to use. By default the
            first (TrainingFraction is a list in config.yaml).
        n_tracks (int, optional): Number of tracks to reconstruct. By default, taken as the number
            of individuals defined in the config.yaml. Another number can be
            passed if the number of animals in the video is different from
            the number of animals the model was trained on.
        animal_names (list, optional): If you want the names given to individuals in the labeled data file, you can
            specify those names as a list here. If given and `n_tracks` is None, `n_tracks`
            will be set to `len(animal_names)`. If `n_tracks` is not None, then it must be
            equal to `len(animal_names)`. If it is not given, then `animal_names` will
            be loaded from the `individuals` in the project config.yaml file.
        min_length (int, optional): Tracklets less than `min_length` frames of length
            are considered to be residuals; i.e., they do not participate
            in building the graph and finding the solution to the
            optimization problem, but are rather added last after
            "almost-complete" tracks are formed. The higher the value,
            the lesser the computational cost, but the higher the chance of
            discarding relatively long and reliable tracklets that are
            essential to solving the stitching task.
            Default is 10, and must be 3 at least.
        split_tracklets (bool, optional): By default, tracklets whose time indices are not consecutive integers
            are split in shorter tracklets whose time continuity is guaranteed.
            This is for example very powerful to get rid of tracking errors
            (e.g., identity switches) which are often signaled by a missing
            time frame at the moment they occur. Note though that for long
            occlusions where tracker re-identification capability can be trusted,
            setting `split_tracklets` to False is preferable.
        prestitch_residuals (bool, optional): Residuals will by default be grouped together according to their
            temporal proximity prior to being added back to the tracks.
            This is done to improve robustness and simultaneously reduce complexity.
        max_gap (int, optional): Maximal temporal gap to allow between a pair of tracklets.
            This is automatically determined by the TrackletStitcher by default.
        weight_func (callable, optional): Function accepting two tracklets as arguments and returning a scalar
            that must be inversely proportional to the likelihood that the tracklets
            belong to the same track; i.e., the higher the confidence that the
            tracklets should be stitched together, the lower the returned value.
        destfolder (string, optional): Destination folder for analysis data (default is the path of the
            video). Note that for subsequent analysis this folder also needs to be passed.
        track_method (string, optional): Tracker used to generate the pose estimation data.
            For multiple animals, must be either 'box', 'skeleton', or 'ellipse'
            and will be taken from the config.yaml file if none is given.
        output_name (str, optional): Name of the output h5 file.
            By default, tracks are automatically stored into the same directory
            as the pickle file and with its name.
        transformer_checkpoint (str, optional): Path to transformer checkpoint for re-ID
            stitching. Defaults to "".
        save_as_csv (bool, optional): Whether to write the tracks to a CSV file too (False by default).
        kwargs (dict, optional): Additional arguments.
            For torch-based shuffles, can be used to specify:
                - snapshot_index
                - detector_snapshot_index

    Returns:
        TrackletStitcher: A TrackletStitcher object.
    """
    from deeplabcut.refine_training_dataset.stitch import stitch_tracklets as _stitch_tracklets

    return _stitch_tracklets(
        config_path=config_path,
        videos=videos,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        n_tracks=n_tracks,
        animal_names=animal_names,
        min_length=min_length,
        split_tracklets=split_tracklets,
        prestitch_residuals=prestitch_residuals,
        max_gap=max_gap,
        weight_func=weight_func,
        destfolder=destfolder,
        modelprefix=modelprefix,
        track_method=track_method,
        output_name=output_name,
        transformer_checkpoint=transformer_checkpoint,
        save_as_csv=save_as_csv,
        **kwargs,
    )
