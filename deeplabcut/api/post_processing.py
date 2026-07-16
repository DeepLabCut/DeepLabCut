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
"""DeepLabCut post-processing API"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from deeplabcut.core.deprecation import renamed_parameter


@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def analyzeskeleton(
    config: str | Path,
    videos: list[str | Path],
    video_extensions: str | Sequence[str] | None = None,
    shuffle=1,
    trainingsetindex=0,
    filtered=False,
    save_as_csv=False,
    destfolder=None,
    modelprefix="",
    track_method="",
    return_data=False,
    **kwargs,
):
    """Extracts length and orientation of each "bone" of the skeleton.

    The bone and skeleton information is defined in the config file.

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
        filtered (bool, optional): Boolean variable indicating if filtered output should
            be plotted rather than frame-by-frame predictions. Filtered version can be
            calculated with ``deeplabcut.filterpredictions``. Defaults to False.
        save_as_csv (bool, optional): Saves the predictions in a .csv file. Defaults to
            False.
        destfolder (string or None, optional): Specifies the destination folder for
            analysis data. If ``None``, the path of the video is used. Note that for
            subsequent analysis this folder also needs to be passed. Defaults to None.
        modelprefix (str, optional): Directory containing the deeplabcut models to use
            when evaluating the network. By default, the models are assumed to exist in
            the project folder. Defaults to "".
        track_method (string, optional): Specifies the tracker used to generate the
            data. Empty by default (corresponding to a single animal project). For
            multiple animals, must be either 'box', 'skeleton', or 'ellipse' and will
            be taken from the config.yaml file if none is given. Defaults to "".
        return_data (bool, optional): If True, returns a dictionary of the filtered data
            keyed by video names. Defaults to False.
        **kwargs: Additional arguments. For torch-based shuffles, can be used to specify:
            - snapshot_index
            - detector_snapshot_index

    Returns:
        dict: Dictionary mapping video filepaths to skeleton dataframes.

        * If no videos exist, the dictionary will be empty.
        * If a video is not analyzed, the corresponding value in the dictionary will be
          None.
    """
    from deeplabcut.post_processing.analyze_skeleton import analyzeskeleton as _analyzeskeleton

    return _analyzeskeleton(
        config=config,
        videos=videos,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        filtered=filtered,
        save_as_csv=save_as_csv,
        destfolder=destfolder,
        modelprefix=modelprefix,
        track_method=track_method,
        return_data=return_data,
        **kwargs,
    )


@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def filterpredictions(
    config: str | Path,
    video: str | Path,
    video_extensions: str | Sequence[str] | None = None,
    shuffle=1,
    trainingsetindex=0,
    filtertype="median",
    windowlength=5,
    p_bound=0.001,
    ARdegree=3,
    MAdegree=1,
    alpha=0.01,
    save_as_csv=True,
    destfolder=None,
    modelprefix="",
    track_method="",
    return_data=False,
    **kwargs,
):
    """Fits frame-by-frame pose predictions.

    The pose predictions are fitted with ARIMA model (filtertype='arima') or median
    filter (default).

    Args:
        config (str | Path): Full path of the config.yaml file.
        video (str | Path): Full path of the video to filter. Make sure that this video is
            already analyzed.
        video_extensions (str | Sequence[str] | None, optional): Controls how ``videos`` are
            filtered, based on file extension. File paths and directory contents are
            treated differently:
            - ``None`` (default): file paths are accepted as-is; directories are
              scanned for files with a recognized video extension.
            - ``str`` or ``Sequence[str]`` (e.g. ``"mp4"`` or ``["mp4", "avi"]``):
              both file paths and directory contents are filtered by the given
              extension(s). Defaults to None.
        shuffle (int, optional): The shuffle index of training dataset. The extracted frames will be stored in
            the labeled-dataset for the corresponding shuffle of training dataset. Defaults to 1.
        trainingsetindex (int, optional): Integer specifying which TrainingsetFraction to use.
            Note that TrainingFraction is a list in config.yaml. Defaults to 0.
        filtertype (string, optional): The filter type - 'arima', 'median' or 'spline'. Defaults to "median".
        windowlength (int, optional): For filtertype='median' filters the input array using a local window-size given
            by windowlength. The array will automatically be zero-padded.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html.
            The windowlenght should be an odd number.
            If filtertype='spline', windowlength is the maximal gap size to fill. Defaults to 5.
        p_bound (float, optional): For filtertype 'arima' this parameter defines the likelihood below,
            below which a body part will be consided as missing data for filtering purposes.
            Defaults to 0.001.
        ARdegree (int, optional): For filtertype 'arima' Autoregressive degree of Sarimax model degree.
            see https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html.
            Defaults to 3.
        MAdegree (int, optional): For filtertype 'arima' Moving Average degree of Sarimax model degree.
            See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html.
            Defaults to 1.
        alpha (float, optional): Significance level for detecting outliers based on the
            confidence interval of the fitted SARIMAX model. Defaults to 0.01.
        save_as_csv (bool, optional): Saves the predictions in a .csv file. Defaults to True.
        destfolder (string, optional): Specifies the destination folder for analysis data. If ``None``, the path of
            the video is used by default. Note that for subsequent analysis this folder
            also needs to be passed. Defaults to None.
        modelprefix (str, optional): Directory containing the deeplabcut models to use when evaluating the network.
            By default, the models are assumed to exist in the project folder. Defaults to "".
        track_method (string, optional): Specifies the tracker used to generate the data.
            Empty by default (corresponding to a single animal project).
            For multiple animals, must be either 'box', 'skeleton', or 'ellipse' and will
            be taken from the config.yaml file if none is given. Defaults to "".
        return_data (bool, optional): If True, returns a dictionary of the filtered data keyed by video names.
            Defaults to False.
        kwargs (dict, optional): Additional arguments.
            For torch-based shuffles, can be used to specify:
                - snapshot_index
                - detector_snapshot_index

    Returns:
        dict | None: If ``return_data`` is True, returns a dictionary mapping video
            filepaths to filtered dataframes. Otherwise returns None.

    Examples:
        Arima model:

            deeplabcut.filterpredictions(
                'C:\\myproject\\reaching-task\\config.yaml',
                ['C:\\myproject\\trailtracking-task\\test.mp4'],
                shuffle=3,
                filtertype='arima',
                ARdegree=5,
                MAdegree=2,
            )

        Use median filter over 10 bins:

            deeplabcut.filterpredictions(
                'C:\\myproject\\reaching-task\\config.yaml',
                ['C:\\myproject\\trailtracking-task\\test.mp4'],
                shuffle=3,
                windowlength=10,
            )

        One can then use the filtered rather than the frame-by-frame predictions by calling:

            deeplabcut.plot_trajectories(
                'C:\\myproject\\reaching-task\\config.yaml',
                ['C:\\myproject\\trailtracking-task\\test.mp4'],
                shuffle=3,
                filtered=True,
            )

            deeplabcut.create_labeled_video(
                'C:\\myproject\\reaching-task\\config.yaml',
                ['C:\\myproject\\trailtracking-task\\test.mp4'],
                shuffle=3,
                filtered=True,
            )
    """
    from deeplabcut.post_processing.filtering import filterpredictions as _filterpredictions

    return _filterpredictions(
        config=config,
        video=video,
        video_extensions=video_extensions,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        filtertype=filtertype,
        windowlength=windowlength,
        p_bound=p_bound,
        ARdegree=ARdegree,
        MAdegree=MAdegree,
        alpha=alpha,
        save_as_csv=save_as_csv,
        destfolder=destfolder,
        modelprefix=modelprefix,
        track_method=track_method,
        return_data=return_data,
        **kwargs,
    )
