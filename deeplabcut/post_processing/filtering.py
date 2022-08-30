"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal


def columnwise_spline_interp(data, max_gap=0):
    """
    Perform cubic spline interpolation over the columns of *data*.
    All gaps of size lower than or equal to *max_gap* are filled,
    and data slightly smoothed.

    Parameters
    ----------
    data : array_like
        2D matrix of data.
    max_gap : int, optional
        Maximum gap size to fill. By default, all gaps are interpolated.

    Returns
    -------
    interpolated data with same shape as *data*
    """
    if np.ndim(data) < 2:
        data = np.expand_dims(data, axis=1)
    nrows, ncols = data.shape
    temp = data.copy()
    valid = ~np.isnan(temp)
    x = np.arange(nrows)
    for i in range(ncols):
        mask = valid[:, i]
        if (
            np.sum(mask) > 3
        ):  # Make sure there are enough points to fit the cubic spline
            spl = CubicSpline(x[mask], temp[mask, i])
            y = spl(x)
            if max_gap > 0:
                inds = np.flatnonzero(np.r_[True, np.diff(mask), True])
                count = np.diff(inds)
                inds = inds[:-1]
                to_fill = np.ones_like(mask)
                for ind, n, is_nan in zip(inds, count, ~mask[inds]):
                    if is_nan and n > max_gap:
                        to_fill[ind : ind + n] = False
                y[~to_fill] = np.nan
            # Get rid of the interpolation beyond the spline knots
            y[y == 0] = np.nan
            temp[:, i] = y
    return temp


def filterpredictions(
    config,
    video,
    videotype="",
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
):
    """Fits frame-by-frame pose predictions.

    The pose predictions are fitted with ARIMA model (filtertype='arima') or median
    filter (default).

    Parameters
    ----------
    config : string
        Full path of the config.yaml file.

    video : string
        Full path of the video to extract the frame from. Make sure that this video is
        already analyzed.

    shuffle : int, optional, default=1
        The shuffle index of training dataset. The extracted frames will be stored in
        the labeled-dataset for the corresponding shuffle of training dataset.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        Note that TrainingFraction is a list in config.yaml.

    filtertype: string, optional, default="median".
        The filter type - 'arima', 'median' or 'spline'.

    windowlength: int, optional, default=5
        For filtertype='median' filters the input array using a local window-size given
        by windowlength. The array will automatically be zero-padded.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html.
        The windowlenght should be an odd number.
        If filtertype='spline', windowlength is the maximal gap size to fill.

    p_bound: float between 0 and 1, optional, default=0.001
        For filtertype 'arima' this parameter defines the likelihood below,
        below which a body part will be consided as missing data for filtering purposes.

    ARdegree: int, optional, default=3
        For filtertype 'arima' Autoregressive degree of Sarimax model degree.
        see https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    MAdegree: int, optional, default=1
        For filtertype 'arima' Moving Average degree of Sarimax model degree.
        See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    alpha: float, optional, default=0.01
        Significance level for detecting outliers based on confidence interval of fitted SARIMAX model.

    save_as_csv: bool, optional, default=True
        Saves the predictions in a .csv file.

    destfolder: string, optional, default=None
        Specifies the destination folder for analysis data. If ``None``, the path of
        the video is used by default. Note that for subsequent analysis this folder
        also needs to be passed.

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    track_method: string, optional, default=""
        Specifies the tracker used to generate the data.
        Empty by default (corresponding to a single animal project).
        For multiple animals, must be either 'box', 'skeleton', or 'ellipse' and will
        be taken from the config.yaml file if none is given.

    Returns
    -------
    None

    Examples
    --------

    Arima model:

    >>> deeplabcut.filterpredictions(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\myproject\\trailtracking-task\\test.mp4'],
            shuffle=3,
            filterype='arima',
            ARdegree=5,
            MAdegree=2,
        )

    Use median filter over 10 bins:

    >>> deeplabcut.filterpredictions(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\myproject\\trailtracking-task\\test.mp4'],
            shuffle=3,
            windowlength=10,
        )

    One can then use the filtered rather than the frame-by-frame predictions by calling:

    >>> deeplabcut.plot_trajectories(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\myproject\\trailtracking-task\\test.mp4'],
            shuffle=3,
            filtered=True,
        )

    >>> deeplabcut.create_labeled_video(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\myproject\\trailtracking-task\\test.mp4'],
            shuffle=3,
            filtered=True,
        )
    """
    cfg = auxiliaryfunctions.read_config(config)
    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )
    Videos = auxiliaryfunctions.get_list_of_videos(video, videotype)

    if not len(Videos):
        print("No video(s) were found. Please check your paths and/or 'videotype'.")
        return

    for video in Videos:
        if destfolder is None:
            destfolder = str(Path(video).parents[0])

        print("Filtering with %s model %s" % (filtertype, video))
        vname = Path(video).stem

        try:
            _ = auxiliaryfunctions.load_analyzed_data(
                destfolder, vname, DLCscorer, True, track_method
            )
            print(f"Data from {vname} were already filtered. Skipping...")
        except FileNotFoundError:  # Data haven't been filtered yet
            try:
                df, filepath, _, _ = auxiliaryfunctions.load_analyzed_data(
                    destfolder, vname, DLCscorer, track_method=track_method
                )
                nrows = df.shape[0]
                if filtertype == "arima":
                    temp = df.values.reshape((nrows, -1, 3))
                    placeholder = np.empty_like(temp)
                    for i in range(temp.shape[1]):
                        x, y, p = temp[:, i].T
                        meanx, _ = FitSARIMAXModel(
                            x, p, p_bound, alpha, ARdegree, MAdegree, False
                        )
                        meany, _ = FitSARIMAXModel(
                            y, p, p_bound, alpha, ARdegree, MAdegree, False
                        )
                        meanx[0] = x[0]
                        meany[0] = y[0]
                        placeholder[:, i] = np.c_[meanx, meany, p]
                    data = pd.DataFrame(
                        placeholder.reshape((nrows, -1)),
                        columns=df.columns,
                        index=df.index,
                    )
                elif filtertype == "median":
                    data = df.copy()
                    mask = data.columns.get_level_values("coords") != "likelihood"
                    data.loc[:, mask] = df.loc[:, mask].apply(
                        signal.medfilt, args=(windowlength,), axis=0
                    )
                elif filtertype == "spline":
                    data = df.copy()
                    mask_data = data.columns.get_level_values("coords").isin(("x", "y"))
                    xy = data.loc[:, mask_data].values
                    prob = data.loc[:, ~mask_data].values
                    missing = np.isnan(xy)
                    xy_filled = columnwise_spline_interp(xy, windowlength)
                    filled = ~np.isnan(xy_filled)
                    xy[filled] = xy_filled[filled]
                    inds = np.argwhere(missing & filled)
                    if inds.size:
                        # Retrieve original individual label indices
                        inds[:, 1] //= 2
                        inds = np.unique(inds, axis=0)
                        prob[inds[:, 0], inds[:, 1]] = 0.01
                        data.loc[:, ~mask_data] = prob
                    data.loc[:, mask_data] = xy
                else:
                    raise ValueError(f"Unknown filter type {filtertype}")

                outdataname = filepath.replace(".h5", "_filtered.h5")
                data.to_hdf(outdataname, "df_with_missing", format="table", mode="w")
                if save_as_csv:
                    print("Saving filtered csv poses!")
                    data.to_csv(outdataname.split(".h5")[0] + ".csv")
            except FileNotFoundError as e:
                print(e)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("videos")
    cli_args = parser.parse_args()
