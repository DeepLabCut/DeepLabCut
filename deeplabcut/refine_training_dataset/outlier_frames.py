"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from skimage.util import img_as_ubyte

from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils
from deeplabcut.utils import auxiliaryfunctions, visualization, frameselectiontools
from deeplabcut.utils.auxfun_videos import VideoWriter


def find_outliers_in_raw_data(
    config,
    pickle_file,
    video_file,
    pcutoff=0.1,
    percentiles=(5, 95),
    with_annotations=True,
    extraction_algo="kmeans",
):
    """
    Extract outlier frames from either raw detections or assemblies of multiple animals.

    Parameter
    ----------
    config : str
        Absolute path to the project config.yaml.

    pickled_file : str
        Path to a *_full.pickle or *_assemblies.pickle.

    video_file : str
        Path to the corresponding video file for frame extraction.

    pcutoff : float, optional (default=0.1)
        Detection confidence threshold below which frames are flagged as
        containing outliers. Only considered if raw detections are passed in.

    percentiles : tuple, optional (default=(5, 95))
        Assemblies are considered outliers if their areas are beyond the 5th
        and 95th percentiles. Must contain a lower and upper bound.

    with_annotations : bool, optional (default=True)
        If true, extract frames and the corresponding network predictions.
        Otherwise, only the frames are extracted.

    extraction_algo : string, optional (default="kmeans")
        Outlier detection algorithm. Must be either ``uniform`` or ``kmeans``.

    """
    if extraction_algo not in ("kmeans", "uniform"):
        raise ValueError(f"Unsupported extraction algorithm {extraction_algo}.")

    video_name = Path(video_file).stem
    pickle_name = Path(pickle_file).stem
    if not pickle_name.startswith(video_name):
        raise ValueError("Video and pickle files do not match.")

    with open(pickle_file, "rb") as file:
        data = pickle.load(file)
    if pickle_file.endswith("_full.pickle"):
        inds, data = find_outliers_in_raw_detections(data, threshold=pcutoff)
        with_annotations = False
    elif pickle_file.endswith("_assemblies.pickle"):
        assemblies = dict()
        for k, lst in data.items():
            if k == "single":
                continue
            ass = []
            for vals in lst:
                a = inferenceutils.Assembly(len(vals))
                a.data = vals
                ass.append(a)
            assemblies[k] = ass
        inds = inferenceutils.find_outlier_assemblies(assemblies, qs=percentiles)
    else:
        raise IOError(f"Raw data file {pickle_file} could not be parsed.")

    cfg = auxiliaryfunctions.read_config(config)
    ExtractFramesbasedonPreselection(
        inds,
        extraction_algo,
        data,
        video=video_file,
        cfg=cfg,
        config=config,
        savelabeled=False,
        with_annotations=with_annotations,
    )


def find_outliers_in_raw_detections(
    pickled_data, algo="uncertain", threshold=0.1, kept_keypoints=None
):
    """
    Find outlier frames from the raw detections of multiple animals.

    Parameter
    ----------
    pickled_data : dict
        Data in the *_full.pickle file obtained after `analyze_videos`.

    algo : string, optional (default="uncertain")
        Outlier detection algorithm. Currently, only 'uncertain' is supported
        for multi-animal raw detections.

    threshold: float, optional (default=0.1)
        Detection confidence threshold below which frames are flagged as
        containing outliers. Only considered if `algo`==`uncertain`.

    kept_keypoints : list, optional (default=None)
        Indices in the list of labeled body parts to be kept of the analysis.
        By default, all keypoints are used for outlier search.

    Returns
    -------
    candidates : list
        Indices of video frames containing potential outliers
    """
    if algo != "uncertain":
        raise ValueError(f"Only method 'uncertain' is currently supported.")

    try:
        _ = pickled_data.pop("metadata")
    except KeyError:
        pass

    def get_frame_ind(s):
        return int(re.findall(r"\d+", s)[0])

    candidates = []
    data = dict()
    for frame_name, dict_ in pickled_data.items():
        frame_ind = get_frame_ind(frame_name)
        temp_coords = dict_["coordinates"][0]
        temp = dict_["confidence"]
        if kept_keypoints is not None:
            temp = [vals for i, vals in enumerate(temp) if i in kept_keypoints]
        coords = np.concatenate(temp_coords).squeeze()
        conf = np.concatenate(temp).squeeze()
        data[frame_ind] = np.c_[coords, conf]
        if np.any(conf < threshold):
            candidates.append(frame_ind)
    return candidates, data


def extract_outlier_frames(
    config,
    videos,
    videotype=".avi",
    shuffle=1,
    trainingsetindex=0,
    outlieralgorithm="jump",
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
    destfolder=None,
    modelprefix="",
    track_method="",
):
    """
    Extracts the outlier frames in case, the predictions are not correct for a certain video from the cropped video running from
    start to stop as defined in config.yaml.

    Another crucial parameter in config.yaml is how many frames to extract 'numframes2extract'.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle : int, optional
        The shufle index of training dataset. The extracted frames will be stored in the labeled-dataset for
        the corresponding shuffle of training dataset. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    outlieralgorithm: 'fitting', 'jump', 'uncertain', or 'manual'
        String specifying the algorithm used to detect the outliers. Currently, deeplabcut supports three methods + a manual GUI option. 'Fitting'
        fits a Auto Regressive Integrated Moving Average model to the data and computes the distance to the estimated data. Larger distances than
        epsilon are then potentially identified as outliers. The methods 'jump' identifies larger jumps than 'epsilon' in any body part; and 'uncertain'
        looks for frames with confidence below p_bound. The default is set to ``jump``.

    comparisonbodyparts: list of strings, optional
        This selects the body parts for which the comparisons with the outliers are carried out. Either ``all``, then all body parts
        from config.yaml are used orr a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    p_bound: float between 0 and 1, optional
        For outlieralgorithm 'uncertain' this parameter defines the likelihood below, below which a body part will be flagged as a putative outlier.

    epsilon; float,optional
        Meaning depends on outlieralgoritm. The default is set to 20 pixels.
        For outlieralgorithm 'fitting': Float bound according to which frames are picked when the (average) body part estimate deviates from model fit
        For outlieralgorithm 'jump': Float bound specifying the distance by which body points jump from one frame to next (Euclidean distance)

    ARdegree: int, optional
        For outlieralgorithm 'fitting': Autoregressive degree of ARIMA model degree. (Note we use SARIMAX without exogeneous and seasonal part)
        see https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    MAdegree: int
        For outlieralgorithm 'fitting': MovingAvarage degree of ARIMA model degree. (Note we use SARIMAX without exogeneous and seasonal part)
        See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    alpha: float
        Significance level for detecting outliers based on confidence interval of fitted ARIMA model. Only the distance is used however.

    extractionalgorithm : string, optional
        String specifying the algorithm to use for selecting the frames from the identified putatative outlier frames. Currently, deeplabcut
        supports either ``kmeans`` or ``uniform`` based selection (same logic as for extract_frames).
        The default is set to``uniform``, if provided it must be either ``uniform`` or ``kmeans``.

    automatic : bool, optional
        Set it to True, if you want to extract outliers without being asked for user feedback.

    cluster_resizewidth: number, default: 30
        For k-means one can change the width to which the images are downsampled (aspect ratio is fixed).

    cluster_color: bool, default: False
        If false then each downsampled image is treated as a grayscale vector (discarding color information). If true, then the color channels are considered. This increases
        the computational complexity.

    opencv: bool, default: True
        Uses openCV for loading & extractiong (otherwise moviepy (legacy))

    savelabeled: bool, default: False
        If true also saves frame with predicted labels in each folder.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    Examples

    Windows example for extracting the frames with default settings
    >>> deeplabcut.extract_outlier_frames('C:\\myproject\\reaching-task\\config.yaml',['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'])
    --------
    for extracting the frames with default settings
    >>> deeplabcut.extract_outlier_frames('/analysis/project/reaching-task/config.yaml',['/analysis/project/video/reachinvideo1.avi'])
    --------
    for extracting the frames with kmeans
    >>> deeplabcut.extract_outlier_frames('/analysis/project/reaching-task/config.yaml',['/analysis/project/video/reachinvideo1.avi'],extractionalgorithm='kmeans')
    --------
    for extracting the frames with kmeans and epsilon = 5 pixels.
    >>> deeplabcut.extract_outlier_frames('/analysis/project/reaching-task/config.yaml',['/analysis/project/video/reachinvideo1.avi'],epsilon = 5,extractionalgorithm='kmeans')
    --------
    """

    cfg = auxiliaryfunctions.read_config(config)
    bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, comparisonbodyparts
    )
    if not len(bodyparts):
        raise ValueError("No valid bodyparts were selected.")

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    Videos = auxiliaryfunctions.Getlistofvideos(videos, videotype)
    if len(Videos) == 0:
        print("No suitable videos found in", videos)

    for video in Videos:
        if destfolder is None:
            videofolder = str(Path(video).parents[0])
        else:
            videofolder = destfolder
        vname = os.path.splitext(os.path.basename(video))[0]

        try:
            df, dataname, _, _ = auxiliaryfunctions.load_analyzed_data(
                videofolder, vname, DLCscorer, track_method=track_method
            )
            nframes = len(df)
            startindex = max([int(np.floor(nframes * cfg["start"])), 0])
            stopindex = min([int(np.ceil(nframes * cfg["stop"])), nframes])
            Index = np.arange(stopindex - startindex) + startindex

            df = df.iloc[Index]
            mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
            df_temp = df.loc[:, mask]
            Indices = []
            if outlieralgorithm == "uncertain":
                p = df_temp.xs("likelihood", level="coords", axis=1)
                ind = df_temp.index[(p < p_bound).any(axis=1)].tolist()
                Indices.extend(ind)
            elif outlieralgorithm == "jump":
                temp_dt = df_temp.diff(axis=0) ** 2
                temp_dt.drop("likelihood", axis=1, level="coords", inplace=True)
                sum_ = temp_dt.sum(axis=1, level=1)
                ind = df_temp.index[(sum_ > epsilon ** 2).any(axis=1)].tolist()
                Indices.extend(ind)
            elif outlieralgorithm == "fitting":
                d, o = compute_deviations(
                    df_temp, dataname, p_bound, alpha, ARdegree, MAdegree
                )
                # Some heuristics for extracting frames based on distance:
                ind = np.flatnonzero(
                    d > epsilon
                )  # time points with at least average difference of epsilon
                if (
                    len(ind) < cfg["numframes2pick"] * 2
                    and len(d) > cfg["numframes2pick"] * 2
                ):  # if too few points qualify, extract the most distant ones.
                    ind = np.argsort(d)[::-1][: cfg["numframes2pick"] * 2]
                Indices.extend(ind)
            elif outlieralgorithm == "manual":
                wd = Path(config).resolve().parents[0]
                os.chdir(str(wd))
                from deeplabcut.gui import outlier_frame_extraction_toolbox

                outlier_frame_extraction_toolbox.show(
                    config,
                    video,
                    shuffle,
                    df,
                    savelabeled,
                    cfg.get("multianimalproject", False),
                )

            # Run always except when the outlieralgorithm == manual.
            if not outlieralgorithm == "manual":
                Indices = np.sort(list(set(Indices)))  # remove repetitions.
                print(
                    "Method ",
                    outlieralgorithm,
                    " found ",
                    len(Indices),
                    " putative outlier frames.",
                )
                print(
                    "Do you want to proceed with extracting ",
                    cfg["numframes2pick"],
                    " of those?",
                )
                if outlieralgorithm == "uncertain" or outlieralgorithm == "jump":
                    print(
                        "If this list is very large, perhaps consider changing the parameters "
                        "(start, stop, p_bound, comparisonbodyparts) or use a different method."
                    )
                elif outlieralgorithm == "fitting":
                    print(
                        "If this list is very large, perhaps consider changing the parameters "
                        "(start, stop, epsilon, ARdegree, MAdegree, alpha, comparisonbodyparts) "
                        "or use a different method."
                    )

                if not automatic:
                    askuser = input("yes/no")
                else:
                    askuser = "Ja"

                if (
                    askuser == "y"
                    or askuser == "yes"
                    or askuser == "Ja"
                    or askuser == "ha"
                ):  # multilanguage support :)
                    # Now extract from those Indices!
                    ExtractFramesbasedonPreselection(
                        Indices,
                        extractionalgorithm,
                        df,
                        video,
                        cfg,
                        config,
                        opencv,
                        cluster_resizewidth,
                        cluster_color,
                        savelabeled,
                    )
                else:
                    print(
                        "Nothing extracted, please change the parameters and start again..."
                    )
        except FileNotFoundError as e:
            print(e)
            print(
                "It seems the video has not been analyzed yet, or the video is not found! "
                "You can only refine the labels after the a video is analyzed. Please run 'analyze_video' first. "
                "Or, please double check your video file path"
            )


def convertparms2start(pn):
    """ Creating a start value for sarimax in case of an value error
    See: https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk """
    if "ar." in pn:
        return 0
    elif "ma." in pn:
        return 0
    elif "sigma" in pn:
        return 1
    else:
        return 0


def FitSARIMAXModel(x, p, pcutoff, alpha, ARdegree, MAdegree, nforecast=0, disp=False):
    # Seasonal Autoregressive Integrated Moving-Average with eXogenous regressors (SARIMAX)
    # see http://www.statsmodels.org/stable/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
    Y = x.copy()
    Y[p < pcutoff] = np.nan  # Set uncertain estimates to nan (modeled as missing data)
    if np.sum(np.isfinite(Y)) > 10:

        # SARIMAX implemetnation has better prediction models than simple ARIMAX (however we do not use the seasonal etc. parameters!)
        mod = sm.tsa.statespace.SARIMAX(
            Y.flatten(),
            order=(ARdegree, 0, MAdegree),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=True,
        )
        # Autoregressive Moving Average ARMA(p,q) Model
        # mod = sm.tsa.ARIMA(Y, order=(ARdegree,0,MAdegree)) #order=(ARdegree,0,MAdegree)
        try:
            res = mod.fit(disp=disp)
        except ValueError:  # https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk (let's update to statsmodels 0.10.0 soon...)
            startvalues = np.array([convertparms2start(pn) for pn in mod.param_names])
            res = mod.fit(start_params=startvalues, disp=disp)
        except np.linalg.LinAlgError:
            # The process is not stationary, but the default SARIMAX model tries to solve for such a distribution...
            # Relaxing those constraints should do the job.
            mod = sm.tsa.statespace.SARIMAX(
                Y.flatten(),
                order=(ARdegree, 0, MAdegree),
                seasonal_order=(0, 0, 0, 0),
                simple_differencing=True,
                enforce_stationarity=False,
                enforce_invertibility=False,
                use_exact_diffuse=False,
            )
            res = mod.fit(disp=disp)

        predict = res.get_prediction(end=mod.nobs + nforecast - 1)
        return predict.predicted_mean, predict.conf_int(alpha=alpha)
    else:
        return np.nan * np.zeros(len(Y)), np.nan * np.zeros((len(Y), 2))


def compute_deviations(
    Dataframe, dataname, p_bound, alpha, ARdegree, MAdegree, storeoutput=None
):
    """ Fits Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model to data and computes confidence interval
    as well as mean fit. """

    print("Fitting state-space models with parameters:", ARdegree, MAdegree)
    df_x, df_y, df_likelihood = Dataframe.values.reshape((Dataframe.shape[0], -1, 3)).T
    preds = []
    for row in range(len(df_x)):
        x = df_x[row]
        y = df_y[row]
        p = df_likelihood[row]
        meanx, CIx = FitSARIMAXModel(x, p, p_bound, alpha, ARdegree, MAdegree)
        meany, CIy = FitSARIMAXModel(y, p, p_bound, alpha, ARdegree, MAdegree)
        distance = np.sqrt((x - meanx) ** 2 + (y - meany) ** 2)
        significant = (
            (x < CIx[:, 0]) + (x > CIx[:, 1]) + (x < CIy[:, 0]) + (y > CIy[:, 1])
        )
        preds.append(np.c_[distance, significant, meanx, meany, CIx, CIy])

    columns = Dataframe.columns
    prod = []
    for i in range(columns.nlevels - 1):
        prod.append(columns.get_level_values(i).unique())
    prod.append(
        [
            "distance",
            "sig",
            "meanx",
            "meany",
            "lowerCIx",
            "higherCIx",
            "lowerCIy",
            "higherCIy",
        ]
    )
    pdindex = pd.MultiIndex.from_product(prod, names=columns.names)
    data = pd.DataFrame(np.concatenate(preds, axis=1), columns=pdindex)
    # average distance and average # significant differences avg. over comparisonbodyparts
    d = data.xs("distance", axis=1, level=-1).mean(axis=1).values
    o = data.xs("sig", axis=1, level=-1).mean(axis=1).values

    if storeoutput == "full":
        data.to_hdf(
            dataname.split(".h5")[0] + "filtered.h5",
            "df_with_missing",
            format="table",
            mode="w",
        )
        return d, o, data
    else:
        return d, o


def ExtractFramesbasedonPreselection(
    Index,
    extractionalgorithm,
    data,
    video,
    cfg,
    config,
    opencv=True,
    cluster_resizewidth=30,
    cluster_color=False,
    savelabeled=True,
    with_annotations=True,
):
    from deeplabcut.create_project import add

    start = cfg["start"]
    stop = cfg["stop"]
    numframes2extract = cfg["numframes2pick"]
    bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg, "all")

    videofolder = str(Path(video).parents[0])
    vname = str(Path(video).stem)
    tmpfolder = os.path.join(cfg["project_path"], "labeled-data", vname)
    if os.path.isdir(tmpfolder):
        print("Frames from video", vname, " already extracted (more will be added)!")
    else:
        auxiliaryfunctions.attempttomakefolder(tmpfolder, recursive=True)

    nframes = len(data)
    print("Loading video...")
    if opencv:
        vid = VideoWriter(video)
        fps = vid.fps
        duration = vid.calc_duration()
    else:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video)
        fps = clip.fps
        duration = clip.duration

    if cfg["cropping"]:  # one might want to adjust
        coords = (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
    else:
        coords = None

    print("Duration of video [s]: ", duration, ", recorded @ ", fps, "fps!")
    print("Overall # of frames: ", nframes, "with (cropped) frame dimensions: ")
    if extractionalgorithm == "uniform":
        if opencv:
            frames2pick = frameselectiontools.UniformFramescv2(
                vid, numframes2extract, start, stop, Index
            )
        else:
            frames2pick = frameselectiontools.UniformFrames(
                clip, numframes2extract, start, stop, Index
            )
    elif extractionalgorithm == "kmeans":
        if opencv:
            frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
                vid,
                numframes2extract,
                start,
                stop,
                cfg["cropping"],
                coords,
                Index,
                resizewidth=cluster_resizewidth,
                color=cluster_color,
            )
        else:
            if cfg["cropping"]:
                clip = clip.crop(y1=cfg["y1"], y2=cfg["x2"], x1=cfg["x1"], x2=cfg["x2"])
            frames2pick = frameselectiontools.KmeansbasedFrameselection(
                clip,
                numframes2extract,
                start,
                stop,
                Index,
                resizewidth=cluster_resizewidth,
                color=cluster_color,
            )

    else:
        print(
            "Please implement this method yourself! Currently the options are 'kmeans', 'jump', 'uniform'."
        )
        frames2pick = []

    # Extract frames + frames with plotted labels and store them in folder (with name derived from video name) nder labeled-data
    print("Let's select frames indices:", frames2pick)
    colors = visualization.get_cmap(len(bodyparts), cfg["colormap"])
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    for index in frames2pick:  ##tqdm(range(0,nframes,10)):
        if opencv:
            PlottingSingleFramecv2(
                vid,
                cfg["cropping"],
                coords,
                data,
                bodyparts,
                tmpfolder,
                index,
                cfg["dotsize"],
                cfg["pcutoff"],
                cfg["alphavalue"],
                colors,
                strwidth,
                savelabeled,
            )
        else:
            PlottingSingleFrame(
                clip,
                data,
                bodyparts,
                tmpfolder,
                index,
                cfg["dotsize"],
                cfg["pcutoff"],
                cfg["alphavalue"],
                colors,
                strwidth,
                savelabeled,
            )
        plt.close("all")

    # close videos
    if opencv:
        vid.close()
    else:
        clip.close()
        del clip

    # Extract annotations based on DeepLabCut and store in the folder (with name derived from video name) under labeled-data
    if len(frames2pick) > 0:
        try:
            if cfg["cropping"]:
                add.add_new_videos(
                    config, [video], coords=[coords]
                )  # make sure you pass coords as a list
            else:
                add.add_new_videos(config, [video], coords=None)
        except:  # can we make a catch here? - in fact we should drop indices from DataCombined if they are in CollectedData.. [ideal behavior; currently this is pretty unlikely]
            print(
                "AUTOMATIC ADDING OF VIDEO TO CONFIG FILE FAILED! You need to do this manually for including it in the config.yaml file!"
            )
            print("Videopath:", video, "Coordinates for cropping:", coords)
            pass

        if with_annotations:
            machinefile = os.path.join(
                tmpfolder, "machinelabels-iter" + str(cfg["iteration"]) + ".h5"
            )
            if isinstance(data, pd.DataFrame):
                df = data.loc[frames2pick]
                df.index = [
                    os.path.join(
                        "labeled-data",
                        vname,
                        "img" + str(index).zfill(strwidth) + ".png",
                    )
                    for index in df.index
                ]  # exchange index number by file names.
            elif isinstance(data, dict):
                idx = [
                    os.path.join(
                        "labeled-data",
                        vname,
                        "img" + str(index).zfill(strwidth) + ".png",
                    )
                    for index in frames2pick
                ]
                filename = os.path.join(
                    str(tmpfolder), f"CollectedData_{cfg['scorer']}.h5"
                )
                try:
                    df_temp = pd.read_hdf(filename, "df_with_missing")
                    columns = df_temp.columns
                except FileNotFoundError:
                    columns = pd.MultiIndex.from_product(
                        [
                            [cfg["scorer"]],
                            cfg["individuals"],
                            cfg["multianimalbodyparts"],
                            ["x", "y"],
                        ],
                        names=["scorer", "individuals", "bodyparts", "coords"],
                    )
                    if cfg["uniquebodyparts"]:
                        columns2 = pd.MultiIndex.from_product(
                            [
                                [cfg["scorer"]],
                                ["single"],
                                cfg["uniquebodyparts"],
                                ["x", "y"],
                            ],
                            names=["scorer", "individuals", "bodyparts", "coords"],
                        )
                        df_temp = pd.concat(
                            (
                                pd.DataFrame(columns=columns),
                                pd.DataFrame(columns=columns2),
                            )
                        )
                        columns = df_temp.columns
                array = np.full((len(frames2pick), len(columns)), np.nan)
                for i, index in enumerate(frames2pick):
                    data_temp = data.get(index)
                    if data_temp is not None:
                        vals = np.concatenate(data_temp)[:, :2].flatten()
                        array[i, : len(vals)] = vals
                df = pd.DataFrame(array, index=idx, columns=columns)
            else:
                return
            if Path(machinefile).is_file():
                Data = pd.read_hdf(machinefile, "df_with_missing")
                DataCombined = pd.concat([Data, df])
                # drop duplicate labels:
                DataCombined = DataCombined[
                    ~DataCombined.index.duplicated(keep="first")
                ]

                DataCombined.to_hdf(machinefile, key="df_with_missing", mode="w")
                DataCombined.to_csv(
                    os.path.join(tmpfolder, "machinelabels.csv")
                )  # this is always the most current one (as reading is from h5)
            else:
                df.to_hdf(machinefile, key="df_with_missing", mode="w")
                df.to_csv(os.path.join(tmpfolder, "machinelabels.csv"))

        print(
            "The outlier frames are extracted. They are stored in the subdirectory labeled-data\%s."
            % vname
        )
        print(
            "Once you extracted frames for all videos, use 'refine_labels' to manually correct the labels."
        )
    else:
        print("No frames were extracted.")


def PlottingSingleFrame(
    clip,
    Dataframe,
    bodyparts2plot,
    tmpfolder,
    index,
    dotsize,
    pcutoff,
    alphavalue,
    colors,
    strwidth=4,
    savelabeled=True,
):
    """ Label frame and save under imagename / this is already cropped (for clip) """
    from skimage import io

    imagename1 = os.path.join(tmpfolder, "img" + str(index).zfill(strwidth) + ".png")
    imagename2 = os.path.join(
        tmpfolder, "img" + str(index).zfill(strwidth) + "labeled.png"
    )

    if not os.path.isfile(
        os.path.join(tmpfolder, "img" + str(index).zfill(strwidth) + ".png")
    ):
        plt.axis("off")
        image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
        io.imsave(imagename1, image)

        if savelabeled:
            if np.ndim(image) > 2:
                h, w, nc = np.shape(image)
            else:
                h, w = np.shape(image)

            bpts = Dataframe.columns.get_level_values("bodyparts")
            all_bpts = bpts.values[::3]
            df_x, df_y, df_likelihood = Dataframe.values.reshape(
                (Dataframe.shape[0], -1, 3)
            ).T
            bplist = bpts.unique().to_list()
            if Dataframe.columns.nlevels == 3:
                map2bp = list(range(len(all_bpts)))
            else:
                map2bp = [bplist.index(bp) for bp in all_bpts]
            keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))

            plt.figure(frameon=False, figsize=(w * 1.0 / 100, h * 1.0 / 100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.imshow(image)
            for i, ind in enumerate(keep):
                if df_likelihood[ind, index] > pcutoff:
                    plt.scatter(
                        df_x[ind, index],
                        df_y[ind, index],
                        s=dotsize ** 2,
                        color=colors(map2bp[i]),
                        alpha=alphavalue,
                    )
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis("off")
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(imagename2)
            plt.close("all")


def PlottingSingleFramecv2(
    cap,
    crop,
    coords,
    Dataframe,
    bodyparts2plot,
    tmpfolder,
    index,
    dotsize,
    pcutoff,
    alphavalue,
    colors,
    strwidth=4,
    savelabeled=True,
):
    """ Label frame and save under imagename / cap is not already cropped. """
    from skimage import io

    imagename1 = os.path.join(tmpfolder, "img" + str(index).zfill(strwidth) + ".png")
    imagename2 = os.path.join(
        tmpfolder, "img" + str(index).zfill(strwidth) + "labeled.png"
    )

    if not os.path.isfile(
        os.path.join(tmpfolder, "img" + str(index).zfill(strwidth) + ".png")
    ):
        plt.axis("off")
        cap.set_to_frame(index)
        frame = cap.read_frame()
        if frame is None:
            print("Frame could not be read.")
            return
        image = img_as_ubyte(frame)
        if crop:
            image = image[
                int(coords[2]) : int(coords[3]), int(coords[0]) : int(coords[1]), :
            ]

        io.imsave(imagename1, image)

        if savelabeled:
            if np.ndim(image) > 2:
                h, w, nc = np.shape(image)
            else:
                h, w = np.shape(image)

            bpts = Dataframe.columns.get_level_values("bodyparts")
            all_bpts = bpts.values[::3]
            df_x, df_y, df_likelihood = Dataframe.values.reshape(
                (Dataframe.shape[0], -1, 3)
            ).T
            bplist = bpts.unique().to_list()
            if Dataframe.columns.nlevels == 3:
                map2bp = list(range(len(all_bpts)))
            else:
                map2bp = [bplist.index(bp) for bp in all_bpts]
            keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))

            plt.figure(frameon=False, figsize=(w * 1.0 / 100, h * 1.0 / 100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.imshow(image)
            for i, ind in enumerate(keep):
                if df_likelihood[ind, index] > pcutoff:
                    plt.scatter(
                        df_x[ind, index],
                        df_y[ind, index],
                        s=dotsize ** 2,
                        color=colors(map2bp[i]),
                        alpha=alphavalue,
                    )
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis("off")
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(imagename2)
            plt.close("all")


def merge_datasets(config, forceiterate=None):
    """
    Checks if the original training dataset can be merged with the newly refined training dataset. To do so it will check
    if the frames in all extracted video sets were relabeled. If this is the case then the iterate variable is advanced by 1.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    forceiterate: int, optional
        If an integer is given the iteration variable is set to this value (this is only done if all datasets were labeled or refined)

    Example
    --------
    >>> deeplabcut.merge_datasets('/analysis/project/reaching-task/config.yaml')
    --------
    """

    cfg = auxiliaryfunctions.read_config(config)
    config_path = Path(config).parents[0]

    bf = Path(str(config_path / "labeled-data"))
    allfolders = [
        os.path.join(bf, fn)
        for fn in os.listdir(bf)
        if "_labeled" not in fn and not fn.startswith(".")
    ]  # exclude labeled data folders and temporary files
    flagged = False
    for findex, folder in enumerate(allfolders):
        if os.path.isfile(
            os.path.join(folder, "MachineLabelsRefine.h5")
        ):  # Folder that was manually refine...
            pass
        elif os.path.isfile(
            os.path.join(folder, "CollectedData_" + cfg["scorer"] + ".h5")
        ):  # Folder that contains human data set...
            pass
        else:
            print("The following folder was not manually refined,...", folder)
            flagged = True
            pass  # this folder does not contain a MachineLabelsRefine file (not updated...)

    if not flagged:
        # updates iteration by 1
        iter_prev = cfg["iteration"]
        if not forceiterate:
            cfg["iteration"] = int(iter_prev + 1)
        else:
            cfg["iteration"] = forceiterate

        auxiliaryfunctions.write_config(config, cfg)

        print(
            "Merged data sets and updated refinement iteration to "
            + str(cfg["iteration"])
            + "."
        )
        print(
            "Now you can create a new training set for the expanded annotated images (use create_training_dataset)."
        )
    else:
        print("Please label, or remove the un-corrected folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("videos")
    cli_args = parser.parse_args()
