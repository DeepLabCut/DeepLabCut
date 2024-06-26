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

Hao Wu, hwu01@g.harvard.edu contributed the original OpenCV class. Thanks!
You can find the directory for your ffmpeg bindings by: "find / | grep ffmpeg" and then setting it.
"""
from __future__ import annotations

import argparse
import os

####################################################
# Dependencies
####################################################
import os.path
from functools import partial
from multiprocessing import get_start_method, Pool
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from skimage.draw import disk, line_aa, set_color
from skimage.util import img_as_ubyte
from tqdm import trange

from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, visualization
from deeplabcut.utils.auxfun_videos import VideoWriter
from deeplabcut.utils.video_processor import (
    VideoProcessorCV as vp,
)  # used to CreateVideo


def get_segment_indices(bodyparts2connect, all_bpts):
    bpts2connect = []
    for bpt1, bpt2 in bodyparts2connect:
        if bpt1 in all_bpts and bpt2 in all_bpts:
            bpts2connect.extend(
                zip(
                    *(
                        np.flatnonzero(all_bpts == bpt1),
                        np.flatnonzero(all_bpts == bpt2),
                    )
                )
            )
    return bpts2connect


def CreateVideo(
    clip,
    Dataframe,
    pcutoff,
    dotsize,
    colormap,
    bodyparts2plot,
    trailpoints,
    cropping,
    x1,
    x2,
    y1,
    y2,
    bodyparts2connect,
    skeleton_color,
    draw_skeleton,
    displaycropped,
    color_by,
    confidence_to_alpha=None,
):
    """Creating individual frames with labeled body parts and making a video"""
    bpts = Dataframe.columns.get_level_values("bodyparts")
    all_bpts = bpts.values[::3]
    if draw_skeleton:
        color_for_skeleton = (
            np.array(mcolors.to_rgba(skeleton_color))[:3] * 255
        ).astype(np.uint8)
        # recode the bodyparts2connect into indices for df_x and df_y for speed
        bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

    if displaycropped:
        ny, nx = y2 - y1, x2 - x1
    else:
        ny, nx = clip.height(), clip.width()

    fps = clip.fps()
    if isinstance(fps, float):
        if fps * 1000 > 65535:
            fps = round(fps)
    nframes = clip.nframes
    duration = nframes / fps

    print(
        "Duration of video [s]: {}, recorded with {} fps!".format(
            round(duration, 2), round(fps, 2)
        )
    )
    print(
        "Overall # of frames: {} with cropped frame dimensions: {} {}".format(
            nframes, nx, ny
        )
    )
    print("Generating frames and creating video.")

    df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T

    if cropping and not displaycropped:
        df_x += x1
        df_y += y1
    colorclass = plt.cm.ScalarMappable(cmap=colormap)

    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    if Dataframe.columns.nlevels == 3:
        nindividuals = int(len(all_bpts) / len(set(all_bpts)))
        map2bp = list(np.repeat(list(range(len(set(all_bpts)))), nindividuals))
        map2id = list(range(nindividuals)) * len(set(all_bpts))
    else:
        nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
        map2bp = [bplist.index(bp) for bp in all_bpts]
        nbpts_per_ind = (
            Dataframe.groupby(level="individuals", axis=1).size().values // 3
        )
        map2id = []
        for i, j in enumerate(nbpts_per_ind):
            map2id.extend([i] * j)
    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]

    if color_by == "bodypart":
        C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
    else:
        C = colorclass.to_rgba(np.linspace(0, 1, nindividuals))
    colors = (C[:, :3] * 255).astype(np.uint8)

    with np.errstate(invalid="ignore"):
        for index in trange(min(nframes, len(Dataframe))):
            image = clip.load_frame()
            if displaycropped:
                image = image[y1:y2, x1:x2]

            # Draw the skeleton for specific bodyparts to be connected as
            # specified in the config file
            if draw_skeleton:
                for bpt1, bpt2 in bpts2connect:
                    if np.all(df_likelihood[[bpt1, bpt2], index] > pcutoff) and not (
                        np.any(np.isnan(df_x[[bpt1, bpt2], index]))
                        or np.any(np.isnan(df_y[[bpt1, bpt2], index]))
                    ):
                        rr, cc, val = line_aa(
                            int(np.clip(df_y[bpt1, index], 0, ny - 1)),
                            int(np.clip(df_x[bpt1, index], 0, nx - 1)),
                            int(np.clip(df_y[bpt2, index], 1, ny - 1)),
                            int(np.clip(df_x[bpt2, index], 1, nx - 1)),
                        )
                        image[rr, cc] = color_for_skeleton

            for ind, num_bp, num_ind in bpts2color:
                if df_likelihood[ind, index] > pcutoff:
                    if color_by == "bodypart":
                        color = colors[num_bp]
                    else:
                        color = colors[num_ind]
                    if trailpoints > 0:
                        for k in range(1, min(trailpoints, index + 1)):
                            rr, cc = disk(
                                (df_y[ind, index - k], df_x[ind, index - k]),
                                dotsize,
                                shape=(ny, nx),
                            )
                            image[rr, cc] = color
                    rr, cc = disk(
                        (df_y[ind, index], df_x[ind, index]), dotsize, shape=(ny, nx)
                    )
                    alpha = 1
                    if confidence_to_alpha is not None:
                        alpha = confidence_to_alpha(df_likelihood[ind, index])

                    set_color(image, (rr, cc), color, alpha)

            clip.save_frame(image)
    clip.close()


def CreateVideoSlow(
    videooutname,
    clip,
    Dataframe,
    tmpfolder,
    dotsize,
    colormap,
    alphavalue,
    pcutoff,
    trailpoints,
    cropping,
    x1,
    x2,
    y1,
    y2,
    save_frames,
    bodyparts2plot,
    outputframerate,
    Frames2plot,
    bodyparts2connect,
    skeleton_color,
    draw_skeleton,
    displaycropped,
    color_by,
):
    """Creating individual frames with labeled body parts and making a video"""
    # scorer=np.unique(Dataframe.columns.get_level_values(0))[0]
    # bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))

    if displaycropped:
        ny, nx = y2 - y1, x2 - x1
    else:
        ny, nx = clip.height(), clip.width()

    fps = clip.fps()
    if outputframerate is None:  # by def. same as input rate.
        outputframerate = fps

    nframes = clip.nframes
    duration = nframes / fps

    print(
        "Duration of video [s]: {}, recorded with {} fps!".format(
            round(duration, 2), round(fps, 2)
        )
    )
    print(
        "Overall # of frames: {} with cropped frame dimensions: {} {}".format(
            nframes, nx, ny
        )
    )
    print("Generating frames and creating video.")
    df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
    if cropping and not displaycropped:
        df_x += x1
        df_y += y1

    bpts = Dataframe.columns.get_level_values("bodyparts")
    all_bpts = bpts.values[::3]
    if draw_skeleton:
        bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    if Dataframe.columns.nlevels == 3:
        nindividuals = int(len(all_bpts) / len(set(all_bpts)))
        map2bp = list(np.repeat(list(range(len(set(all_bpts)))), nindividuals))
        map2id = list(range(nindividuals)) * len(set(all_bpts))
    else:
        nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
        map2bp = [bplist.index(bp) for bp in all_bpts]
        nbpts_per_ind = (
            Dataframe.groupby(level="individuals", axis=1).size().values // 3
        )
        map2id = []
        for i, j in enumerate(nbpts_per_ind):
            map2id.extend([i] * j)
    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]
    if color_by == "individual":
        colors = visualization.get_cmap(nindividuals, name=colormap)
    else:
        colors = visualization.get_cmap(nbodyparts, name=colormap)

    nframes_digits = int(np.ceil(np.log10(nframes)))
    if nframes_digits > 9:
        raise Exception(
            "Your video has more than 10**9 frames, we recommend chopping it up."
        )

    if Frames2plot is None:
        Index = set(range(nframes))
    else:
        Index = {int(k) for k in Frames2plot if 0 <= k < nframes}

    # Prepare figure
    prev_backend = plt.get_backend()
    plt.switch_backend("agg")
    dpi = 100
    fig = plt.figure(frameon=False, figsize=(nx / dpi, ny / dpi))
    ax = fig.add_subplot(111)

    writer = FFMpegWriter(fps=outputframerate, codec="h264")
    with writer.saving(fig, videooutname, dpi=dpi), np.errstate(invalid="ignore"):
        for index in trange(min(nframes, len(Dataframe))):
            imagename = tmpfolder + "/file" + str(index).zfill(nframes_digits) + ".png"
            image = img_as_ubyte(clip.load_frame())
            if index in Index:  # then extract the frame!
                if cropping and displaycropped:
                    image = image[y1:y2, x1:x2]
                ax.imshow(image)

                if draw_skeleton:
                    for bpt1, bpt2 in bpts2connect:
                        if np.all(df_likelihood[[bpt1, bpt2], index] > pcutoff):
                            ax.plot(
                                [df_x[bpt1, index], df_x[bpt2, index]],
                                [df_y[bpt1, index], df_y[bpt2, index]],
                                color=skeleton_color,
                                alpha=alphavalue,
                            )

                for ind, num_bp, num_ind in bpts2color:
                    if df_likelihood[ind, index] > pcutoff:
                        if color_by == "bodypart":
                            color = colors(num_bp)
                        else:
                            color = colors(num_ind)
                        if trailpoints > 0:
                            ax.scatter(
                                df_x[ind][max(0, index - trailpoints) : index],
                                df_y[ind][max(0, index - trailpoints) : index],
                                s=dotsize ** 2,
                                color=color,
                                alpha=alphavalue * 0.75,
                            )
                        ax.scatter(
                            df_x[ind, index],
                            df_y[ind, index],
                            s=dotsize ** 2,
                            color=color,
                            alpha=alphavalue,
                        )
                ax.set_xlim(0, nx)
                ax.set_ylim(0, ny)
                ax.axis("off")
                ax.invert_yaxis()
                fig.subplots_adjust(
                    left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
                )
                if save_frames:
                    fig.savefig(imagename)
                writer.grab_frame()
                ax.clear()

    print("Labeled video {} successfully created.".format(videooutname))
    plt.switch_backend(prev_backend)


def create_labeled_video(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    filtered=False,
    fastmode=True,
    save_frames=False,
    keypoints_only=False,
    Frames2plot=None,
    displayedbodyparts="all",
    displayedindividuals="all",
    codec="mp4v",
    outputframerate=None,
    destfolder=None,
    draw_skeleton=False,
    trailpoints=0,
    displaycropped=False,
    color_by="bodypart",
    modelprefix="",
    init_weights="",
    track_method="",
    superanimal_name="",
    pcutoff=None,
    skeleton=[],
    skeleton_color="white",
    dotsize=8,
    colormap="rainbow",
    alphavalue=0.5,
    overwrite=False,
    confidence_to_alpha: Union[bool, Callable[[float], float]] = False,
):
    """Labels the bodyparts in a video.

    Make sure the video is already analyzed by the function
    ``deeplabcut.analyze_videos``.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file.

    videos : list[str]
        A list of strings containing the full paths to videos for analysis or a path
        to the directory, where all the videos with same extension are stored.

    videotype: str, optional, default=""
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions
        ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle : int, optional, default=1
        Number of shuffles of training dataset.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        Note that TrainingFraction is a list in config.yaml.

    filtered: bool, optional, default=False
        Boolean variable indicating if filtered output should be plotted rather than
        frame-by-frame predictions. Filtered version can be calculated with
        ``deeplabcut.filterpredictions``.

    fastmode: bool, optional, default=True
        If ``True``, uses openCV (much faster but less customization of video) instead
        of matplotlib if ``False``. You can also "save_frames" individually or not in
        the matplotlib mode (if you set the "save_frames" variable accordingly).
        However, using matplotlib to create the frames it therefore allows much more
        flexible (one can set transparency of markers, crop, and easily customize).

    save_frames: bool, optional, default=False
        If ``True``, creates each frame individual and then combines into a video.
        Setting this to ``True`` is relatively slow as it stores all individual frames.

    keypoints_only: bool, optional, default=False
        By default, both video frames and keypoints are visible. If ``True``, only the
        keypoints are shown. These clips are an hommage to Johansson movies,
        see https://www.youtube.com/watch?v=1F5ICP9SYLU and of course his seminal
        paper: "Visual perception of biological motion and a model for its analysis"
        by Gunnar Johansson in Perception & Psychophysics 1973.

    Frames2plot: List[int] or None, optional, default=None
        If not ``None`` and ``save_frames=True`` then the frames corresponding to the
        index will be plotted. For example, ``Frames2plot=[0,11]`` will plot the first
        and the 12th frame.

    displayedbodyparts: list[str] or str, optional, default="all"
        This selects the body parts that are plotted in the video. If ``all``, then all
        body parts from config.yaml are used. If a list of strings that are a subset of
        the full list. E.g. ['hand','Joystick'] for the demo
        Reaching-Mackenzie-2018-08-30/config.yaml to select only these body parts.

    displayedindividuals: list[str] or str, optional, default="all"
        Individuals plotted in the video.
        By default, all individuals present in the config will be showed.

    codec: str, optional, default="mp4v"
        Codec for labeled video. For available options, see
        http://www.fourcc.org/codecs.php. Note that this depends on your ffmpeg
        installation.

    outputframerate: int or None, optional, default=None
        Positive number, output frame rate for labeled video (only available for the
        mode with saving frames.) If ``None``, which results in the original video
        rate.

    destfolder: string or None, optional, default=None
        Specifies the destination folder that was used for storing analysis data. If
        ``None``, the path of the video file is used.

    draw_skeleton: bool, optional, default=False
        If ``True`` adds a line connecting the body parts making a skeleton on each
        frame. The body parts to be connected and the color of these connecting lines
        are specified in the config file.

    trailpoints: int, optional, default=0
        Number of previous frames whose body parts are plotted in a frame
        (for displaying history).

    displaycropped: bool, optional, default=False
        Specifies whether only cropped frame is displayed (with labels analyzed
        therein), or the original frame with the labels analyzed in the cropped subset.

    color_by : string, optional, default='bodypart'
        Coloring rule. By default, each bodypart is colored differently.
        If set to 'individual', points belonging to a single individual are colored the
        same.

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    init_weights: str,
        Checkpoint path to the super model

    track_method: string, optional, default=""
        Specifies the tracker used to generate the data.
        Empty by default (corresponding to a single animal project).
        For multiple animals, must be either 'box', 'skeleton', or 'ellipse' and will
        be taken from the config.yaml file if none is given.

    pcutoff: string, optional, default=None
        Overrides the pcutoff set in the project configuration to plot the trajectories.

    overwrite: bool, optional, default=False
        If ``True`` overwrites existing labeled videos.

    confidence_to_alpha: Union[bool, Callable[[float], float], default=False
        If False, all keypoints will be plot with alpha=1. Otherwise, this can be
        defined as a function f: [0, 1] -> [0, 1] such that the alpha value for a
        keypoint will be set as a function of its score: alpha = f(score). The default
        function used when True is f(x) = max(0, (x - pcutoff)/(1 - pcutoff)).

    Returns
    -------
        results : list[bool]
        ``True`` if the video is successfully created for each item in ``videos``.

    Examples
    --------

    Create the labeled video for a single video

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/reachingvideo1.avi'],
        )

    Create the labeled video for a single video and store the individual frames

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/reachingvideo1.avi'],
            fastmode=True,
            save_frames=True,
        )

    Create the labeled video for multiple videos

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
        )

    Create the labeled video for all the videos with an .avi extension in a directory.

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/'],
        )

    Create the labeled video for all the videos with an .mp4 extension in a directory.

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/'],
            videotype='mp4',
        )
    """
    if config == "":
        if pcutoff is None:
            pcutoff = 0.6
    else:
        cfg = auxiliaryfunctions.read_config(config)
        trainFraction = cfg["TrainingFraction"][trainingsetindex]
        track_method = auxfun_multianimal.get_track_method(
            cfg, track_method=track_method
        )
        if pcutoff is None:
            pcutoff = cfg["pcutoff"]

    if init_weights == "":
        DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
            cfg, shuffle, trainFraction, modelprefix=modelprefix
        )  # automatically loads corresponding model (even training iteration based on snapshot index)
    else:
        DLCscorer = "DLC_" + Path(init_weights).stem
        DLCscorerlegacy = "DLC_" + Path(init_weights).stem

    if save_frames:
        fastmode = False  # otherwise one cannot save frames
        keypoints_only = False

    # parse the alpha selection function
    if isinstance(confidence_to_alpha, bool):
        confidence_to_alpha = _get_default_conf_to_alpha(confidence_to_alpha, pcutoff)

    if superanimal_name != "":
        dlc_root_path = auxiliaryfunctions.get_deeplabcut_path()
        dataset_name = "_".join(superanimal_name.split("_")[:-1])
        test_cfg = auxiliaryfunctions.read_plainconfig(
            os.path.join(
                dlc_root_path,
                "modelzoo",
                "project_configs",
                f"{dataset_name}.yaml",
            )
        )

        bodyparts = test_cfg["bodyparts"]
        cfg = {
            "skeleton": skeleton,
            "skeleton_color": skeleton_color,
            "pcutoff": pcutoff,
            "dotsize": dotsize,
            "alphavalue": alphavalue,
            "colormap": colormap,
        }
    else:
        bodyparts = (
            auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
                cfg, displayedbodyparts
            )
        )

    individuals = auxfun_multianimal.IntersectionofIndividualsandOnesGivenbyUser(
        cfg, displayedindividuals
    )
    if draw_skeleton:
        bodyparts2connect = cfg["skeleton"]
        if displayedbodyparts != "all":
            bodyparts2connect = [
                pair
                for pair in bodyparts2connect
                if all(element in displayedbodyparts for element in pair)
            ]
        skeleton_color = cfg["skeleton_color"]
    else:
        bodyparts2connect = None
        skeleton_color = None

    start_path = os.getcwd()
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)

    if not Videos:
        return []

    func = partial(
        proc_video,
        videos,
        destfolder,
        filtered,
        DLCscorer,
        DLCscorerlegacy,
        track_method,
        cfg,
        individuals,
        color_by,
        bodyparts,
        codec,
        bodyparts2connect,
        trailpoints,
        save_frames,
        outputframerate,
        Frames2plot,
        draw_skeleton,
        skeleton_color,
        displaycropped,
        fastmode,
        keypoints_only,
        overwrite,
        init_weights=init_weights,
        pcutoff=pcutoff,
        confidence_to_alpha=confidence_to_alpha,
    )

    if get_start_method() == "fork":
        with Pool(min(os.cpu_count(), len(Videos))) as pool:
            results = pool.map(func, Videos)
    else:
        results = []
        for video in Videos:
            results.append(func(video))

    os.chdir(start_path)
    return results


def proc_video(
    videos,
    destfolder,
    filtered,
    DLCscorer,
    DLCscorerlegacy,
    track_method,
    cfg,
    individuals,
    color_by,
    bodyparts,
    codec,
    bodyparts2connect,
    trailpoints,
    save_frames,
    outputframerate,
    Frames2plot,
    draw_skeleton,
    skeleton_color,
    displaycropped,
    fastmode,
    keypoints_only,
    overwrite,
    video,
    init_weights="",
    pcutoff: float | None = None,
    confidence_to_alpha: Optional[Callable[[float], float]] = None,
):
    """Helper function for create_videos

    Parameters
    ----------


    Returns
    -------
        result : bool
        ``True`` if a video is successfully created.
    """
    videofolder = Path(video).parents[0]
    if destfolder is None:
        destfolder = videofolder  # where your folder with videos is.

    if pcutoff is None:
        pcutoff = cfg["pcutoff"]

    auxiliaryfunctions.attempt_to_make_folder(destfolder)

    os.chdir(destfolder)  # THE VIDEO IS STILL IN THE VIDEO FOLDER
    print("Starting to process video: {}".format(video))
    vname = str(Path(video).stem)

    if init_weights != "":
        DLCscorer = "DLC_" + Path(init_weights).stem
        DLCscorerlegacy = "DLC_" + Path(init_weights).stem

    if filtered:
        videooutname1 = os.path.join(vname + DLCscorer + "filtered_labeled.mp4")
        videooutname2 = os.path.join(vname + DLCscorerlegacy + "filtered_labeled.mp4")
    else:
        videooutname1 = os.path.join(vname + DLCscorer + "_labeled.mp4")
        videooutname2 = os.path.join(vname + DLCscorerlegacy + "_labeled.mp4")

    if (
        os.path.isfile(videooutname1) or os.path.isfile(videooutname2)
    ) and not overwrite:
        print("Labeled video {} already created.".format(vname))
        return True
    else:
        print("Loading {} and data.".format(video))
        try:
            df, filepath, _, _ = auxiliaryfunctions.load_analyzed_data(
                destfolder, vname, DLCscorer, filtered, track_method
            )
            metadata = auxiliaryfunctions.load_video_metadata(
                destfolder, vname, DLCscorer
            )
            if cfg.get("multianimalproject", False):
                s = "_id" if color_by == "individual" else "_bp"
            else:
                s = ""

            videooutname = filepath.replace(
                ".h5", f"{s}_p{int(100 * pcutoff)}_labeled.mp4"
            )
            if os.path.isfile(videooutname) and not overwrite:
                print("Labeled video already created. Skipping...")
                return

            if all(individuals):
                df = df.loc(axis=1)[:, individuals]
            cropping = metadata["data"]["cropping"]
            [x1, x2, y1, y2] = metadata["data"]["cropping_parameters"]
            labeled_bpts = [
                bp
                for bp in df.columns.get_level_values("bodyparts").unique()
                if bp in bodyparts
            ]

            if keypoints_only:
                # Mask rather than drop unwanted bodyparts to ensure consistent coloring
                mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
                df.loc[:, ~mask] = np.nan
                inds = None
                if bodyparts2connect:
                    all_bpts = df.columns.get_level_values("bodyparts")[::3]
                    inds = get_segment_indices(bodyparts2connect, all_bpts)
                clip = vp(fname=video, fps=outputframerate)
                create_video_with_keypoints_only(
                    df,
                    videooutname,
                    inds,
                    pcutoff,
                    cfg["dotsize"],
                    cfg["alphavalue"],
                    skeleton_color=skeleton_color,
                    color_by=color_by,
                    colormap=cfg["colormap"],
                    fps=clip.fps(),
                )
                clip.close()
            elif not fastmode:
                tmpfolder = os.path.join(str(videofolder), "temp-" + vname)
                if save_frames:
                    auxiliaryfunctions.attempt_to_make_folder(tmpfolder)
                clip = vp(video)
                CreateVideoSlow(
                    videooutname,
                    clip,
                    df,
                    tmpfolder,
                    cfg["dotsize"],
                    cfg["colormap"],
                    cfg["alphavalue"],
                    pcutoff,
                    trailpoints,
                    cropping,
                    x1,
                    x2,
                    y1,
                    y2,
                    save_frames,
                    labeled_bpts,
                    outputframerate,
                    Frames2plot,
                    bodyparts2connect,
                    skeleton_color,
                    draw_skeleton,
                    displaycropped,
                    color_by,
                )
                clip.close()
            else:
                _create_labeled_video(
                    video,
                    filepath,
                    keypoints2show=labeled_bpts,
                    animals2show=individuals,
                    bbox=(x1, x2, y1, y2),
                    codec=codec,
                    output_path=videooutname,
                    pcutoff=pcutoff,
                    dotsize=cfg["dotsize"],
                    cmap=cfg["colormap"],
                    color_by=color_by,
                    skeleton_edges=bodyparts2connect,
                    skeleton_color=skeleton_color,
                    trailpoints=trailpoints,
                    fps=outputframerate,
                    display_cropped=displaycropped,
                    confidence_to_alpha=confidence_to_alpha,
                )
            return True

        except FileNotFoundError as e:
            print(e)
            return False


def _create_labeled_video(
    video,
    h5file,
    keypoints2show="all",
    animals2show="all",
    skeleton_edges=None,
    pcutoff=0.6,
    dotsize=6,
    cmap="rainbow",
    color_by="bodypart",
    skeleton_color="k",
    trailpoints=0,
    bbox=None,
    display_cropped=False,
    codec="mp4v",
    fps=None,
    output_path="",
    confidence_to_alpha=None,
):
    if color_by not in ("bodypart", "individual"):
        raise ValueError("`color_by` should be either 'bodypart' or 'individual'.")

    if not output_path:
        s = "_id" if color_by == "individual" else "_bp"
        output_path = h5file.replace(".h5", f"{s}_labeled.mp4")

    x1, x2, y1, y2 = bbox
    if display_cropped:
        sw = x2 - x1
        sh = y2 - y1
    else:
        sw = sh = ""

    clip = vp(
        fname=video,
        sname=output_path,
        codec=codec,
        sw=sw,
        sh=sh,
        fps=fps,
    )
    cropping = bbox != (0, clip.w, 0, clip.h)
    df = pd.read_hdf(h5file)
    try:
        animals = df.columns.get_level_values("individuals").unique().to_list()
        if animals2show != "all" and isinstance(animals, Iterable):
            animals = [a for a in animals if a in animals2show]
        df = df.loc(axis=1)[:, animals]
    except KeyError:
        pass
    kpts = df.columns.get_level_values("bodyparts").unique().to_list()
    if keypoints2show != "all" and isinstance(keypoints2show, Iterable):
        kpts = [kpt for kpt in kpts if kpt in keypoints2show]
    CreateVideo(
        clip,
        df,
        pcutoff,
        dotsize,
        cmap,
        kpts,
        trailpoints,
        cropping,
        x1,
        x2,
        y1,
        y2,
        skeleton_edges,
        skeleton_color,
        bool(skeleton_edges),
        display_cropped,
        color_by,
        confidence_to_alpha=confidence_to_alpha,
    )


def create_video_with_keypoints_only(
    df,
    output_name,
    ind_links=None,
    pcutoff=0.6,
    dotsize=8,
    alpha=0.7,
    background_color="k",
    skeleton_color="navy",
    color_by="bodypart",
    colormap="viridis",
    fps=25,
    dpi=200,
    codec="h264",
):
    bodyparts = df.columns.get_level_values("bodyparts")[::3]
    bodypart_names = bodyparts.unique()
    n_bodyparts = len(bodypart_names)
    nx = int(np.nanmax(df.xs("x", axis=1, level="coords")))
    ny = int(np.nanmax(df.xs("y", axis=1, level="coords")))

    n_frames = df.shape[0]
    xyp = df.values.reshape((n_frames, -1, 3))

    if color_by == "bodypart":
        map_ = bodyparts.map(dict(zip(bodypart_names, range(n_bodyparts))))
        cmap = plt.get_cmap(colormap, n_bodyparts)
    elif color_by == "individual":
        try:
            individuals = df.columns.get_level_values("individuals")[::3]
            individual_names = individuals.unique().to_list()
            n_individuals = len(individual_names)
            map_ = individuals.map(dict(zip(individual_names, range(n_individuals))))
            cmap = plt.get_cmap(colormap, n_individuals)
        except KeyError as e:
            raise Exception(
                "Coloring by individuals is only valid for multi-animal data"
            ) from e
    else:
        raise ValueError(f"Invalid color_by={color_by}")

    prev_backend = plt.get_backend()
    plt.switch_backend("agg")
    fig = plt.figure(frameon=False, figsize=(nx / dpi, ny / dpi))
    ax = fig.add_subplot(111)
    scat = ax.scatter([], [], s=dotsize ** 2, alpha=alpha)
    coords = xyp[0, :, :2]
    coords[xyp[0, :, 2] < pcutoff] = np.nan
    scat.set_offsets(coords)
    colors = cmap(map_)
    scat.set_color(colors)
    segs = coords[tuple(zip(*tuple(ind_links))), :].swapaxes(0, 1) if ind_links else []
    coll = LineCollection(segs, colors=skeleton_color, alpha=alpha)
    ax.add_collection(coll)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=background_color, transform=ax.transAxes, zorder=-1
        )
    )
    ax.invert_yaxis()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    writer = FFMpegWriter(fps=fps, codec=codec)
    with writer.saving(fig, output_name, dpi=dpi):
        writer.grab_frame()
        for index, _ in enumerate(trange(n_frames - 1), start=1):
            coords = xyp[index, :, :2]
            coords[xyp[index, :, 2] < pcutoff] = np.nan
            scat.set_offsets(coords)
            if ind_links:
                segs = coords[tuple(zip(*tuple(ind_links))), :].swapaxes(0, 1)
            coll.set_segments(segs)
            writer.grab_frame()
    plt.close(fig)
    plt.switch_backend(prev_backend)


def create_video_with_all_detections(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    displayedbodyparts="all",
    cropping: Optional[List[int]] = None,
    destfolder=None,
    modelprefix="",
    confidence_to_alpha: Union[bool, Callable[[float], float]] = False,
):
    """
    Create a video labeled with all the detections stored in a '*_full.pickle' file.

    Parameters
    ----------
    config : str
        Absolute path to the config.yaml file

    videos : list of str
        A list of strings containing the full paths to videos for analysis or a path to the directory,
        where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle : int, optional
        Number of shuffles of training dataset. Default is set to 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    displayedbodyparts: list of strings, optional
        This selects the body parts that are plotted in the video. Either ``all``, then all body parts
        from config.yaml are used orr a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    cropping: list[int], optional (default=None)
        If passed in, the [x1, x2, y1, y2] crop coordinates are used to shift detections appropriately.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    confidence_to_alpha: Union[bool, Callable[[float], float], default=False
        If False, all keypoints will be plot with alpha=1. Otherwise, this can be
        defined as a function f: [0, 1] -> [0, 1] such that the alpha value for a
        keypoint will be set as a function of its score: alpha = f(score). The default
        function used when True is f(x) = x.
    """
    import re

    from deeplabcut.core.inferenceutils import Assembler

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    DLCscorername, _ = auxiliaryfunctions.get_scorer_name(
        cfg, shuffle, trainFraction, modelprefix=modelprefix
    )

    videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
    if not videos:
        return

    if isinstance(confidence_to_alpha, bool):
        confidence_to_alpha = _get_default_conf_to_alpha(confidence_to_alpha, 0)

    for video in videos:
        videofolder = os.path.splitext(video)[0]

        if destfolder is None:
            outputname = "{}_full.mp4".format(videofolder + DLCscorername)
            full_pickle = os.path.join(videofolder + DLCscorername + "_full.pickle")
        else:
            auxiliaryfunctions.attempt_to_make_folder(destfolder)
            outputname = os.path.join(
                destfolder, str(Path(video).stem) + DLCscorername + "_full.mp4"
            )
            full_pickle = os.path.join(
                destfolder, str(Path(video).stem) + DLCscorername + "_full.pickle"
            )

        if not (os.path.isfile(outputname)):
            video_name = str(Path(video).stem)
            print("Creating labeled video for ", video_name)
            h5file = full_pickle.replace("_full.pickle", ".h5")
            data, metadata = auxfun_multianimal.LoadFullMultiAnimalData(h5file)
            data = dict(
                data
            )  # Cast to dict (making a copy) so items can safely be popped

            x1, y1 = 0, 0
            if cropping is not None:
                x1, _, y1, _ = cropping
            elif metadata.get("data", {}).get("cropping"):
                x1, _, y1, _ = metadata["data"]["cropping_parameters"]

            header = data.pop("metadata")
            all_jointnames = header["all_joints_names"]

            if displayedbodyparts == "all":
                numjoints = len(all_jointnames)
                bpts = range(numjoints)
            else:  # select only "displayedbodyparts"
                bpts = []
                for bptindex, bp in enumerate(all_jointnames):
                    if bp in displayedbodyparts:
                        bpts.append(bptindex)
                numjoints = len(bpts)
            frame_names = list(data)
            frames = [int(re.findall(r"\d+", name)[0]) for name in frame_names]
            colorclass = plt.cm.ScalarMappable(cmap=cfg["colormap"])
            C = colorclass.to_rgba(np.linspace(0, 1, numjoints))
            colors = (C[:, :3] * 255).astype(np.uint8)

            pcutoff = cfg["pcutoff"]
            dotsize = cfg["dotsize"]
            clip = vp(fname=video, sname=outputname, codec="mp4v")
            ny, nx = clip.height(), clip.width()

            for n in trange(clip.nframes):
                frame = clip.load_frame()
                if frame is None:
                    continue
                try:
                    ind = frames.index(n)
                    dets = Assembler._flatten_detections(data[frame_names[ind]])
                    for det in dets:
                        if det.label not in bpts or det.confidence < pcutoff:
                            continue
                        x, y = det.pos
                        x += x1
                        y += y1
                        rr, cc = disk((y, x), dotsize, shape=(ny, nx))
                        alpha = 1
                        if confidence_to_alpha is not None:
                            alpha = confidence_to_alpha(det.confidence)

                        set_color(
                            frame,
                            (rr, cc),
                            colors[bpts.index(det.label)],
                            alpha,
                        )
                except ValueError as err:  # No data stored for that particular frame
                    print(n, f"no data: {err}")
                    pass
                try:
                    clip.save_frame(frame)
                except:
                    print(n, "frame writing error.")
                    pass
            clip.close()
        else:
            print("Detections already plotted, ", outputname)


def _create_video_from_tracks(video, tracks, destfolder, output_name, pcutoff, scale=1):
    import subprocess

    from tqdm import tqdm

    if not os.path.isdir(destfolder):
        os.mkdir(destfolder)

    vid = VideoWriter(video)
    nframes = len(vid)
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    nx, ny = vid.dimensions
    # cropping!
    X2 = nx  # 1600
    X1 = 0
    # nx=X2-X1
    numtracks = len(tracks.keys()) - 1
    trackids = [t for t in tracks.keys() if t != "header"]
    cc = np.random.rand(numtracks + 1, 3)
    fig, ax = visualization.prepare_figure_axes(nx, ny, scale)
    im = ax.imshow(np.zeros((ny, nx)))
    markers = sum([ax.plot([], [], ".", c=c) for c in cc], [])
    for index in tqdm(range(nframes)):
        vid.set_to_frame(index)
        imname = "frame" + str(index).zfill(strwidth)
        image_output = os.path.join(destfolder, imname + ".png")
        frame = vid.read_frame()
        if frame is not None and not os.path.isfile(image_output):
            im.set_data(frame[:, X1:X2])
            for n, trackid in enumerate(trackids):
                if imname in tracks[trackid]:
                    x, y, p = tracks[trackid][imname].reshape((-1, 3)).T
                    markers[n].set_data(x[p > pcutoff], y[p > pcutoff])
                else:
                    markers[n].set_data([], [])
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(image_output)

    outputframerate = 30
    os.chdir(destfolder)

    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            str(vid.fps),
            "-i",
            f"frame%0{strwidth}d.png",
            "-r",
            str(outputframerate),
            output_name,
        ]
    )


def create_video_from_pickled_tracks(
    video, pickle_file, destfolder="", output_name="", pcutoff=0.6
):
    if not destfolder:
        destfolder = os.path.splitext(video)[0]
    if not output_name:
        video_name, ext = os.path.splitext(os.path.split(video)[1])
        output_name = video_name + "DLClabeled" + ext
    tracks = auxiliaryfunctions.read_pickle(pickle_file)
    _create_video_from_tracks(video, tracks, destfolder, output_name, pcutoff)


def _get_default_conf_to_alpha(
    confidence_to_alpha: bool,
    pcutoff: float,
) -> Optional[Callable[[float], float]]:
    """Creates the default confidence_to_alpha function"""
    if not confidence_to_alpha:
        return None

    def default_confidence_to_alpha(x):
        if pcutoff == 0:
            return x
        return np.clip((x - pcutoff) / (1 - pcutoff), 0, 1)

    return default_confidence_to_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("videos")
    cli_args = parser.parse_args()
