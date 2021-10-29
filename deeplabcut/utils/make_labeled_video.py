"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Hao Wu, hwu01@g.harvard.edu contributed the original OpenCV class. Thanks!
You can find the directory for your ffmpeg bindings by: "find / | grep ffmpeg" and then setting it.
"""

import argparse
import os

####################################################
# Dependencies
####################################################
import os.path
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from skimage.draw import disk, line_aa
from skimage.util import img_as_ubyte
from tqdm import trange

from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, visualization
from deeplabcut.utils.video_processor import (
    VideoProcessorCV as vp,
)  # used to CreateVideo
from deeplabcut.utils.auxfun_videos import VideoWriter


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
        nindividuals = 1
        map2bp = list(range(len(all_bpts)))
        map2id = [0 for _ in map2bp]
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
                    image[rr, cc] = color

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
        nindividuals = 1
        map2bp = list(range(len(all_bpts)))
        map2id = [0 for _ in map2bp]
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
    videotype="avi",
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
    track_method="",
):
    """
    Labels the bodyparts in a video. Make sure the video is already analyzed by the function 'analyze_video'

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle : int, optional
        Number of shuffles of training dataset. Default is set to 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    filtered: bool, default false
        Boolean variable indicating if filtered output should be plotted rather than frame-by-frame predictions. Filtered version can be calculated with deeplabcut.filterpredictions

    fastmode: bool
        If true uses openCV (much faster but less customization of video) vs matplotlib (if false). You can also
        "save_frames" individually or not in the matplotlib mode (if you set the "save_frames" variable accordingly).
        However, using matplotlib to create the frames it therefore allows much more flexible (one can set transparency of markers, crop, and easily customize).

    save_frames: bool
        If true creates each frame individual and then combines into a video. This variant is relatively slow as
        it stores all individual frames.

    keypoints_only: bool, optional
        By default, both video frames and keypoints are visible. If true, only the keypoints are shown. These clips are an hommage to Johansson movies,
        see https://www.youtube.com/watch?v=1F5ICP9SYLU and of course his seminal paper: "Visual perception of biological motion and a model for its analysis"
        by Gunnar Johansson in Perception & Psychophysics 1973.

    Frames2plot: List of indices
        If not None & save_frames=True then the frames corresponding to the index will be plotted. For example, Frames2plot=[0,11] will plot the first and the 12th frame.

    displayedbodyparts: list of strings, optional
        This selects the body parts that are plotted in the video. Either ``all``, then all body parts
        from config.yaml are used orr a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    displayedindividuals: list of strings, optional
        Individuals plotted in the video. By default, all individuals present in the config will be showed.

    codec: codec for labeled video. Options see http://www.fourcc.org/codecs.php [depends on your ffmpeg installation.]

    outputframerate: positive number, output frame rate for labeled video (only available for the mode with saving frames.) By default: None, which results in the original video rate.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    draw_skeleton: bool
        If ``True`` adds a line connecting the body parts making a skeleton on on each frame. The body parts to be connected and the color of these connecting lines are specified in the config file. By default: ``False``

    trailpoints: int
        Number of revious frames whose body parts are plotted in a frame (for displaying history). Default is set to 0.

    displaycropped: bool, optional
        Specifies whether only cropped frame is displayed (with labels analyzed therein), or the original frame with the labels analyzed in the cropped subset.

    color_by : string, optional (default='bodypart')
        Coloring rule. By default, each bodypart is colored differently.
        If set to 'individual', points belonging to a single individual are colored the same.

    Examples
    --------
    If you want to create the labeled video for only 1 video
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'])
    --------

    If you want to create the labeled video for only 1 video and store the individual frames
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'],fastmode=True, save_frames=True)
    --------

    If you want to create the labeled video for multiple videos
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'])
    --------

    If you want to create the labeled video for all the videos (as .avi extension) in a directory.
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/'])

    --------
    If you want to create the labeled video for all the videos (as .mp4 extension) in a directory.
    >>> deeplabcut.create_labeled_video('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/'],videotype='mp4')

    --------

    """
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg, shuffle, trainFraction, modelprefix=modelprefix
    )  # automatically loads corresponding model (even training iteration based on snapshot index)

    if save_frames:
        fastmode = False  # otherwise one cannot save frames
        keypoints_only = False

    bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, displayedbodyparts
    )
    individuals = auxfun_multianimal.IntersectionofIndividualsandOnesGivenbyUser(
        cfg, displayedindividuals
    )
    if draw_skeleton:
        bodyparts2connect = cfg["skeleton"]
        skeleton_color = cfg["skeleton_color"]
    else:
        bodyparts2connect = None
        skeleton_color = None

    start_path = os.getcwd()
    Videos = auxiliaryfunctions.Getlistofvideos(videos, videotype)

    if not Videos:
        print("No video(s) were found. Please check your paths and/or 'video_type'.")
        return

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
    )

    with Pool(min(os.cpu_count(), len(Videos))) as pool:
        pool.map(func, Videos)

    os.chdir(start_path)


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
    video,
):
    """Helper function for create_videos

    Parameters
    ----------


    """
    videofolder = Path(video).parents[0]
    if destfolder is None:
        destfolder = videofolder  # where your folder with videos is.

    auxiliaryfunctions.attempttomakefolder(destfolder)

    os.chdir(destfolder)  # THE VIDEO IS STILL IN THE VIDEO FOLDER
    print("Starting to process video: {}".format(video))
    vname = str(Path(video).stem)

    if filtered:
        videooutname1 = os.path.join(vname + DLCscorer + "filtered_labeled.mp4")
        videooutname2 = os.path.join(vname + DLCscorerlegacy + "filtered_labeled.mp4")
    else:
        videooutname1 = os.path.join(vname + DLCscorer + "_labeled.mp4")
        videooutname2 = os.path.join(vname + DLCscorerlegacy + "_labeled.mp4")

    if os.path.isfile(videooutname1) or os.path.isfile(videooutname2):
        print("Labeled video {} already created.".format(vname))
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
            videooutname = filepath.replace(".h5", f"{s}_labeled.mp4")
            if os.path.isfile(videooutname):
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
                    cfg["pcutoff"],
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
                    auxiliaryfunctions.attempttomakefolder(tmpfolder)
                clip = vp(video)
                CreateVideoSlow(
                    videooutname,
                    clip,
                    df,
                    tmpfolder,
                    cfg["dotsize"],
                    cfg["colormap"],
                    cfg["alphavalue"],
                    cfg["pcutoff"],
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
                if displaycropped:  # then the cropped video + the labels is depicted
                    clip = vp(
                        fname=video,
                        sname=videooutname,
                        codec=codec,
                        sw=x2 - x1,
                        sh=y2 - y1,
                        fps=outputframerate,
                    )
                else:  # then the full video + the (perhaps in cropped mode analyzed labels) are depicted
                    clip = vp(fname=video, sname=videooutname, codec=codec, fps=outputframerate)
                CreateVideo(
                    clip,
                    df,
                    cfg["pcutoff"],
                    cfg["dotsize"],
                    cfg["colormap"],
                    labeled_bpts,
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
                )

        except FileNotFoundError as e:
            print(e)


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
    videotype="avi",
    shuffle=1,
    trainingsetindex=0,
    displayedbodyparts="all",
    destfolder=None,
    modelprefix="",
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
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle : int, optional
        Number of shuffles of training dataset. Default is set to 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    displayedbodyparts: list of strings, optional
        This selects the body parts that are plotted in the video. Either ``all``, then all body parts
        from config.yaml are used orr a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    """
    from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import Assembler
    import pickle, re

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    DLCscorername, _ = auxiliaryfunctions.GetScorerName(
        cfg, shuffle, trainFraction, modelprefix=modelprefix
    )

    videos = auxiliaryfunctions.Getlistofvideos(videos, videotype)
    if not videos:
        print("No video(s) were found. Please check your paths and/or 'video_type'.")
        return

    for video in videos:
        videofolder = os.path.splitext(video)[0]

        if destfolder is None:
            outputname = "{}_full.mp4".format(videofolder + DLCscorername)
            full_pickle = os.path.join(videofolder + DLCscorername + "_full.pickle")
        else:
            auxiliaryfunctions.attempttomakefolder(destfolder)
            outputname = os.path.join(
                destfolder, str(Path(video).stem) + DLCscorername + "_full.mp4"
            )
            full_pickle = os.path.join(
                destfolder, str(Path(video).stem) + DLCscorername + "_full.pickle"
            )

        if not (os.path.isfile(outputname)):
            print("Creating labeled video for ", str(Path(video).stem))
            with open(full_pickle, "rb") as file:
                data = pickle.load(file)

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
                try:
                    ind = frames.index(n)
                    dets = Assembler._flatten_detections(data[frame_names[ind]])
                    for det in dets:
                        if det.label not in bpts or det.confidence < pcutoff:
                            continue
                        x, y = det.pos
                        rr, cc = disk((y, x), dotsize, shape=(ny, nx))
                        frame[rr, cc] = colors[bpts.index(det.label)]
                except ValueError:  # No data stored for that particular frame
                    print(n, "no data")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("videos")
    cli_args = parser.parse_args()
