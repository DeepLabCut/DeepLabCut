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
"""
from __future__ import annotations

import argparse
import os
import pickle
import pandas as pd

####################################################
# Dependencies
####################################################
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deeplabcut.core import crossvalutils
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, visualization


def Histogram(vector, color, bins, ax=None, linewidth=1.0):
    dvector = np.diff(vector)
    dvector = dvector[np.isfinite(dvector)]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.hist(dvector, color=color, histtype="step", bins=bins, linewidth=linewidth)


def PlottingResults(
    tmpfolder,
    Dataframe,
    cfg,
    bodyparts2plot,
    individuals2plot,
    showfigures=False,
    suffix=".png",
    resolution=100,
    linewidth=1.0,
):
    """Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods."""
    pcutoff = cfg["pcutoff"]
    colors = visualization.get_cmap(len(bodyparts2plot), name=cfg["colormap"])
    alphavalue = cfg["alphavalue"]
    if individuals2plot:
        Dataframe = Dataframe.loc(axis=1)[:, individuals2plot]
    animal_bpts = Dataframe.columns.get_level_values("bodyparts")
    # Pose X vs pose Y
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("X position in pixels")
    ax1.set_ylabel("Y position in pixels")
    ax1.invert_yaxis()

    # Poses vs time
    fig2 = plt.figure(figsize=(10, 3))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("X-(dashed) and Y- (solid) position in pixels")

    # Likelihoods
    fig3 = plt.figure(figsize=(10, 3))
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Likelihood (use to set pcutoff)")

    # Histograms
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_ylabel("Count")
    ax4.set_xlabel("DeltaX and DeltaY")
    bins = np.linspace(0, np.amax(Dataframe.max()), 100)

    with np.errstate(invalid="ignore"):
        for bpindex, bp in enumerate(bodyparts2plot):
            if (
                bp in animal_bpts
            ):  # Avoid 'unique' bodyparts only present in the 'single' animal
                prob = Dataframe.xs(
                    (bp, "likelihood"), level=(-2, -1), axis=1
                ).values.squeeze()
                mask = prob < pcutoff
                temp_x = np.ma.array(
                    Dataframe.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
                    mask=mask,
                )
                temp_y = np.ma.array(
                    Dataframe.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
                    mask=mask,
                )
                ax1.plot(temp_x, temp_y, ".", color=colors(bpindex), alpha=alphavalue)

                ax2.plot(
                    temp_x,
                    "--",
                    color=colors(bpindex),
                    linewidth=linewidth,
                    alpha=alphavalue,
                )
                ax2.plot(
                    temp_y,
                    "-",
                    color=colors(bpindex),
                    linewidth=linewidth,
                    alpha=alphavalue,
                )

                ax3.plot(
                    prob,
                    "-",
                    color=colors(bpindex),
                    linewidth=linewidth,
                    alpha=alphavalue,
                )

                Histogram(temp_x, colors(bpindex), bins, ax4, linewidth=linewidth)
                Histogram(temp_y, colors(bpindex), bins, ax4, linewidth=linewidth)

    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cfg["colormap"]),
        norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1),
    )
    sm._A = []
    for ax in ax1, ax2, ax3, ax4:
        cbar = plt.colorbar(sm, ax=ax, ticks=range(len(bodyparts2plot)))
        cbar.set_ticklabels(bodyparts2plot)

    fig1.savefig(
        os.path.join(tmpfolder, "trajectory" + suffix),
        bbox_inches="tight",
        dpi=resolution,
    )
    fig2.savefig(
        os.path.join(tmpfolder, "plot" + suffix), bbox_inches="tight", dpi=resolution
    )
    fig3.savefig(
        os.path.join(tmpfolder, "plot-likelihood" + suffix),
        bbox_inches="tight",
        dpi=resolution,
    )
    fig4.savefig(
        os.path.join(tmpfolder, "hist" + suffix), bbox_inches="tight", dpi=resolution
    )

    if not showfigures:
        plt.close("all")
    else:
        plt.show()


##################################################
# Looping analysis over video
##################################################


def plot_trajectories(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    filtered=False,
    displayedbodyparts="all",
    displayedindividuals="all",
    showfigures=False,
    destfolder=None,
    modelprefix="",
    imagetype=".png",
    resolution=100,
    linewidth=1.0,
    track_method="",
    pcutoff: float | None = None,
):
    """Plots the trajectories of various bodyparts across the video.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    videos: list[str]
        Full paths to videos for analysis or a path to the directory, where all the
        videos with same extension are stored.

    videotype: str, optional, default=""
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions
        ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional, default=1
        Integer specifying the shuffle index of the training dataset.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        Note that TrainingFraction is a list in config.yaml.

    filtered: bool, optional, default=False
        Boolean variable indicating if filtered output should be plotted rather than
        frame-by-frame predictions. Filtered version can be calculated with
        ``deeplabcut.filterpredictions``.

    displayedbodyparts: list[str] or str, optional, default="all"
        This select the body parts that are plotted in the video.
        Either ``all``, then all body parts from config.yaml are used,
        or a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml
        to select only these two body parts.

    showfigures: bool, optional, default=False
        If ``True`` then plots are also displayed.

    destfolder: string or None, optional, default=None
        Specifies the destination folder that was used for storing analysis data. If
        ``None``, the path of the video is used.

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    imagetype: string, optional, default=".png"
        Specifies the output image format - '.tif', '.jpg', '.svg' and ".png".

    resolution: int, optional, default=100
        Specifies the resolution (in dpi) of saved figures.
        Note higher resolution figures take longer to generate.

    linewidth: float, optional, default=1.0
        Specifies width of line for line and histogram plots.

    track_method: string, optional, default=""
         Specifies the tracker used to generate the data.
         Empty by default (corresponding to a single animal project).
         For multiple animals, must be either 'box', 'skeleton', or 'ellipse' and will
         be taken from the config.yaml file if none is given.

    pcutoff: string, optional, default=None
        Overrides the pcutoff set in the project configuration to plot the trajectories.

    Returns
    -------
    None

    Examples
    --------

    To label the frames

    >>> deeplabcut.plot_trajectories(
            'home/alex/analysis/project/reaching-task/config.yaml',
            ['/home/alex/analysis/project/videos/reachingvideo1.avi'],
        )
    """
    cfg = auxiliaryfunctions.read_config(config)

    if pcutoff is None:
        pcutoff = cfg["pcutoff"]

    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg, shuffle, trainFraction, modelprefix=modelprefix
    )  # automatically loads corresponding model (even training iteration based on snapshot index)
    bodyparts = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        cfg, displayedbodyparts
    )
    individuals = auxfun_multianimal.IntersectionofIndividualsandOnesGivenbyUser(
        cfg, displayedindividuals
    )
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
    if not len(Videos):
        print(
            "No videos found. Make sure you passed a list of videos and that *videotype* is right."
        )
        return

    failures, multianimal_errors = [], []
    for video in Videos:
        if destfolder is None:
            videofolder = str(Path(video).parents[0])
        else:
            videofolder = destfolder

        vname = str(Path(video).stem)
        print("Loading ", video, "and data.")
        try:
            df, filepath, _, suffix = auxiliaryfunctions.load_analyzed_data(
                videofolder, vname, DLCscorer, filtered, track_method
            )
            tmpfolder = os.path.join(videofolder, "plot-poses", vname)
            _plot_trajectories(
                filepath,
                bodyparts,
                individuals,
                showfigures,
                resolution,
                linewidth,
                cfg["colormap"],
                cfg["alphavalue"],
                pcutoff,
                suffix,
                imagetype,
                tmpfolder,
            )
        except FileNotFoundError as e:
            print(e)
            failures.append(video)
            if track_method != "":
                # In a multi animal scenario, show more verbose errors.
                try:
                    _ = auxiliaryfunctions.load_detection_data(
                        video, DLCscorer, track_method
                    )
                    error_message = 'Call "deeplabcut.stitch_tracklets() prior to plotting the trajectories.'
                except FileNotFoundError as e:
                    print(e)
                    error_message = (
                        f"Make sure {video} was previously analyzed, and that "
                        "detections were successively converted to tracklets using "
                        '"deeplabcut.convert_detections2tracklets()" and "deeplabcut.stitch_tracklets()".'
                    )
                multianimal_errors.append(error_message)

    if len(failures) > 0:
        # Some vidoes were not evaluated.
        failed_videos = ",".join(failures)
        if len(multianimal_errors) > 0:
            verbose_error = ": " + " ".join(multianimal_errors)
        else:
            verbose_error = "."
        print(
            f"Plots could not be created for {failed_videos}. "
            f"Videos were not evaluated with the current scorer {DLCscorer}"
            + verbose_error
        )
    else:
        print(
            'Plots created! Please check the directory "plot-poses" within the video directory'
        )


def _plot_trajectories(
    h5file,
    bodyparts=None,
    individuals=None,
    show=False,
    resolution=100,
    linewidth=1.0,
    colormap="viridis",
    alpha=1.0,
    pcutoff=0.01,
    suffix="",
    image_type=".png",
    dest_folder=None,
):
    df = pd.read_hdf(h5file)
    if bodyparts is None:
        bodyparts = list(df.columns.get_level_values("bodyparts").unique())
    if individuals is None:
        try:
            individuals = set(df.columns.get_level_values("individuals"))
        except KeyError:
            individuals = [""]
    if dest_folder is None:
        vname = os.path.basename(h5file).split("DLC")[0]
        vid_folder = os.path.dirname(h5file)
        dest_folder = os.path.join(vid_folder, "plot-poses", vname)
    auxiliaryfunctions.attempt_to_make_folder(dest_folder, recursive=True)
    # Keep only the individuals and bodyparts that were labeled
    labeled_bpts = [
        bp
        for bp in df.columns.get_level_values("bodyparts").unique()
        if bp in bodyparts
    ]
    # Either display the animals defined in the config if they are found
    # in the dataframe, or all the trajectories regardless of their names
    try:
        animals = set(df.columns.get_level_values("individuals"))
    except KeyError:
        animals = {""}
    cfg = {
        "colormap": colormap,
        "alphavalue": alpha,
        "pcutoff": pcutoff,
    }
    for animal in animals.intersection(individuals) or animals:
        PlottingResults(
            dest_folder,
            df,
            cfg,
            labeled_bpts,
            animal,
            show,
            suffix + animal + image_type,
            resolution=resolution,
            linewidth=linewidth,
        )


def _plot_paf_performance(
    within,
    between,
    nbins=51,
    kde=True,
    colors=None,
    ax=None,
):
    import seaborn as sns

    bins = np.linspace(0, 1, nbins)
    if colors is None:
        colors = "#EFC9AF", "#1F8AC0"
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True, figsize=(3, 3))
    sns.histplot(within, kde=kde, ax=ax, stat="probability", color=colors[0], bins=bins)
    sns.histplot(
        between, kde=kde, ax=ax, stat="probability", color=colors[1], bins=bins
    )
    return ax


def plot_edge_affinity_distributions(
    eval_pickle_file,
    include_bodyparts="all",
    output_name="",
    figsize=(10, 7),
):
    """
    Display the distribution of affinity costs of within- and between-animal edges.

    Parameters
    ----------
    eval_pickle_file : string
        Path to a *_full.pickle from the evaluation-results folder.

    include_bodyparts : list of strings, optional
        A list of body part names whose edges are to be shown.
        By default, all body parts and their corresponding edges are analyzed.
        We recommend only passing a subset of body parts for projects with large graphs.

    output_name: string, optional
        Path where the plot is saved. By default, it is stored as costdist.png.

    figsize: tuple
        Figure size in inches.

    """

    with open(eval_pickle_file, "rb") as file:
        data = pickle.load(file)
    meta_pickle_file = eval_pickle_file.replace("_full.", "_meta.")
    with open(meta_pickle_file, "rb") as file:
        metadata = pickle.load(file)
    (w_train, _), (b_train, _) = crossvalutils._calc_within_between_pafs(
        data,
        metadata,
        train_set_only=True,
    )
    data.pop("metadata", None)
    nonempty = set(i for i, vals in w_train.items() if vals)
    meta = metadata["data"]["DLC-model-config file"]
    bpts = list(map(str.lower, meta["all_joints_names"]))
    inds_multi = set(b for edge in meta["partaffinityfield_graph"] for b in edge)
    if include_bodyparts == "all":
        include_bodyparts = inds_multi
    else:
        include_bodyparts = set(bpts.index(bpt) for bpt in include_bodyparts)
    edges_to_keep = set()
    graph = meta["partaffinityfield_graph"]
    for n, edge in enumerate(graph):
        if not any(i in include_bodyparts for i in edge):
            continue
        edges_to_keep.add(n)
    edge_inds = edges_to_keep.intersection(nonempty)
    nrows = int(np.ceil(np.sqrt(len(edge_inds))))
    ncols = int(np.ceil(len(edge_inds) / nrows))
    fig, axes_ = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        tight_layout=True,
        squeeze=False,
    )
    axes = axes_.flatten()
    for ax in axes:
        ax.axis("off")
    for n, ind in enumerate(edge_inds):
        i1, i2 = graph[ind]
        w_tr = w_train[ind]
        b_tr = b_train[ind]
        sep, _ = crossvalutils._calc_separability(b_tr, w_tr, metric="auc")
        axes[n].text(
            0.5,
            0.8,
            f"{bpts[i1]}–{bpts[i2]}\n{sep:.2f}",
            size=8,
            ha="center",
            transform=axes[n].transAxes,
        )
        _plot_paf_performance(w_tr, b_tr, ax=axes[n], kde=False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if not output_name:
        output_name = "costdist.jpg"
    fig.savefig(output_name, dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("video")
    cli_args = parser.parse_args()
