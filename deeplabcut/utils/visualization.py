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

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from skimage import io, color
from tqdm import trange

from deeplabcut.utils import auxiliaryfunctions, auxfun_videos


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def make_labeled_image(
    frame,
    DataCombined,
    imagenr,
    pcutoff,
    Scorers,
    bodyparts,
    colors,
    cfg,
    labels=["+", ".", "x"],
    scaling=1,
    ax=None,
):
    """Creating a labeled image with the original human labels, as well as the DeepLabCut's!"""

    alphavalue = cfg["alphavalue"]  # .5
    dotsize = cfg["dotsize"]  # =15

    if ax is None:
        if np.ndim(frame) > 2:  # color image!
            h, w, numcolors = np.shape(frame)
        else:
            h, w = np.shape(frame)
        _, ax = prepare_figure_axes(w, h, scaling)
    ax.imshow(frame, "gray")
    for scorerindex, loopscorer in enumerate(Scorers):
        for bpindex, bp in enumerate(bodyparts):
            if np.isfinite(
                DataCombined[loopscorer][bp]["y"][imagenr]
                + DataCombined[loopscorer][bp]["x"][imagenr]
            ):
                y, x = (
                    int(DataCombined[loopscorer][bp]["y"][imagenr]),
                    int(DataCombined[loopscorer][bp]["x"][imagenr]),
                )
                if cfg["scorer"] not in loopscorer:
                    p = DataCombined[loopscorer][bp]["likelihood"][imagenr]
                    if p > pcutoff:
                        ax.plot(
                            x,
                            y,
                            labels[1],
                            ms=dotsize,
                            alpha=alphavalue,
                            color=colors(int(bpindex)),
                        )
                    else:
                        ax.plot(
                            x,
                            y,
                            labels[2],
                            ms=dotsize,
                            alpha=alphavalue,
                            color=colors(int(bpindex)),
                        )
                else:  # this is the human labeler
                    ax.plot(
                        x,
                        y,
                        labels[0],
                        ms=dotsize,
                        alpha=alphavalue,
                        color=colors(int(bpindex)),
                    )
    return ax


def make_multianimal_labeled_image(
    frame,
    coords_truth,
    coords_pred,
    probs_pred,
    colors,
    dotsize=12,
    alphavalue=0.7,
    pcutoff=0.6,
    labels=["+", ".", "x"],
    ax=None,
):
    if ax is None:
        h, w, _ = np.shape(frame)
        _, ax = prepare_figure_axes(w, h)
    ax.imshow(frame, "gray")
    for n, data in enumerate(zip(coords_truth, coords_pred, probs_pred)):
        color = colors(n)
        coord_gt, coord_pred, prob_pred = data

        ax.plot(*coord_gt.T, labels[0], ms=dotsize, alpha=alphavalue, color=color)
        if not coord_pred.shape[0]:
            continue

        reliable = np.repeat(prob_pred >= pcutoff, coord_pred.shape[1], axis=1)
        ax.plot(
            *coord_pred[reliable[:, 0]].T,
            labels[1],
            ms=dotsize,
            alpha=alphavalue,
            color=color,
        )
        if not np.all(reliable):
            ax.plot(
                *coord_pred[~reliable[:, 0]].T,
                labels[2],
                ms=dotsize,
                alpha=alphavalue,
                color=color,
            )
    return ax


def plot_and_save_labeled_frame(
    DataCombined,
    ind,
    trainIndices,
    cfg,
    colors,
    comparisonbodyparts,
    DLCscorer,
    foldername,
    fig,
    ax,
    scaling=1,
):
    if isinstance(DataCombined.index[ind], tuple):
        image_path = os.path.join(cfg["project_path"], *DataCombined.index[ind])
    else:
        image_path = os.path.join(cfg["project_path"], DataCombined.index[ind])
    frame = io.imread(image_path)
    if np.ndim(frame) > 2:  # color image!
        h, w, numcolors = np.shape(frame)
    else:
        h, w = np.shape(frame)
    fig.set_size_inches(w / 100, h / 100)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax = make_labeled_image(
        frame,
        DataCombined,
        ind,
        cfg["pcutoff"],
        [cfg["scorer"], DLCscorer],
        comparisonbodyparts,
        colors,
        cfg,
        scaling=scaling,
        ax=ax,
    )
    save_labeled_frame(fig, image_path, foldername, ind in trainIndices)
    return ax


def save_labeled_frame(fig, image_path, dest_folder, belongs_to_train):
    path = Path(image_path)
    imagename = path.parts[-1]
    imfoldername = path.parts[-2]
    if belongs_to_train:
        dest = "-".join(("Training", imfoldername, imagename))
    else:
        dest = "-".join(("Test", imfoldername, imagename))
    full_path = os.path.join(dest_folder, dest)

    # Windows throws error if file path is > 260 characters, can fix with prefix.
    # See https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
    if len(full_path) >= 260 and os.name == "nt":
        full_path = "\\\\?\\" + full_path
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(full_path)


def create_minimal_figure(dpi=100):
    fig, ax = plt.subplots(frameon=False, dpi=dpi)
    ax.axis("off")
    ax.invert_yaxis()
    return fig, ax


def erase_artists(ax):
    for artist in ax.lines + ax.collections + ax.artists + ax.patches + ax.images:
        artist.remove()
    ax.figure.canvas.draw_idle()


def prepare_figure_axes(width, height, scale=1.0, dpi=100):
    fig = plt.figure(
        frameon=False, figsize=(width * scale / dpi, height * scale / dpi), dpi=dpi
    )
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    return fig, ax


def make_labeled_images_from_dataframe(
    df,
    cfg,
    destfolder="",
    scale=1.0,
    dpi=100,
    keypoint="+",
    draw_skeleton=True,
    color_by="bodypart",
):
    """
    Write labeled frames to disk from a DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the labeled data. Typically, the DataFrame is obtained
        through pandas.read_csv() or pandas.read_hdf().
    cfg : dict
        Project configuration.
    destfolder : string, optional
        Destination folder into which images will be stored. By default, same location as the labeled data.
        Note that the folder will be created if it does not exist.
    scale : float, optional
        Up/downscale the output dimensions.
        By default, outputs are of the same dimensions as the original images.
    dpi : int, optional
        Output resolution. 100 dpi by default.
    keypoint : str, optional
        Keypoint appearance. By default, keypoints are marked by a + sign.
        Refer to https://matplotlib.org/3.2.1/api/markers_api.html for a list of all possible options.
    draw_skeleton : bool, optional
        Whether to draw the animal skeleton as defined in *cfg*. True by default.
    color_by : str, optional
        Color scheme of the keypoints. Must be either 'bodypart' or 'individual'.
        By default, keypoints are colored relative to the bodypart they represent.
    """

    bodyparts = df.columns.get_level_values("bodyparts")
    bodypart_names = bodyparts.unique()
    nbodyparts = len(bodypart_names)
    bodyparts = bodyparts[::2]
    draw_skeleton = (
        draw_skeleton and cfg["skeleton"]
    )  # Only draw if a skeleton is defined

    if color_by == "bodypart":
        map_ = bodyparts.map(dict(zip(bodypart_names, range(nbodyparts))))
        cmap = get_cmap(nbodyparts, cfg["colormap"])
        colors = cmap(map_)
    elif color_by == "individual":
        try:
            individuals = df.columns.get_level_values("individuals")
            individual_names = individuals.unique().to_list()
            nindividuals = len(individual_names)
            individuals = individuals[::2]
            map_ = individuals.map(dict(zip(individual_names, range(nindividuals))))
            cmap = get_cmap(nindividuals, cfg["colormap"])
            colors = cmap(map_)
        except KeyError as e:
            raise Exception(
                "Coloring by individuals is only valid for multi-animal data"
            ) from e
    else:
        raise ValueError("`color_by` must be either `bodypart` or `individual`.")

    bones = []
    if draw_skeleton:
        for bp1, bp2 in cfg["skeleton"]:
            match1, match2 = [], []
            for j, bp in enumerate(bodyparts):
                if bp == bp1:
                    match1.append(j)
                elif bp == bp2:
                    match2.append(j)
            bones.extend(zip(match1, match2))
    ind_bones = tuple(zip(*bones))

    images_list = [
        os.path.join(cfg["project_path"], *tuple_) for tuple_ in df.index.tolist()
    ]
    if not destfolder:
        destfolder = os.path.dirname(images_list[0])
    tmpfolder = destfolder + "_labeled"
    auxiliaryfunctions.attempt_to_make_folder(tmpfolder)
    ic = io.imread_collection(images_list)

    h, w = ic[0].shape[:2]
    all_same_shape = True
    for array in ic[1:]:
        if array.shape[:2] != (h, w):
            all_same_shape = False
            break

    xy = df.values.reshape((df.shape[0], -1, 2))
    segs = xy[:, ind_bones].swapaxes(1, 2)

    s = cfg["dotsize"]
    alpha = cfg["alphavalue"]
    if all_same_shape:  # Very efficient, avoid re-drawing the whole plot
        fig, ax = prepare_figure_axes(w, h, scale, dpi)
        im = ax.imshow(np.zeros((h, w)), "gray")
        pts = [ax.plot([], [], keypoint, ms=s, alpha=alpha, color=c)[0] for c in colors]
        coll = LineCollection([], colors=cfg["skeleton_color"], alpha=alpha)
        ax.add_collection(coll)
        for i in trange(len(ic)):
            filename = ic.files[i]
            ind = images_list.index(filename)
            coords = xy[ind]
            img = ic[i]
            if img.ndim == 2 or img.shape[-1] == 1:
                img = color.gray2rgb(ic[i])
            im.set_data(img)
            for pt, coord in zip(pts, coords):
                pt.set_data(*np.expand_dims(coord, axis=1))
            if ind_bones:
                coll.set_segments(segs[ind])
            imagename = os.path.basename(filename)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.savefig(
                os.path.join(tmpfolder, imagename.replace(".png", f"_{color_by}.png")),
                dpi=dpi,
            )
        plt.close(fig)

    else:  # Good old inelegant way
        for i in trange(len(ic)):
            filename = ic.files[i]
            ind = images_list.index(filename)
            coords = xy[ind]
            image = ic[i]
            h, w = image.shape[:2]
            fig, ax = prepare_figure_axes(w, h, scale, dpi)
            ax.imshow(image)
            for coord, c in zip(coords, colors):
                ax.plot(*coord, keypoint, ms=s, alpha=alpha, color=c)
            if ind_bones:
                coll = LineCollection(
                    segs[ind], colors=cfg["skeleton_color"], alpha=alpha
                )
                ax.add_collection(coll)
            imagename = os.path.basename(filename)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.savefig(
                os.path.join(tmpfolder, imagename.replace(".png", f"_{color_by}.png")),
                dpi=dpi,
            )
            plt.close(fig)


def plot_evaluation_results(
    df_combined: pd.DataFrame,
    project_root: str,
    scorer: str,
    model_name: str,
    output_folder: str,
    in_train_set: bool,
    plot_unique_bodyparts: bool = False,
    mode: str = "bodypart",
    colormap: str = "rainbow",
    dot_size: int = 12,
    alpha_value: float = 0.7,
    p_cutoff: float = 0.6,
) -> None:
    """
    Creates labeled images using the results of inference, and saves them to an output
    folder.

    Args:
        df_combined: dataframe with multiindex rows ("labeled-data", video_name,
            image_name) and columns ("scorer", "individuals", "bodyparts", "coords").
            There should be two scorers: scorer (for ground truth data) and model_name
            (for prediction data)
        project_root: the project root path
        scorer: the name of the scorer for ground truth data in df_combined
        model_name: the name of the model for predictions in df_combined
        output_folder: the name of the folder where images should be saved
        in_train_set: whether df_combined is for train set images
        plot_unique_bodyparts: whether we should plot unique bodyparts
        mode: one of {"bodypart", "individual"}. Determines the keypoint color grouping
        colormap: the colormap to use for keypoints
        dot_size: the dot size to use for keypoints
        alpha_value: the alpha value to use for keypoints
        p_cutoff: the p-cutoff for "confident" keypoints
    """
    for row_index, row in df_combined.iterrows():
        if isinstance(row_index, str):
            image_rel_path = Path(row_index)
            data_folder = image_rel_path.parent.parent.name
            video = image_rel_path.parent.name
            image = image_rel_path.name
        else:
            data_folder, video, image = row_index

        image_path = Path(project_root) / data_folder / video / image
        frame = auxfun_videos.imread(str(image_path), mode="skimage")

        row_multi = row.loc[
            (slice(None), row.index.get_level_values("individuals") != "single")
        ]
        individuals = len(row_multi.index.get_level_values("individuals").unique())
        bodyparts = len(row_multi.index.get_level_values("bodyparts").unique())
        df_gt = row_multi[scorer]
        df_predictions = row_multi[model_name]

        # Shape (num_individuals, num_bodyparts, xy)
        ground_truth = df_gt.to_numpy().reshape((individuals, bodyparts, 2))
        predictions = df_predictions.to_numpy().reshape((individuals, bodyparts, 3))

        if plot_unique_bodyparts:
            row_unique = row.loc[
                (slice(None), row.index.get_level_values("individuals") == "single")
            ]
            unique_individuals = 1
            unique_bodyparts = len(
                row_unique.index.get_level_values("bodyparts").unique()
            )
            unique_ground_truth = (
                row_unique[scorer]
                .to_numpy()
                .reshape((unique_individuals, unique_bodyparts, 2))
            )
            unique_predictions = (
                row_unique[model_name]
                .to_numpy()
                .reshape((unique_individuals, unique_bodyparts, 3))
            )

        fig, ax = create_minimal_figure()
        h, w, _ = np.shape(frame)
        fig.set_size_inches(w / 100, h / 100)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()

        if mode == "bodypart":
            num_colors = bodyparts
            if plot_unique_bodyparts:
                num_colors += unique_bodyparts

            colors = get_cmap(num_colors, name=colormap)
            predictions = predictions.swapaxes(0, 1)
            ground_truth = ground_truth.swapaxes(0, 1)
        elif mode == "individual":
            colors = get_cmap(individuals + 1, name=colormap)
        else:
            colors = []

        ax = make_multianimal_labeled_image(
            frame,
            ground_truth,
            predictions[:, :, :2],
            predictions[:, :, 2:],
            colors,
            dot_size,
            alpha_value,
            p_cutoff,
            ax=ax,
        )
        if plot_unique_bodyparts:
            unique_predictions = unique_predictions.swapaxes(0, 1)
            unique_ground_truth = unique_ground_truth.swapaxes(0, 1)
            ax = make_multianimal_labeled_image(
                frame,
                unique_ground_truth,
                unique_predictions[:, :, :2],
                unique_predictions[:, :, 2:],
                colors,
                dot_size,
                alpha_value,
                p_cutoff,
                ax=ax,
            )

        save_labeled_frame(
            fig,
            str(image_path),
            output_folder,
            belongs_to_train=in_train_set,
        )
        erase_artists(ax)
        plt.close()
