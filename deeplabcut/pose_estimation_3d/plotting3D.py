"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from deeplabcut.utils import (
    auxiliaryfunctions,
    auxiliaryfunctions_3d,
    make_labeled_video,
)
from deeplabcut.utils.auxfun_videos import VideoReader

matplotlib_axes_logger.setLevel("ERROR")
from matplotlib import gridspec
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm


def set_up_grid(figsize, xlim, ylim, zlim, view):
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    fig = plt.figure(figsize=figsize)
    axes1 = fig.add_subplot(gs[0, 0])
    axes2 = fig.add_subplot(gs[0, 1])
    axes3 = fig.add_subplot(gs[0, 2], projection="3d")
    axes3.set_xlim3d(xlim)
    axes3.set_ylim3d(ylim)
    axes3.set_zlim3d(zlim)
    axes3.set_box_aspect((1, 1, 1))
    axes3.set_xticklabels([])
    axes3.set_yticklabels([])
    axes3.set_zticklabels([])
    axes3.xaxis.grid(False)
    axes3.view_init(view[0], view[1])
    axes3.set_xlabel("X", fontsize=10)
    axes3.set_ylabel("Y", fontsize=10)
    axes3.set_zlabel("Z", fontsize=10)
    return fig, axes1, axes2, axes3


def create_labeled_video_3d(
    config,
    path,
    videofolder=None,
    start=0,
    end=None,
    trailpoints=0,
    videotype="",
    view=(-113, -270),
    xlim=None,
    ylim=None,
    zlim=None,
    draw_skeleton=True,
    color_by="bodypart",
    figsize=(20, 8),
    fps=30,
    dpi=300,
):
    """
    Creates a video with views from the two cameras and the 3d reconstruction for a selected number of frames.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    path : list
        A list of strings containing the full paths to triangulated files for analysis or a path to the directory, where all the triangulated files are stored.

    videofolder: string
        Full path of the folder where the videos are stored. Use this if the vidoes are stored in a different location other than where the triangulation files are stored. By default is ``None`` and therefore looks for video files in the directory where the triangulation file is stored.

    start: int
        Integer specifying the start of frame index to select. Default is set to 0.

    end: int
        Integer specifying the end of frame index to select. Default is set to None, where all the frames of the video are used for creating the labeled video.

    trailpoints: int
        Number of revious frames whose body parts are plotted in a frame (for displaying history). Default is set to 0.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    view: list
        A list that sets the elevation angle in z plane and azimuthal angle in x,y plane of 3d view. Useful for rotating the axis for 3d view

    xlim: list
        A list of integers specifying the limits for xaxis of 3d view. By default it is set to [None,None], where the x limit is set by taking the minimum and maximum value of the x coordinates for all the bodyparts.

    ylim: list
        A list of integers specifying the limits for yaxis of 3d view. By default it is set to [None,None], where the y limit is set by taking the minimum and maximum value of the y coordinates for all the bodyparts.

    zlim: list
        A list of integers specifying the limits for zaxis of 3d view. By default it is set to [None,None], where the z limit is set by taking the minimum and maximum value of the z coordinates for all the bodyparts.

    draw_skeleton: bool
        If ``True`` adds a line connecting the body parts making a skeleton on on each frame. The body parts to be connected and the color of these connecting lines are specified in the config file. By default: ``True``

    color_by : string, optional (default='bodypart')
        Coloring rule. By default, each bodypart is colored differently.
        If set to 'individual', points belonging to a single individual are colored the same.

    Example
    -------
    Linux/MacOs
    >>> deeplabcut.create_labeled_video_3d(config,['/data/project1/videos/3d.h5'],start=100, end=500)

    To create labeled videos for all the triangulated files in the folder
    >>> deeplabcut.create_labeled_video_3d(config,['/data/project1/videos'],start=100, end=500)

    To set the xlim, ylim, zlim and rotate the view of the 3d axis
    >>> deeplabcut.create_labeled_video_3d(config,['/data/project1/videos'],start=100, end=500,view=[30,90],xlim=[-12,12],ylim=[15,25],zlim=[20,30])

    """
    start_path = os.getcwd()

    # Read the config file and related variables
    cfg_3d = auxiliaryfunctions.read_config(config)
    cam_names = cfg_3d["camera_names"]
    pcutoff = cfg_3d["pcutoff"]
    markerSize = cfg_3d["dotsize"]
    alphaValue = cfg_3d["alphaValue"]
    cmap = cfg_3d["colormap"]
    bodyparts2connect = cfg_3d["skeleton"]
    skeleton_color = cfg_3d["skeleton_color"]
    scorer_3d = cfg_3d["scorername_3d"]

    if color_by not in ("bodypart", "individual"):
        raise ValueError(f"Invalid color_by={color_by}")

    file_list = auxiliaryfunctions_3d.Get_list_of_triangulated_and_videoFiles(
        path, videotype, scorer_3d, cam_names, videofolder
    )
    print(file_list)
    if file_list == []:
        raise Exception(
            "No corresponding video file(s) found for the specified triangulated file or folder. Did you specify the video file type? If videos are stored in a different location, please use the ``videofolder`` argument to specify their path."
        )

    for file in file_list:
        path_h5_file = Path(file[0]).parents[0]
        triangulate_file = file[0]
        # triangulated file is a list which is always sorted as [triangulated.h5,camera-1.videotype,camera-2.videotype]
        # name for output video
        file_name = str(Path(triangulate_file).stem)
        videooutname = os.path.join(path_h5_file, file_name + ".mp4")
        if os.path.isfile(videooutname):
            print("Video already created...")
        else:
            string_to_remove = str(Path(triangulate_file).suffix)
            pickle_file = triangulate_file.replace(string_to_remove, "_meta.pickle")
            metadata_ = auxiliaryfunctions_3d.LoadMetadata3d(pickle_file)

            base_filename_cam1 = str(Path(file[1]).stem).split(videotype)[
                0
            ]  # required for searching the filtered file
            base_filename_cam2 = str(Path(file[2]).stem).split(videotype)[
                0
            ]  # required for searching the filtered file
            cam1_view_video = file[1]
            cam2_view_video = file[2]
            cam1_scorer = metadata_["scorer_name"][cam_names[0]]
            cam2_scorer = metadata_["scorer_name"][cam_names[1]]
            print(
                "Creating 3D video from %s and %s using %s"
                % (
                    Path(cam1_view_video).name,
                    Path(cam2_view_video).name,
                    Path(triangulate_file).name,
                )
            )

            # Read the video files and corresponfing h5 files
            vid_cam1 = VideoReader(cam1_view_video)
            vid_cam2 = VideoReader(cam2_view_video)

            # Look for the filtered predictions file
            try:
                print("Looking for filtered predictions...")
                df_cam1 = pd.read_hdf(
                    glob.glob(
                        os.path.join(
                            path_h5_file,
                            str(
                                "*" + base_filename_cam1 + cam1_scorer + "*filtered.h5"
                            ),
                        )
                    )[0]
                )
                df_cam2 = pd.read_hdf(
                    glob.glob(
                        os.path.join(
                            path_h5_file,
                            str(
                                "*" + base_filename_cam2 + cam2_scorer + "*filtered.h5"
                            ),
                        )
                    )[0]
                )
                # print("Found filtered predictions, will be use these for triangulation.")
                print(
                    "Found the following filtered data: ",
                    os.path.join(
                        path_h5_file,
                        str("*" + base_filename_cam1 + cam1_scorer + "*filtered.h5"),
                    ),
                    os.path.join(
                        path_h5_file,
                        str("*" + base_filename_cam2 + cam2_scorer + "*filtered.h5"),
                    ),
                )
            except FileNotFoundError:
                print(
                    "No filtered predictions found, the unfiltered predictions will be used instead."
                )
                df_cam1 = pd.read_hdf(
                    glob.glob(
                        os.path.join(
                            path_h5_file, str(base_filename_cam1 + cam1_scorer + "*.h5")
                        )
                    )[0]
                )
                df_cam2 = pd.read_hdf(
                    glob.glob(
                        os.path.join(
                            path_h5_file, str(base_filename_cam2 + cam2_scorer + "*.h5")
                        )
                    )[0]
                )

            df_3d = pd.read_hdf(triangulate_file)
            try:
                num_animals = df_3d.columns.get_level_values("individuals").unique().size
            except KeyError:
                num_animals = 1

            if end is None:
                end = len(df_3d)  # All the frames
            end = min(end, min(len(vid_cam1), len(vid_cam2)))
            frames = list(range(start, end))

            output_folder = Path(os.path.join(path_h5_file, "temp_" + file_name))
            output_folder.mkdir(parents=True, exist_ok=True)

            # Flatten the list of bodyparts to connect
            bodyparts2plot = list(
                np.unique([val for sublist in bodyparts2connect for val in sublist])
            )

            # Format data
            mask2d = df_cam1.columns.get_level_values('bodyparts').isin(bodyparts2plot)
            xy1 = df_cam1.loc[:, mask2d].to_numpy().reshape((len(df_cam1), -1, 3))
            visible1 = xy1[..., 2] >= pcutoff
            xy1[~visible1] = np.nan
            xy2 = df_cam2.loc[:, mask2d].to_numpy().reshape((len(df_cam1), -1, 3))
            visible2 = xy2[..., 2] >= pcutoff
            xy2[~visible2] = np.nan
            mask = df_3d.columns.get_level_values('bodyparts').isin(bodyparts2plot)
            xyz = df_3d.loc[:, mask].to_numpy().reshape((len(df_3d), -1, 3))
            xyz[~(visible1 & visible2)] = np.nan

            bpts = df_3d.columns.get_level_values('bodyparts')[mask][::3]
            links = make_labeled_video.get_segment_indices(
                bodyparts2connect, bpts,
            )
            ind_links = tuple(zip(*links))

            if color_by == "bodypart":
                color = plt.cm.get_cmap(cmap, len(bodyparts2plot))
                colors_ = color(range(len(bodyparts2plot)))
                colors = np.tile(colors_, (num_animals, 1))
            elif color_by == "individual":
                color = plt.cm.get_cmap(cmap, num_animals)
                colors_ = color(range(num_animals))
                colors = np.repeat(colors_, len(bodyparts2plot), axis=0)

            # Trick to force equal aspect ratio of 3D plots
            minmax = np.nanpercentile(xyz[frames], q=[25, 75], axis=(0, 1)).T
            minmax *= 1.1
            minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
            if xlim is None:
                mid_x = np.mean(minmax[0])
                xlim = mid_x - minmax_range, mid_x + minmax_range
            if ylim is None:
                mid_y = np.mean(minmax[1])
                ylim = mid_y - minmax_range, mid_y + minmax_range
            if zlim is None:
                mid_z = np.mean(minmax[2])
                zlim = mid_z - minmax_range, mid_z + minmax_range

            # Set up the matplotlib figure beforehand
            fig, axes1, axes2, axes3 = set_up_grid(figsize, xlim, ylim, zlim, view)
            points_2d1 = axes1.scatter(
                *np.zeros((2, len(bodyparts2plot))), s=markerSize, alpha=alphaValue,
            )
            im1 = axes1.imshow(np.zeros((vid_cam1.height, vid_cam1.width)))
            points_2d2 = axes2.scatter(
                *np.zeros((2, len(bodyparts2plot))), s=markerSize, alpha=alphaValue,
            )
            im2 = axes2.imshow(np.zeros((vid_cam2.height, vid_cam2.width)))
            points_3d = axes3.scatter(
                *np.zeros((3, len(bodyparts2plot))), s=markerSize, alpha=alphaValue,
            )
            if draw_skeleton:
                # Set up skeleton LineCollections
                segs = np.zeros((2, len(ind_links), 2))
                coll1 = LineCollection(segs, colors=skeleton_color)
                coll2 = LineCollection(segs, colors=skeleton_color)
                axes1.add_collection(coll1)
                axes2.add_collection(coll2)
                segs = np.zeros((2, len(ind_links), 3))
                coll_3d = Line3DCollection(segs, colors=skeleton_color)
                axes3.add_collection(coll_3d)

            writer = FFMpegWriter(fps=fps)
            with writer.saving(fig, videooutname, dpi=dpi):
                for k in tqdm(frames):
                    vid_cam1.set_to_frame(k)
                    vid_cam2.set_to_frame(k)
                    frame_cam1 = vid_cam1.read_frame()
                    frame_cam2 = vid_cam2.read_frame()
                    if frame_cam1 is None or frame_cam2 is None:
                        raise IOError("A video frame is empty.")

                    im1.set_data(frame_cam1)
                    im2.set_data(frame_cam2)

                    sl = slice(max(0, k - trailpoints), k + 1)
                    coords3d = xyz[sl]
                    coords1 = xy1[sl, :, :2]
                    coords2 = xy2[sl, :, :2]
                    points_3d._offsets3d = coords3d.reshape((-1, 3)).T
                    points_3d.set_color(colors)
                    points_2d1.set_offsets(coords1.reshape((-1, 2)))
                    points_2d1.set_color(colors)
                    points_2d2.set_offsets(coords2.reshape((-1, 2)))
                    points_2d2.set_color(colors)
                    if draw_skeleton:
                        segs3d = xyz[k][tuple([ind_links])].swapaxes(0, 1)
                        coll_3d.set_segments(segs3d)
                        segs1 = xy1[k, :, :2][tuple([ind_links])].swapaxes(0, 1)
                        coll1.set_segments(segs1)
                        segs2 = xy2[k, :, :2][tuple([ind_links])].swapaxes(0, 1)
                        coll2.set_segments(segs2)

                    writer.grab_frame()
