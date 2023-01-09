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

import glob
import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Foldernames3Dproject(cfg_3d):
    """Definitions of subfolders in 3D projects"""

    img_path = os.path.join(cfg_3d["project_path"], "calibration_images")
    path_corners = os.path.join(cfg_3d["project_path"], "corners")
    path_camera_matrix = os.path.join(cfg_3d["project_path"], "camera_matrix")
    path_undistort = os.path.join(cfg_3d["project_path"], "undistortion")
    path_removed_images = os.path.join(
        cfg_3d["project_path"], "removed_calibration_images"
    )

    return (
        img_path,
        path_corners,
        path_camera_matrix,
        path_undistort,
        path_removed_images,
    )


def create_empty_df(dataframe, scorer, flag):
    # Creates an empty dataFrame of same shape as df_side_view
    # flag = 2d or 3d

    df = dataframe
    bodyparts = df.columns.get_level_values("bodyparts").unique()
    a = np.full((df.shape[0], 3), np.nan)
    dataFrame = None
    for bodypart in bodyparts:
        if flag == "2d":
            pdindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "likelihood"]],
                names=["scorer", "bodyparts", "coords"],
            )
        elif flag == "3d":
            pdindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "z"]],
                names=["scorer", "bodyparts", "coords"],
            )
        frame = pd.DataFrame(a, columns=pdindex, index=range(0, df.shape[0]))
        dataFrame = pd.concat([frame, dataFrame], axis=1)
    return (dataFrame, scorer, bodyparts)


def compute_triangulation_calibration_images(
    stereo_matrix, projectedPoints1, projectedPoints2, path_undistort, cfg_3d, plot=True
):
    """
    Performs triangulation of the calibration images.
    """
    triangulate = []
    P1 = stereo_matrix["P1"]
    P2 = stereo_matrix["P2"]
    cmap = cfg_3d["colormap"]
    colormap = plt.get_cmap(cmap)
    markerSize = cfg_3d["dotsize"]
    markerType = cfg_3d["markerType"]

    for i in range(projectedPoints1.shape[0]):
        X_l = triangulatePoints(P1, P2, projectedPoints1[i], projectedPoints2[i])
        triangulate.append(X_l)
    triangulate = np.asanyarray(triangulate)

    # Plotting
    if plot == True:
        col = colormap(np.linspace(0, 1, triangulate.shape[0]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(triangulate.shape[0]):
            xs = triangulate[i, 0, :]
            ys = triangulate[i, 1, :]
            zs = triangulate[i, 2, :]
            ax.scatter(xs, ys, zs, c=col[i], marker=markerType, s=markerSize)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        plt.savefig(os.path.join(str(path_undistort), "checkerboard_3d.png"))
    return triangulate


def triangulatePoints(P1, P2, x1, x2):
    X = cv2.triangulatePoints(P1[:3], P2[:3], x1, x2)
    return X / X[3]


def get_camerawise_videos(path, cam_names, videotype):
    """
    This function returns the list of videos corresponding to the camera names specified in the cam_names.
    e.g. if cam_names = ['camera-1','camera-2']

    then it will return [['somename-camera-1-othername.avi', 'somename-camera-2-othername.avi']]
    """
    import glob
    from pathlib import Path

    vid = []

    # Find videos only specific to the cam names
    videos = [
        glob.glob(os.path.join(path, str("*" + cam_names[i] + "*" + videotype)))
        for i in range(len(cam_names))
    ]
    videos = [y for x in videos for y in x]

    # Exclude the labeled video files
    if "." in videotype:
        file_to_exclude = str("labeled" + videotype)
    else:
        file_to_exclude = str("labeled." + videotype)
    videos = [v for v in videos if os.path.isfile(v) and not (file_to_exclude in v)]
    video_list = []
    cam = cam_names[0]  # camera1
    vid.append(
        [
            name
            for name in glob.glob(os.path.join(path, str("*" + cam + "*" + videotype)))
        ]
    )  # all videos with cam
    # print("here is what I found",vid)
    for k in range(len(vid[0])):
        if cam in str(Path(vid[0][k]).stem):
            ending = Path(vid[0][k]).suffix
            pref = str(Path(vid[0][k]).stem).split(cam)[0]
            suf = str(Path(vid[0][k]).stem).split(cam)[1]
            if pref == "":
                if suf == "":
                    print("Strange naming convention on your part. Respect.")
                else:
                    putativecam2name = os.path.join(path, cam_names[1] + suf + ending)
            else:
                if suf == "":
                    putativecam2name = os.path.join(path, pref + cam_names[1] + ending)
                else:
                    putativecam2name = os.path.join(
                        path, pref + cam_names[1] + suf + ending
                    )
            # print([os.path.join(path,pref+cam+suf+ending),putativecam2name])
            if os.path.isfile(putativecam2name):
                # found a pair!!!
                video_list.append(
                    [os.path.join(path, pref + cam + suf + ending), putativecam2name]
                )
    return video_list


def Get_list_of_triangulated_and_videoFiles(
    filepath, videotype, scorer_3d, cam_names, videofolder
):
    """
    Returns the list of triangulated h5 and the corresponding video files.
    """

    prefix = []
    suffix = []
    file_list = []
    string_to_search = scorer_3d + ".h5"

    # Checks if filepath is a directory
    if [os.path.isdir(i) for i in filepath] == [True]:
        """
        Analyzes all the videos in the directory.
        """
        print("Analyzing all the videos in the directory")
        videofolder = filepath[0]
        cwd = os.getcwd()
        os.chdir(videofolder)
        triangulated_file_list = [
            fn for fn in os.listdir(os.curdir) if (string_to_search in fn)
        ]
        video_list = get_camerawise_videos(videofolder, cam_names, videotype)
        os.chdir(cwd)
        triangulated_folder = videofolder
    else:
        triangulated_file_list = [
            str(Path(fn).name) for fn in filepath if (string_to_search in fn)
        ]
        triangulated_folder = [
            str(Path(fn).parents[0]) for fn in filepath if (string_to_search in fn)
        ]
        triangulated_folder = triangulated_folder[0]

        if videofolder is None:
            videofolder = str(Path(filepath[0]).parents[0])
        video_list = get_camerawise_videos(videofolder, cam_names, videotype)

    # Get the filename of the triangulated file excluing the scorer name and remove any '-' or _ from it
    filename = [i.split(string_to_search)[0] for i in triangulated_file_list]
    for i in range(len(filename)):
        if filename[i][-1] == "_" or filename[i][-1] == "-":
            filename[i] = filename[i][:-1]
        if filename[i][0] == "_" or filename[i][0] == "-":
            filename[i] = filename[i][1:]

    # Get the suffix and prefix of the video filenames so that they can be used for matching the triangulated file names.
    for i in range(len(video_list)):
        pre = [
            str(Path(video_list[i][0]).stem).split(cam_names[0])[0],
            str(Path(video_list[i][1]).stem).split(cam_names[1])[0],
        ]
        suf = [
            str(Path(video_list[i][0]).stem).split(cam_names[0])[-1],
            str(Path(video_list[i][1]).stem).split(cam_names[1])[-1],
        ]
        for i in range(len(cam_names)):
            if pre[i] == "":
                pass
            elif pre[i][-1] == "_" or pre[i][-1] == "-":
                pre[i] = pre[i][:-1]
            if suf[i] == "":
                pass
            elif suf[i][0] == "_" or suf[i][0] == "-":
                suf[i] = suf[i][1:]
        suffix.append(suf)
        prefix.append(pre)

    # Match the suffix and prefix with the triangulated file name and return the list with triangulated file and corresponding video files.
    for k in range(len(filename)):
        for j in range(len(prefix)):
            if (prefix[j][0] in filename[k] and prefix[j][1] in filename[k]) and (
                suffix[j][0] in filename[k] and suffix[j][1] in filename[k]
            ):
                triangulated_file = glob.glob(
                    os.path.join(
                        triangulated_folder,
                        str("*" + filename[k] + "*" + string_to_search),
                    )
                )
                vfiles = get_camerawise_videos(videofolder, cam_names, videotype)
                vfiles = [
                    z for z in vfiles if prefix[j][0] in z[0] and suffix[j][0] in z[1]
                ][0]
                file_list.append(triangulated_file + vfiles)

    return file_list


def SaveMetadata3d(metadatafilename, metadata):
    with open(metadatafilename, "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def LoadMetadata3d(metadatafilename):
    with open(metadatafilename, "rb") as f:
        metadata = pickle.load(f)
        return metadata


def _reconstruct_tracks_as_tracklets(df):
    """
    Parameters:
    -----------
    df: DataFrame
        loaded from an .h5 tracks file (obtained from `stitch_tracklets()`)
    """
    from deeplabcut.refine_training_dataset.stitch import Tracklet

    tracklets = []
    for _, group in df.groupby("individuals", axis=1):
        temp = group.dropna(how="all")
        inds = temp.index.to_numpy()
        track = Tracklet(temp.to_numpy().reshape((len(temp), -1, 3)), inds)
        track = track.interpolate(max_gap=len(group))
        tracklets.append(track)
    return tracklets


def _associate_paired_view_tracks(tracklets1, tracklets2, F):
    """
    Computes the optimal matching between tracks in two cameras
    using the xFx'=0 epipolar constraint equation.

    Parameters:
    -----------
    tracklets1/2: Tracklet() object (defined in stitch.py)
    F: nd.array
        Fundamental matrix between cam1 and cam2
    """
    from scipy.optimize import linear_sum_assignment

    # Initialize costs matrix
    costs = np.zeros([len(tracklets1), len(tracklets2)])

    for i, t1 in enumerate(tracklets1):
        for j, t2 in enumerate(tracklets2):
            # get common bodypart detections in track pair
            _t1 = t1.xy[np.isin(t1.inds, t2.inds)]
            _t2 = t2.xy[np.isin(t2.inds, t1.inds)]

            # add 3rd dim to the points
            _t1 = np.c_[_t1, np.ones((*_t1.shape[:2], 1))]
            _t2 = np.c_[_t2, np.ones((*_t2.shape[:2], 1))]

            # cost for any point in time of t1 being the same
            # any point in time of t2
            cost = np.abs(np.nansum(np.matmul(_t1, F) * _t2, axis=2))

            # Get average cost of the entire track
            cost = cost.mean()
            costs[i, j] = cost

    match_inds = linear_sum_assignment(np.abs(costs))
    voting = dict(zip(*match_inds))

    return costs, voting


def cross_view_match_dataframes(df1, df2, F):
    """
    Computes the costs and matched voting for tracks between
    a camera pair

    df: Data read from .h5 track file
    F: fundamental matrix from OpenCV
    """

    tracks1 = _reconstruct_tracks_as_tracklets(df1)
    tracks2 = _reconstruct_tracks_as_tracklets(df2)
    costs, voting = _associate_paired_view_tracks(tracks1, tracks2, F)

    return costs, voting
