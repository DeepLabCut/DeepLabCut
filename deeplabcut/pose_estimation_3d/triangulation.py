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

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from deeplabcut.pose_estimation_tensorflow.lib.trackingutils import TRACK_METHODS

matplotlib_axes_logger.setLevel("ERROR")


def triangulate(
    config,
    video_path,
    videotype="",
    filterpredictions=True,
    filtertype="median",
    gputouse=None,
    destfolder=None,
    save_as_csv=False,
    track_method="",
):
    """
    This function triangulates the detected DLC-keypoints from the two camera views
    using the camera matrices (derived from calibration) to calculate 3D predictions.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    video_path : string/list of list
        Full path of the directory where videos are saved. If the user wants to analyze
        only a pair of videos, the user needs to pass them as a list of list of videos,
        i.e. [['video1-camera-1.avi','video1-camera-2.avi']]

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.


    filterpredictions: Bool, optional
        Filter the predictions with filter specified by "filtertype". If specified it
        should be either ``True`` or ``False``.

    filtertype: string
        Select which filter, 'arima' or 'median' filter (currently supported).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi).
        If you do not have a GPU put None.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video)

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``

    Example
    -------
    Linux/MacOS
    To analyze all the videos in the directory:
    >>> deeplabcut.triangulate(config,'/data/project1/videos/')

    To analyze only a few pairs of videos:
    >>> deeplabcut.triangulate(config,[['/data/project1/videos/video1-camera-1.avi','/data/project1/videos/video1-camera-2.avi'],['/data/project1/videos/video2-camera-1.avi','/data/project1/videos/video2-camera-2.avi']])


    Windows
    To analyze all the videos in the directory:
    >>> deeplabcut.triangulate(config,'C:\\yourusername\\rig-95\\Videos')

    To analyze only a few pair of videos:
    >>> deeplabcut.triangulate(config,[['C:\\yourusername\\rig-95\\Videos\\video1-camera-1.avi','C:\\yourusername\\rig-95\\Videos\\video1-camera-2.avi'],['C:\\yourusername\\rig-95\\Videos\\video2-camera-1.avi','C:\\yourusername\\rig-95\\Videos\\video2-camera-2.avi']])
    """
    from deeplabcut.pose_estimation_tensorflow import predict_videos
    from deeplabcut.post_processing import filtering

    cfg_3d = auxiliaryfunctions.read_config(config)
    cam_names = cfg_3d["camera_names"]
    pcutoff = cfg_3d["pcutoff"]
    scorer_3d = cfg_3d["scorername_3d"]

    snapshots = {}
    for cam in cam_names:
        snapshots[cam] = cfg_3d[str("config_file_" + cam)]
        # Check if the config file exists
        if not os.path.exists(snapshots[cam]):
            raise Exception(
                str(
                    "It seems the file specified in the variable config_file_"
                    + str(cam)
                )
                + " does not exist. Please edit the config file with correct file path and retry."
            )

    # flag to check if the video_path variable is a string or a list of list
    flag = False  # assumes that video path is a list
    if isinstance(video_path, str) == True:
        flag = True
        video_list = auxiliaryfunctions_3d.get_camerawise_videos(
            video_path, cam_names, videotype=videotype
        )
    else:
        video_list = video_path

    if video_list == []:
        print("No videos found in the specified video path.", video_path)
        print(
            "Please make sure that the video names are specified with correct camera names as entered in the config file or"
        )
        print(
            "perhaps the videotype is distinct from the videos in the path, I was looking for:",
            videotype,
        )

    print("List of pairs:", video_list)
    scorer_name = {}
    run_triangulate = False
    for i in range(len(video_list)):
        dataname = []
        for j in range(len(video_list[i])):  # looping over cameras
            if cam_names[j] not in video_list[i][j]:
                raise ValueError(
                    f"Camera name '{cam_names[j]}' "
                    f"not found in video list '{video_list[i][j]}'."
                )
            else:
                print(
                    "Analyzing video %s using %s"
                    % (video_list[i][j], str("config_file_" + cam_names[j]))
                )

                config_2d = snapshots[cam_names[j]]
                cfg = auxiliaryfunctions.read_config(config_2d)

                # Get track_method and do related checks
                track_method = auxfun_multianimal.get_track_method(
                    cfg, track_method=track_method
                )
                if (
                    len(cfg.get("multianimalbodyparts", [])) == 1
                    and track_method != "box"
                ):
                    warnings.warn(
                        "Switching to `box` tracker for single point tracking..."
                    )
                    track_method = "box"

                # Get track method suffix
                tr_method_suffix = TRACK_METHODS.get(track_method, "")

                shuffle = cfg_3d[str("shuffle_" + cam_names[j])]
                trainingsetindex = cfg_3d[str("trainingsetindex_" + cam_names[j])]
                trainFraction = cfg["TrainingFraction"][trainingsetindex]
                if flag == True:
                    video = os.path.join(video_path, video_list[i][j])
                else:
                    video_path = str(Path(video_list[i][j]).parents[0])
                    video = os.path.join(video_path, video_list[i][j])

                if destfolder is None:
                    destfolder = str(Path(video).parents[0])

                vname = Path(video).stem
                prefix = str(vname).split(cam_names[j])[0]
                suffix = str(vname).split(cam_names[j])[-1]
                if prefix == "":
                    pass
                elif prefix[-1] == "_" or prefix[-1] == "-":
                    prefix = prefix[:-1]

                if suffix == "":
                    pass
                elif suffix[0] == "_" or suffix[0] == "-":
                    suffix = suffix[1:]

                if prefix == "":
                    output_file = os.path.join(destfolder, suffix)
                else:
                    if suffix == "":
                        output_file = os.path.join(destfolder, prefix)
                    else:
                        output_file = os.path.join(destfolder, prefix + "_" + suffix)

                output_filename = os.path.join(
                    output_file + "_" + scorer_3d
                )  # Check if the videos are already analyzed for 3d
                if os.path.isfile(output_filename + ".h5"):
                    if save_as_csv is True and not os.path.exists(
                        output_filename + ".csv"
                    ):
                        # In case user adds save_as_csv is True after triangulating
                        pd.read_hdf(output_filename + ".h5").to_csv(
                            str(output_filename + ".csv")
                        )

                    print(
                        "Already analyzed...Checking the meta data for any change in the camera matrices and/or scorer names",
                        vname,
                    )
                    pickle_file = str(output_filename + "_meta.pickle")
                    metadata_ = auxiliaryfunctions_3d.LoadMetadata3d(pickle_file)
                    (
                        img_path,
                        path_corners,
                        path_camera_matrix,
                        path_undistort,
                        _,
                    ) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)
                    path_stereo_file = os.path.join(
                        path_camera_matrix, "stereo_params.pickle"
                    )
                    stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
                    cam_pair = str(cam_names[0] + "-" + cam_names[1])
                    is_video_analyzed = False  # variable to keep track if the video was already analyzed
                    # Check for the camera matrix
                    for k in metadata_["stereo_matrix"].keys():
                        if np.all(
                            metadata_["stereo_matrix"][k] == stereo_file[cam_pair][k]
                        ):
                            pass
                        else:
                            run_triangulate = True

                    # Check for scorer names in the pickle file of 3d output
                    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
                        cfg, shuffle, trainFraction, trainingsiterations="unknown"
                    )

                    if (
                        metadata_["scorer_name"][cam_names[j]] == DLCscorer
                    ):  # TODO: CHECK FOR BOTH?
                        is_video_analyzed = True
                    elif metadata_["scorer_name"][cam_names[j]] == DLCscorerlegacy:
                        is_video_analyzed = True
                    else:
                        is_video_analyzed = False
                        run_triangulate = True

                    if is_video_analyzed:
                        print("This file is already analyzed!")
                        dataname.append(
                            os.path.join(
                                destfolder, vname + DLCscorer + tr_method_suffix + ".h5"
                            )
                        )
                        scorer_name[cam_names[j]] = DLCscorer
                    else:
                        # Analyze video if score name is different
                        DLCscorer = predict_videos.analyze_videos(
                            config_2d,
                            [video],
                            videotype=videotype,
                            shuffle=shuffle,
                            trainingsetindex=trainingsetindex,
                            gputouse=gputouse,
                            destfolder=destfolder,
                        )
                        scorer_name[cam_names[j]] = DLCscorer
                        is_video_analyzed = False
                        run_triangulate = True
                        suffix = tr_method_suffix
                        if filterpredictions:
                            filtering.filterpredictions(
                                config_2d,
                                [video],
                                videotype=videotype,
                                shuffle=shuffle,
                                trainingsetindex=trainingsetindex,
                                filtertype=filtertype,
                                destfolder=destfolder,
                            )
                            suffix += "_filtered"

                        dataname.append(
                            os.path.join(destfolder, vname + DLCscorer + suffix + ".h5")
                        )

                else:  # need to do the whole jam.
                    DLCscorer = predict_videos.analyze_videos(
                        config_2d,
                        [video],
                        videotype=videotype,
                        shuffle=shuffle,
                        trainingsetindex=trainingsetindex,
                        gputouse=gputouse,
                        destfolder=destfolder,
                    )
                    scorer_name[cam_names[j]] = DLCscorer
                    run_triangulate = True
                    print(destfolder, vname, DLCscorer)
                    suffix = tr_method_suffix
                    if filterpredictions:
                        filtering.filterpredictions(
                            config_2d,
                            [video],
                            videotype=videotype,
                            shuffle=shuffle,
                            trainingsetindex=trainingsetindex,
                            filtertype=filtertype,
                            destfolder=destfolder,
                        )
                        suffix += "_filtered"
                    dataname.append(
                        os.path.join(destfolder, vname + DLCscorer + suffix + ".h5")
                    )

        if run_triangulate:
            #        if len(dataname)>0:
            # undistort points for this pair
            print("Undistorting...")
            (
                dataFrame_camera1_undistort,
                dataFrame_camera2_undistort,
                stereomatrix,
                path_stereo_file,
            ) = undistort_points(
                config, dataname, str(cam_names[0] + "-" + cam_names[1])
            )
            if len(dataFrame_camera1_undistort) != len(dataFrame_camera2_undistort):
                import warnings

                warnings.warn(
                    "The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry! Excluding the extra frames from the longer video."
                )
                if len(dataFrame_camera1_undistort) > len(dataFrame_camera2_undistort):
                    dataFrame_camera1_undistort = dataFrame_camera1_undistort[
                        : len(dataFrame_camera2_undistort)
                    ]
                if len(dataFrame_camera2_undistort) > len(dataFrame_camera1_undistort):
                    dataFrame_camera2_undistort = dataFrame_camera2_undistort[
                        : len(dataFrame_camera1_undistort)
                    ]
            #                raise Exception("The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry!")
            scorer_cam1 = dataFrame_camera1_undistort.columns.get_level_values(0)[0]
            scorer_cam2 = dataFrame_camera2_undistort.columns.get_level_values(0)[0]

            bodyparts = dataFrame_camera1_undistort.columns.get_level_values(
                "bodyparts"
            ).unique()

            P1 = stereomatrix["P1"]
            P2 = stereomatrix["P2"]
            F = stereomatrix["F"]

            print("Computing the triangulation...")

            num_frames = dataFrame_camera1_undistort.shape[0]
            ### Assign nan to [X,Y] of low likelihood predictions ###
            # Convert the data to a np array to easily mask out the low likelihood predictions
            data_cam1_tmp = dataFrame_camera1_undistort.to_numpy().reshape(
                (num_frames, -1, 3)
            )
            data_cam2_tmp = dataFrame_camera2_undistort.to_numpy().reshape(
                (num_frames, -1, 3)
            )
            # Assign [X,Y] = nan to low likelihood predictions
            data_cam1_tmp[data_cam1_tmp[..., 2] < pcutoff, :2] = np.nan
            data_cam2_tmp[data_cam2_tmp[..., 2] < pcutoff, :2] = np.nan

            # Reshape data back to original shape
            data_cam1_tmp = data_cam1_tmp.reshape(num_frames, -1)
            data_cam2_tmp = data_cam2_tmp.reshape(num_frames, -1)

            # put data back to the dataframes
            dataFrame_camera1_undistort[:] = data_cam1_tmp
            dataFrame_camera2_undistort[:] = data_cam2_tmp

            if cfg.get("multianimalproject"):
                # Check individuals are the same in both views
                individuals_view1 = (
                    dataFrame_camera1_undistort.columns.get_level_values("individuals")
                    .unique()
                    .to_list()
                )
                individuals_view2 = (
                    dataFrame_camera2_undistort.columns.get_level_values("individuals")
                    .unique()
                    .to_list()
                )
                if individuals_view1 != individuals_view2:
                    raise ValueError(
                        "The individuals do not match between the two DataFrames"
                    )

                # Cross-view match individuals
                _, voting = auxiliaryfunctions_3d.cross_view_match_dataframes(
                    dataFrame_camera1_undistort, dataFrame_camera2_undistort, F
                )
            else:
                # Create a dummy variables for single-animal
                individuals_view1 = ["indie"]
                voting = {0: 0}

            # Cleaner variable (since inds view1 == inds view2)
            individuals = individuals_view1

            # Reshape: (num_framex, num_individuals, num_bodyparts , 2)
            all_points_cam1 = dataFrame_camera1_undistort.to_numpy().reshape(
                (num_frames, len(individuals), -1, 3)
            )[..., :2]
            all_points_cam2 = dataFrame_camera2_undistort.to_numpy().reshape(
                (num_frames, len(individuals), -1, 3)
            )[..., :2]

            # Triangulate data
            triangulate = []
            for i, _ in enumerate(individuals):
                # i is individual in view 1
                # voting[i] is the matched individual in view 2

                pts_indv_cam1 = all_points_cam1[:, i].reshape((-1, 2)).T
                pts_indv_cam2 = all_points_cam2[:, voting[i]].reshape((-1, 2)).T

                indv_points_3d = auxiliaryfunctions_3d.triangulatePoints(
                    P1, P2, pts_indv_cam1, pts_indv_cam2
                )

                indv_points_3d = indv_points_3d[:3].T.reshape((num_frames, -1, 3))

                triangulate.append(indv_points_3d)

            triangulate = np.asanyarray(triangulate)
            metadata = {}
            metadata["stereo_matrix"] = stereomatrix
            metadata["stereo_matrix_file"] = path_stereo_file
            metadata["scorer_name"] = {
                cam_names[0]: scorer_name[cam_names[0]],
                cam_names[1]: scorer_name[cam_names[1]],
            }

            # Create 3D DataFrame column and row indices
            axis_labels = ("x", "y", "z")
            if cfg.get("multianimalproject"):
                columns = pd.MultiIndex.from_product(
                    [[scorer_3d], individuals, bodyparts, axis_labels],
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )

            else:
                columns = pd.MultiIndex.from_product(
                    [[scorer_3d], bodyparts, axis_labels],
                    names=["scorer", "bodyparts", "coords"],
                )

            inds = range(num_frames)

            # Swap num_animals with num_frames axes to ensure well-behaving reshape
            triangulate = triangulate.swapaxes(0, 1).reshape((num_frames, -1))

            # Fill up 3D dataframe
            df_3d = pd.DataFrame(triangulate, columns=columns, index=inds)

            df_3d.to_hdf(
                str(output_filename + ".h5"),
                "df_with_missing",
                format="table",
                mode="w",
            )

            # Reorder 2D dataframe in view 2 to match order of view 1
            if cfg.get("multianimalproject"):
                df_2d_view2 = pd.read_hdf(dataname[1])
                individuals_order = [individuals[i] for i in list(voting.values())]
                df_2d_view2 = auxfun_multianimal.reorder_individuals_in_df(
                    df_2d_view2, individuals_order
                )
                df_2d_view2.to_hdf(
                    dataname[1],
                    "tracks",
                    format="table",
                    mode="w",
                )

            auxiliaryfunctions_3d.SaveMetadata3d(
                str(output_filename + "_meta.pickle"), metadata
            )

            if save_as_csv:
                df_3d.to_csv(str(output_filename + ".csv"))

            print("Triangulated data for video", video_list[i])
            print("Results are saved under: ", destfolder)
            # have to make the dest folder none so that it can be updated for a new pair of videos
            if destfolder == str(Path(video).parents[0]):
                destfolder = None

    if len(video_list) > 0:
        print("All videos were analyzed...")
        print("Now you can create 3D video(s) using deeplabcut.create_labeled_video_3d")


def _undistort_points(points, mat, coeffs, p, r):
    pts = points.reshape((-1, 3))
    pts_undist = cv2.undistortPoints(
        src=pts[:, :2].astype(np.float32),
        cameraMatrix=mat,
        distCoeffs=coeffs,
        P=p,
        R=r,
    )
    pts[:, :2] = pts_undist.squeeze()
    return pts.reshape((points.shape[0], -1))


def _undistort_views(df_view_pairs, stereo_params):
    df_views_undist = []
    for df_view_pair, camera_pair in zip(df_view_pairs, stereo_params):
        params = stereo_params[camera_pair]
        dfs = []
        for i, df_view in enumerate(df_view_pair, start=1):
            pts_undist = _undistort_points(
                df_view.to_numpy(),
                params[f"cameraMatrix{i}"],
                params[f"distCoeffs{i}"],
                params[f"P{i}"],
                params[f"R{i}"],
            )
            df = pd.DataFrame(pts_undist, df_view.index, df_view.columns)
            dfs.append(df)
        df_views_undist.append(dfs)
    return df_views_undist


def undistort_points(config, dataframe, camera_pair):
    cfg_3d = auxiliaryfunctions.read_config(config)
    path_camera_matrix = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)[2]
    """
    path_undistort = destfolder
    filename_cam1 = Path(dataframe[0]).stem
    filename_cam2 = Path(dataframe[1]).stem

    #currently no interm. saving of this due to high speed.
    # check if the undistorted files are already present
    if os.path.exists(os.path.join(path_undistort,filename_cam1 + '_undistort.h5')) and os.path.exists(os.path.join(path_undistort,filename_cam2 + '_undistort.h5')):
        print("The undistorted files are already present at %s" % os.path.join(path_undistort,filename_cam1))
        dataFrame_cam1_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam1 + '_undistort.h5'))
        dataFrame_cam2_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam2 + '_undistort.h5'))
    else:
    """
    if len(dataframe) != 2:
        raise ValueError(
            f"undistort_points(config, dataframe, camera_pair) needs filenames to two data frames, but got dataframe={dataframe}."
        )
    for filename in dataframe:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Dataframe path '{filename}' could not be found in the filesystem."
            )
    if not os.path.exists(path_camera_matrix):
        raise FileNotFoundError(
            f"Camera matrix file '{path_camera_matrix}' could not be found in the filesystem."
        )
    # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
    dataframe_cam1 = pd.read_hdf(dataframe[0])
    dataframe_cam2 = pd.read_hdf(dataframe[1])
    path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
    stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
    dataFrame_cam1_undistort, dataFrame_cam2_undistort = _undistort_views(
        [(dataframe_cam1, dataframe_cam2)],
        stereo_file,
    )[0]

    return (
        dataFrame_cam1_undistort,
        dataFrame_cam2_undistort,
        stereo_file[camera_pair],
        path_stereo_file,
    )
