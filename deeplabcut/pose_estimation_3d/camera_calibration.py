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

import glob
import os
import pickle
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d

matplotlib_axes_logger.setLevel("ERROR")


def calibrate_cameras(config, cbrow=8, cbcol=6, calibrate=False, alpha=0.4):
    """This function extracts the corners points from the calibration images, calibrates the camera and stores the calibration files in the project folder (defined in the config file).

    Make sure you have around 20-60 pairs of calibration images. The function should be used iteratively to select the right set of calibration images.

    A pair of calibration image is considered "correct", if the corners are detected correctly in both the images. It may happen that during the first run of this function,
    the extracted corners are incorrect or the order of detected corners does not align for the corresponding views (i.e. camera-1 and camera-2 images).

    In such a case, remove those pairs of images and re-run this function. Once the right number of calibration images are selected,
    use the parameter ``calibrate=True`` to calibrate the cameras.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    cbrow : int
        Integer specifying the number of rows in the calibration image.

    cbcol : int
        Integer specifying the number of columns in the calibration image.

    calibrate : bool
        If this is set to True, the cameras are calibrated with the current set of calibration images. The default is ``False``
        Set it to True, only after checking the results of the corner detection method and removing dysfunctional images!

    alpha: float
        Floating point number between 0 and 1 specifying the free scaling parameter. When alpha = 0, the rectified images with only valid pixels are stored
        i.e. the rectified images are zoomed in. When alpha = 1, all the pixels from the original images are retained.
        For more details: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Example
    --------
    Linux/MacOs/Windows
    >>> deeplabcut.calibrate_camera(config)

    Once the right set of calibration images are selected,
    >>> deeplabcut.calibrate_camera(config,calibrate=True)

    """
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Read the config file
    cfg_3d = auxiliaryfunctions.read_config(config)
    (
        img_path,
        path_corners,
        path_camera_matrix,
        path_undistort,
        path_removed_images,
    ) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)

    images = glob.glob(os.path.join(img_path, "*.jpg"))
    cam_names = cfg_3d["camera_names"]

    # update the variable snapshot* in config file according to the name of the cameras
    try:
        for i in range(len(cam_names)):
            cfg_3d[str("config_file_" + cam_names[i])] = cfg_3d.pop(
                str("config_file_camera-" + str(i + 1))
            )
        for i in range(len(cam_names)):
            cfg_3d[str("shuffle_" + cam_names[i])] = cfg_3d.pop(
                str("shuffle_camera-" + str(i + 1))
            )
    except:
        pass

    project_path = cfg_3d["project_path"]
    projconfigfile = os.path.join(str(project_path), "config.yaml")
    auxiliaryfunctions.write_config_3d(projconfigfile, cfg_3d)

    # Initialize the dictionary
    img_shape = {}
    objpoints = {}  # 3d point in real world space
    imgpoints = {}  # 2d points in image plane.
    dist_pickle = {}
    stereo_params = {}
    for cam in cam_names:
        objpoints.setdefault(cam, [])
        imgpoints.setdefault(cam, [])
        dist_pickle.setdefault(cam, [])

    # Sort the images.
    images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if len(images) == 0:
        raise Exception(
            "No calibration images found. Make sure the calibration images are saved as .jpg and with prefix as the camera name as specified in the config.yaml file."
        )

    skip_images = []
    for fname in images:
        for cam in cam_names:
            if cam in fname and Path(fname).name not in skip_images:
                filename = Path(fname).stem
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(
                    gray, (cbcol, cbrow), None
                )  #  (8,6) pattern (dimensions = common points of black squares)
                # If found, add object points, image points (after refining them)

                if ret == True:
                    img_shape[cam] = gray.shape[::-1]
                    objpoints[cam].append(objp)
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    imgpoints[cam].append(corners)
                    # Draw the corners and store the images
                    img = cv2.drawChessboardCorners(img, (cbcol, cbrow), corners, ret)
                    cv2.imwrite(
                        os.path.join(str(path_corners), filename + "_corner.jpg"), img
                    )
                else:
                    print("Corners not found for the image %s" % Path(fname).name)
                    for new_cam in cam_names:
                        remove_fname = Path(fname).name.replace(cam, new_cam)
                        os.rename(
                            os.path.join(str(img_path), remove_fname),
                            os.path.join(str(path_removed_images), remove_fname),
                        )
                        if new_cam != cam:
                            skip_images.append(remove_fname)

    try:
        h, w = img.shape[:2]
    except:
        raise Exception(
            "It seems that the name of calibration images does not match with the camera names in the config file. Please make sure that the calibration images are named with camera names as specified in the config.yaml file."
        )

    # Perform calibration for each cameras and store the matrices as a pickle file
    if calibrate == True:
        # Calibrating each camera
        for cam in cam_names:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints[cam], imgpoints[cam], img_shape[cam], None, None
            )

            # Save the camera calibration result for later use (we won't use rvecs / tvecs)
            dist_pickle[cam] = {
                "mtx": mtx,
                "dist": dist,
                "objpoints": objpoints[cam],
                "imgpoints": imgpoints[cam],
            }
            pickle.dump(
                dist_pickle,
                open(
                    os.path.join(path_camera_matrix, cam + "_intrinsic_params.pickle"),
                    "wb",
                ),
            )
            print(
                "Saving intrinsic camera calibration matrices for %s as a pickle file in %s"
                % (cam, os.path.join(path_camera_matrix))
            )

            # Compute mean re-projection errors for individual cameras
            mean_error = 0
            for i in range(len(objpoints[cam])):
                imgpoints_proj, _ = cv2.projectPoints(
                    objpoints[cam][i], rvecs[i], tvecs[i], mtx, dist
                )
                error = cv2.norm(imgpoints[cam][i], imgpoints_proj, cv2.NORM_L2) / len(
                    imgpoints_proj
                )
                mean_error += error
            print(
                "Mean re-projection error for %s images: %.3f pixels "
                % (cam, mean_error / len(objpoints[cam]))
            )

        # Compute stereo calibration for each pair of cameras
        camera_pair = [[cam_names[0], cam_names[1]]]
        for pair in camera_pair:
            print("Computing stereo calibration for " % pair)
            (
                retval,
                cameraMatrix1,
                distCoeffs1,
                cameraMatrix2,
                distCoeffs2,
                R,
                T,
                E,
                F,
            ) = cv2.stereoCalibrate(
                objpoints[pair[0]],
                imgpoints[pair[0]],
                imgpoints[pair[1]],
                dist_pickle[pair[0]]["mtx"],
                dist_pickle[pair[0]]["dist"],
                dist_pickle[pair[1]]["mtx"],
                dist_pickle[pair[1]]["dist"],
                (h, w),
                flags=cv2.CALIB_FIX_INTRINSIC,
            )

            # Stereo Rectification
            rectify_scale = alpha  # Free scaling parameter check this https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-stereorectify
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                cameraMatrix1,
                distCoeffs1,
                cameraMatrix2,
                distCoeffs2,
                (h, w),
                R,
                T,
                alpha=rectify_scale,
            )

            stereo_params[pair[0] + "-" + pair[1]] = {
                "cameraMatrix1": cameraMatrix1,
                "cameraMatrix2": cameraMatrix2,
                "distCoeffs1": distCoeffs1,
                "distCoeffs2": distCoeffs2,
                "R": R,
                "T": T,
                "E": E,
                "F": F,
                "R1": R1,
                "R2": R2,
                "P1": P1,
                "P2": P2,
                "roi1": roi1,
                "roi2": roi2,
                "Q": Q,
                "image_shape": [img_shape[pair[0]], img_shape[pair[1]]],
            }

        print(
            "Saving the stereo parameters for every pair of cameras as a pickle file in %s"
            % str(os.path.join(path_camera_matrix))
        )

        auxiliaryfunctions.write_pickle(
            os.path.join(path_camera_matrix, "stereo_params.pickle"), stereo_params
        )
        print(
            "Camera calibration done! Use the function ``check_undistortion`` to check the check the calibration"
        )
    else:
        print(
            "Corners extracted! You may check for the extracted corners in the directory %s and remove the pair of images where the corners are incorrectly detected. If all the corners are detected correctly with right order, then re-run the same function and use the flag ``calibrate=True``, to calbrate the camera."
            % str(path_corners)
        )


def check_undistortion(config, cbrow=8, cbcol=6, plot=True):
    """
    This function undistorts the calibration images based on the camera matrices and stores them in the project folder(defined in the config file)
    to visually check if the camera matrices are correct.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    cbrow : int
        Int specifying the number of rows in the calibration image.

    cbcol : int
        Int specifying the number of columns in the calibration image.

    plot : bool
        If this is set to True, the results of undistortion are saved as plots. The default is ``True``; if provided it must be either ``True`` or ``False``.

    Example
    --------
    Linux/MacOs/Windows
    >>> deeplabcut.check_undistortion(config, cbrow = 8,cbcol = 6)

    """

    # Read the config file
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cfg_3d = auxiliaryfunctions.read_config(config)
    (
        img_path,
        path_corners,
        path_camera_matrix,
        path_undistort,
        path_removed_images,
    ) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)

    # colormap = plt.get_cmap(cfg_3d['colormap'])
    markerSize = cfg_3d["dotsize"]
    alphaValue = cfg_3d["alphaValue"]
    markerType = cfg_3d["markerType"]
    markerColor = cfg_3d["markerColor"]
    cam_names = cfg_3d["camera_names"]

    images = glob.glob(os.path.join(img_path, "*.jpg"))

    # Sort the images
    images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    """
    for fname in images:
        for cam in cam_names:
            if cam in fname:
                filename = Path(fname).stem
                ext = Path(fname).suffix
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    """
    camera_pair = [[cam_names[0], cam_names[1]]]
    stereo_params = auxiliaryfunctions.read_pickle(
        os.path.join(path_camera_matrix, "stereo_params.pickle")
    )

    for pair in camera_pair:
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            stereo_params[pair[0] + "-" + pair[1]]["cameraMatrix1"],
            stereo_params[pair[0] + "-" + pair[1]]["distCoeffs1"],
            stereo_params[pair[0] + "-" + pair[1]]["R1"],
            stereo_params[pair[0] + "-" + pair[1]]["P1"],
            (stereo_params[pair[0] + "-" + pair[1]]["image_shape"][0]),
            cv2.CV_16SC2,
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            stereo_params[pair[0] + "-" + pair[1]]["cameraMatrix2"],
            stereo_params[pair[0] + "-" + pair[1]]["distCoeffs2"],
            stereo_params[pair[0] + "-" + pair[1]]["R2"],
            stereo_params[pair[0] + "-" + pair[1]]["P2"],
            (stereo_params[pair[0] + "-" + pair[1]]["image_shape"][1]),
            cv2.CV_16SC2,
        )
        cam1_undistort = []
        cam2_undistort = []

        for fname in images:
            if pair[0] in fname:
                filename = Path(fname).stem
                img1 = cv2.imread(fname)
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                h, w = img1.shape[:2]
                _, corners1 = cv2.findChessboardCorners(gray1, (cbcol, cbrow), None)
                corners_origin1 = cv2.cornerSubPix(
                    gray1, corners1, (11, 11), (-1, -1), criteria
                )

                # Remapping dataFrame_camera1_undistort
                im_remapped1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LANCZOS4)
                imgpoints_proj_undistort = cv2.undistortPoints(
                    src=corners_origin1,
                    cameraMatrix=stereo_params[pair[0] + "-" + pair[1]][
                        "cameraMatrix1"
                    ],
                    distCoeffs=stereo_params[pair[0] + "-" + pair[1]]["distCoeffs1"],
                    P=stereo_params[pair[0] + "-" + pair[1]]["P1"],
                    R=stereo_params[pair[0] + "-" + pair[1]]["R1"],
                )
                cam1_undistort.append(imgpoints_proj_undistort)
                cv2.imwrite(
                    os.path.join(str(path_undistort), filename + "_undistort.jpg"),
                    im_remapped1,
                )
                imgpoints_proj_undistort = []

            elif pair[1] in fname:
                filename = Path(fname).stem
                img2 = cv2.imread(fname)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                h, w = img2.shape[:2]
                _, corners2 = cv2.findChessboardCorners(gray2, (cbcol, cbrow), None)
                corners_origin2 = cv2.cornerSubPix(
                    gray2, corners2, (11, 11), (-1, -1), criteria
                )

                # Remapping
                im_remapped2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LANCZOS4)
                imgpoints_proj_undistort2 = cv2.undistortPoints(
                    src=corners_origin2,
                    cameraMatrix=stereo_params[pair[0] + "-" + pair[1]][
                        "cameraMatrix2"
                    ],
                    distCoeffs=stereo_params[pair[0] + "-" + pair[1]]["distCoeffs2"],
                    P=stereo_params[pair[0] + "-" + pair[1]]["P2"],
                    R=stereo_params[pair[0] + "-" + pair[1]]["R2"],
                )
                cam2_undistort.append(imgpoints_proj_undistort2)
                cv2.imwrite(
                    os.path.join(str(path_undistort), filename + "_undistort.jpg"),
                    im_remapped2,
                )
                imgpoints_proj_undistort2 = []

        cam1_undistort = np.array(cam1_undistort)
        cam2_undistort = np.array(cam2_undistort)
        print("All images are undistorted and stored in %s" % str(path_undistort))
        print(
            "Use the function ``triangulate`` to undistort the dataframes and compute the triangulation"
        )

        if plot == True:
            f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            f1.suptitle(
                str("Original Image: Views from " + pair[0] + " and " + pair[1]),
                fontsize=25,
            )

            # Display images in RGB
            ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

            norm = mcolors.Normalize(vmin=0.0, vmax=cam1_undistort.shape[1])
            plt.savefig(os.path.join(str(path_undistort), "Original_Image.png"))

            # Plot the undistorted corner points
            f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            f2.suptitle(
                "Undistorted corner points on camera-1 and camera-2", fontsize=25
            )
            ax1.imshow(cv2.cvtColor(im_remapped1, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(im_remapped2, cv2.COLOR_BGR2RGB))
            for i in range(0, cam1_undistort.shape[1]):
                ax1.scatter(
                    [cam1_undistort[-1][i, 0, 0]],
                    [cam1_undistort[-1][i, 0, 1]],
                    marker=markerType,
                    s=markerSize,
                    color=markerColor,
                    alpha=alphaValue,
                )
                ax2.scatter(
                    [cam2_undistort[-1][i, 0, 0]],
                    [cam2_undistort[-1][i, 0, 1]],
                    marker=markerType,
                    s=markerSize,
                    color=markerColor,
                    alpha=alphaValue,
                )
            plt.savefig(os.path.join(str(path_undistort), "undistorted_points.png"))

            # Triangulate
            triangulate = (
                auxiliaryfunctions_3d.compute_triangulation_calibration_images(
                    stereo_params[pair[0] + "-" + pair[1]],
                    cam1_undistort,
                    cam2_undistort,
                    path_undistort,
                    cfg_3d,
                    plot=True,
                )
            )
            auxiliaryfunctions.write_pickle("triangulate.pickle", triangulate)
