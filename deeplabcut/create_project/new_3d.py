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

from deeplabcut import DEBUG


def create_new_project_3d(project, experimenter, num_cameras=2, working_directory=None):
    r"""Creates a new project directory, sub-directories and a basic configuration file for 3d project.
    The configuration file is loaded with the default values. Adjust the parameters to your project's needs.

    Parameters
    ----------
    project : string
        String containing the name of the project.

    experimenter : string
        String containing the name of the experimenter.

    num_cameras : int
        An integer value specifying the number of cameras.

    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string.


    Example
    --------
    Linux/MacOs
    >>> deeplabcut.create_new_project_3d('reaching-task','Linus',2)

    Windows:
    >>> deeplabcut.create_new_project('reaching-task','Bill',2)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \\ )

    """
    from datetime import datetime as dt
    from deeplabcut.utils import auxiliaryfunctions

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")

    if working_directory is None:
        working_directory = "."

    wd = Path(working_directory).resolve()
    project_name = "{pn}-{exp}-{date}-{triangulate}".format(
        pn=project, exp=experimenter, date=date, triangulate="3d"
    )
    project_path = wd / project_name
    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return

    camera_matrix_path = project_path / "camera_matrix"
    calibration_images_path = project_path / "calibration_images"
    undistortion_path = project_path / "undistortion"
    path_corners = project_path / "corners"
    path_removed_images = project_path / "removed_calibration_images"

    for p in [
        camera_matrix_path,
        calibration_images_path,
        undistortion_path,
        path_corners,
        path_removed_images,
    ]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # Create config file
    cfg_file_3d, ruamelFile_3d = auxiliaryfunctions.create_config_template_3d()
    cfg_file_3d["Task"] = project
    cfg_file_3d["scorer"] = experimenter
    cfg_file_3d["date"] = d
    cfg_file_3d["project_path"] = str(project_path)
    #    cfg_file_3d['config_files']= [str('Enter the path of the config file ')+str(i)+ ' to include' for i in range(1,3)]
    #    cfg_file_3d['config_files']= ['Enter the path of the config file 1']
    cfg_file_3d["colormap"] = "jet"
    cfg_file_3d["dotsize"] = 15
    cfg_file_3d["alphaValue"] = 0.8
    cfg_file_3d["markerType"] = "*"
    cfg_file_3d["markerColor"] = "r"
    cfg_file_3d["pcutoff"] = 0.4
    cfg_file_3d["num_cameras"] = num_cameras
    cfg_file_3d["camera_names"] = [
        str("camera-" + str(i)) for i in range(1, num_cameras + 1)
    ]
    cfg_file_3d["scorername_3d"] = "DLC_3D"

    cfg_file_3d["skeleton"] = [
        ["bodypart1", "bodypart2"],
        ["bodypart2", "bodypart3"],
        ["bodypart3", "bodypart4"],
        ["bodypart4", "bodypart5"],
    ]
    cfg_file_3d["skeleton_color"] = "black"

    for i in range(num_cameras):
        path = str(
            "/home/mackenzie/DEEPLABCUT/DeepLabCut/2DprojectCam"
            + str(i + 1)
            + "-Mackenzie-2019-06-05/config.yaml"
        )
        cfg_file_3d.insert(
            len(cfg_file_3d), str("config_file_camera-" + str(i + 1)), path
        )

    for i in range(num_cameras):
        cfg_file_3d.insert(len(cfg_file_3d), str("shuffle_camera-" + str(i + 1)), 1)
        cfg_file_3d.insert(
            len(cfg_file_3d), str("trainingsetindex_camera-" + str(i + 1)), 0
        )

    projconfigfile = os.path.join(str(project_path), "config.yaml")
    auxiliaryfunctions.write_config_3d(projconfigfile, cfg_file_3d)

    print('Generated "{}"'.format(project_path / "config.yaml"))
    print(
        "\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. If you have not calibrated the cameras, then use the function 'calibrate_camera' to start calibrating the camera otherwise use the function ``triangulate`` to triangulate the dataframe"
        % (project_name, wd)
    )
    return projconfigfile
