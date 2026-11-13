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


from pathlib import Path

from deeplabcut import DEBUG


def create_new_project_3d(project, experimenter, num_cameras=2, working_directory=None):
    r"""Creates a new project directory, sub-directories and a basic configuration file
    for 3d project. The configuration file is loaded with the default values. Adjust the
    parameters to your project's needs.

    Args:
        project (string): String containing the name of the project.
        experimenter (string): String containing the name of the experimenter.
        num_cameras (int): An integer value specifying the number of cameras.
        working_directory (string, optional): The directory where the project will be
            created. The default is the ``current working directory``; if provided, it
            must be a string.

    Examples:
        Linux/MacOs:

            deeplabcut.create_new_project_3d("reaching-task", "Linus", 2)

        Windows:

            deeplabcut.create_new_project_3d("reaching-task", "Bill", 2)

        On Windows, paths should be formatted as ``r`"C:\"`` or ``"C:\\"`` (i.e. a double backslash).
    """
    from datetime import datetime as dt

    from deeplabcut.core.config import CameraConfig3D, ProjectConfig3D

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")

    if working_directory is None:
        working_directory = "."

    wd = Path(working_directory).absolute()
    project_name = "{pn}-{exp}-{date}-{triangulate}".format(pn=project, exp=experimenter, date=date, triangulate="3d")
    project_path = wd / project_name
    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print(f'Project "{project_path}" already exists!')
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
        print(f'Created "{p}"')

    # Create config file
    cfg_3d = ProjectConfig3D(
        Task=project,
        scorer=experimenter,
        date=d,
        project_path=project_path,
        skeleton=[
            ["bodypart1", "bodypart2"],
            ["bodypart2", "bodypart3"],
            ["bodypart3", "bodypart4"],
            ["bodypart4", "bodypart5"],
        ],
        num_cameras=num_cameras,
        # The 2D project of each camera (config_file, shuffle, trainingsetindex) is filled in by the user
        cameras={f"camera-{i}": CameraConfig3D() for i in range(1, num_cameras + 1)},
    )
    projconfigfile = project_path / "config.yaml"
    cfg_3d.to_yaml(projconfigfile)

    print('Generated "{}"'.format(project_path / "config.yaml"))
    print(
        f"\nA new project with name {project_name} is created at {wd} and a configurable file (config.yaml) is stored"
        f"there. If you have not calibrated the cameras, then use the function 'calibrate_camera' to start calibrating"
        f"the camera otherwise use the function ``triangulate`` to triangulate the dataframe"
    )
    return projconfigfile
