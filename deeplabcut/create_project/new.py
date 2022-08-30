"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
import shutil
import warnings
from pathlib import Path
from deeplabcut import DEBUG
from deeplabcut.utils.auxfun_videos import VideoReader


def create_new_project(
    project,
    experimenter,
    videos,
    working_directory=None,
    copy_videos=False,
    videotype="",
    multianimal=False,
):
    r"""Create the necessary folders and files for a new project.

    Creating a new project involves creating the project directory, sub-directories and
    a basic configuration file. The configuration file is loaded with the default
    values. Change its parameters to your projects need.

    Parameters
    ----------
    project : string
        The name of the project.

    experimenter : string
        The name of the experimenter.

    videos : list[str]
        A list of strings representing the full paths of the videos to include in the
        project. If the strings represent a directory instead of a file, all videos of
        ``videotype`` will be imported.

    working_directory : string, optional
        The directory where the project will be created. The default is the
        ``current working directory``.

    copy_videos : bool, optional, Default: False.
        If True, the videos are copied to the ``videos`` directory. If False, symlinks
        of the videos will be created in the ``project/videos`` directory; in the event
        of a failure to create symbolic links, videos will be moved instead.

    multianimal: bool, optional. Default: False.
        For creating a multi-animal project (introduced in DLC 2.2)

    Returns
    -------
    str
        Path to the new project configuration file.

    Examples
    --------

    Linux/MacOS:

    >>> deeplabcut.create_new_project(
            project='reaching-task',
            experimenter='Linus',
            videos=[
                '/data/videos/mouse1.avi',
                '/data/videos/mouse2.avi',
                '/data/videos/mouse3.avi'
            ],
            working_directory='/analysis/project/',
        )
    >>> deeplabcut.create_new_project(
            project='reaching-task',
            experimenter='Linus',
            videos=['/data/videos'],
            videotype='.mp4',
        )

    Windows:

    >>> deeplabcut.create_new_project(
            'reaching-task',
            'Bill',
            [r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'],
            copy_videos=True,
        )

    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    """
    from datetime import datetime as dt
    from deeplabcut.utils import auxiliaryfunctions

    months_3letter = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    date = dt.today()
    month = months_3letter[date.month]
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    if working_directory is None:
        working_directory = "."
    wd = Path(working_directory).resolve()
    project_name = "{pn}-{exp}-{date}".format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return  os.path.join(str(project_path), "config.yaml")
    video_path = project_path / "videos"
    data_path = project_path / "labeled-data"
    shuffles_path = project_path / "training-datasets"
    results_path = project_path / "dlc-models"
    for p in [video_path, data_path, shuffles_path, results_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # Add all videos in the folder. Multiple folders can be passed in a list, similar to the video files. Folders and video files can also be passed!
    vids = []
    for i in videos:
        # Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [
                os.path.join(i, vp) for vp in os.listdir(i) if vp.endswith(videotype)
            ]
            vids = vids + vids_in_dir
            if len(vids_in_dir) == 0:
                print("No videos found in", i)
                print(
                    "Perhaps change the videotype, which is currently set to:",
                    videotype,
                )
            else:
                videos = vids
                print(
                    len(vids_in_dir),
                    " videos from the directory",
                    i,
                    "were added to the project.",
                )
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids

    videos = [Path(vp) for vp in videos]
    dirs = [data_path / Path(i.stem) for i in videos]
    for p in dirs:
        """
        Creates directory under data
        """
        p.mkdir(parents=True, exist_ok=True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_videos:
        print("Copying the videos")
        for src, dst in zip(videos, destinations):
            shutil.copy(
                os.fspath(src), os.fspath(dst)
            )  # https://www.python.org/dev/peps/pep-0519/
    else:
        # creates the symlinks of the video and puts it in the videos directory.
        print("Attempting to create a symbolic link of the video ...")
        for src, dst in zip(videos, destinations):
            if dst.exists() and not DEBUG:
                raise FileExistsError("Video {} exists already!".format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
                print("Created the symlink of {} to {}".format(src, dst))
            except OSError:
                try:
                    import subprocess

                    subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
                except (OSError, subprocess.CalledProcessError):
                    print(
                        "Symlink creation impossible (exFat architecture?): "
                        "cutting/pasting the video instead."
                    )
                    shutil.move(os.fspath(src), os.fspath(dst))
                    print("{} moved to {}".format(src, dst))
            videos = destinations

    if copy_videos:
        videos = destinations  # in this case the *new* location should be added to the config file

    # adds the video list to the config.yaml file
    video_sets = {}
    for video in videos:
        print(video)
        try:
            # For windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path = os.path.realpath(video)]
            rel_video_path = str(Path.resolve(Path(video)))
        except:
            rel_video_path = os.readlink(str(video))

        try:
            vid = VideoReader(rel_video_path)
            video_sets[rel_video_path] = {"crop": ", ".join(map(str, vid.get_bbox()))}
        except IOError:
            warnings.warn("Cannot open the video file! Skipping to the next one...")
            os.remove(video)  # Removing the video or link from the project

    if not len(video_sets):
        # Silently sweep the files that were already written.
        shutil.rmtree(project_path, ignore_errors=True)
        warnings.warn(
            "No valid videos were found. The project was not created... "
            "Verify the video files and re-create the project."
        )
        return "nothingcreated"

    # Set values to config file:
    if multianimal:  # parameters specific to multianimal project
        cfg_file, ruamelFile = auxiliaryfunctions.create_config_template(multianimal)
        cfg_file["multianimalproject"] = multianimal
        cfg_file["identity"] = False
        cfg_file["individuals"] = ["individual1", "individual2", "individual3"]
        cfg_file["multianimalbodyparts"] = ["bodypart1", "bodypart2", "bodypart3"]
        cfg_file["uniquebodyparts"] = []
        cfg_file["bodyparts"] = "MULTI!"
        cfg_file["skeleton"] = [
            ["bodypart1", "bodypart2"],
            ["bodypart2", "bodypart3"],
            ["bodypart1", "bodypart3"],
        ]
        cfg_file["default_augmenter"] = "multi-animal-imgaug"
        cfg_file["default_net_type"] = "dlcrnet_ms5"
        cfg_file["default_track_method"] = "ellipse"
    else:
        cfg_file, ruamelFile = auxiliaryfunctions.create_config_template()
        cfg_file["multianimalproject"] = False
        cfg_file["bodyparts"] = ["bodypart1", "bodypart2", "bodypart3", "objectA"]
        cfg_file["skeleton"] = [["bodypart1", "bodypart2"], ["objectA", "bodypart3"]]
        cfg_file["default_augmenter"] = "default"
        cfg_file["default_net_type"] = "resnet_50"

    # common parameters:
    cfg_file["Task"] = project
    cfg_file["scorer"] = experimenter
    cfg_file["video_sets"] = video_sets
    cfg_file["project_path"] = str(project_path)
    cfg_file["date"] = d
    cfg_file["cropping"] = False
    cfg_file["start"] = 0
    cfg_file["stop"] = 1
    cfg_file["numframes2pick"] = 20
    cfg_file["TrainingFraction"] = [0.95]
    cfg_file["iteration"] = 0
    cfg_file["snapshotindex"] = -1
    cfg_file["x1"] = 0
    cfg_file["x2"] = 640
    cfg_file["y1"] = 277
    cfg_file["y2"] = 624
    cfg_file[
        "batch_size"
    ] = 8  # batch size during inference (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242
    cfg_file["corner2move2"] = (50, 50)
    cfg_file["move2corner"] = True
    cfg_file["skeleton_color"] = "black"
    cfg_file["pcutoff"] = 0.6
    cfg_file["dotsize"] = 12  # for plots size of dots
    cfg_file["alphavalue"] = 0.7  # for plots transparency of markers
    cfg_file["colormap"] = "rainbow"  # for plots type of colormap

    projconfigfile = os.path.join(str(project_path), "config.yaml")
    # Write dictionary to yaml  config file
    auxiliaryfunctions.write_config(projconfigfile, cfg_file)

    print('Generated "{}"'.format(project_path / "config.yaml"))
    print(
        "\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. Change the parameters in this file to adapt to your project's needs.\n Once you have changed the configuration file, use the function 'extract_frames' to select frames for labeling.\n. [OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)."
        % (project_name, str(wd))
    )
    return projconfigfile
