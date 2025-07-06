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

This script tests various functionalities in an automatic way.
It produces nothing of interest scientifically.
"""
import os, deeplabcut
import zipfile, urllib.request, shutil
from datetime import datetime as dt
import glob
from pathlib import Path
import subprocess


if __name__ == "__main__":
    print("Imported DLC!")
    task = "TEST3D"  # Enter the name of your experiment Task
    scorer = "Alex"  # Enter the name of the experimenter/labeler
    num_cameras = 2  # Enter the number of cameras

    basepath = str(Path(os.path.realpath(__file__)).parents[0])
    videoname = "reachingvideo1"
    video = [
        os.path.join(
            basepath,
            "Reaching-Mackenzie-2018-08-30",
            "videos",
            videoname + ".avi",
        )
    ]

    folder = os.path.join(basepath, "3Dtestviews_videos")
    deeplabcut.auxiliaryfunctions.attempt_to_make_folder(folder)

    # copying demo video from reaching data set and create two "views":
    dst_videoname1 = "vid1_camera-1"
    dst_videoname2 = "vid1_camera-2"
    dst_videoname3 = "long_camera-2"
    output1 = os.path.join(folder, dst_videoname1 + ".avi")
    output2 = os.path.join(folder, dst_videoname2 + ".avi")
    output3 = os.path.join(folder, dst_videoname3 + ".avi")
    shutil.copyfile(video[0], output3)

    vname = "brief"
    try:  # you need ffmpeg command line interface
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                video[0],
                "-ss",
                "00:00:00",
                "-to",
                "00:00:00.4",
                "-c",
                "copy",
                output1,
            ]
        )
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                video[0],
                "-ss",
                "00:00:00",
                "-to",
                "00:00:00.4",
                "-c",
                "copy",
                output2,
            ]
        )
    except:
        pass

    """
    # copying demo video from reaching data set and create two "views":
    dst_videoname1 = 'vid1_camera-1'
    dst_videoname2 = 'vid1_camera-2'
    output1 = os.path.join(basepath,folder,dst_videoname1+'.avi')
    output2 = os.path.join(basepath,folder,dst_videoname2+'.avi')
    shutil.copyfile(video[0], output1)
    shutil.copyfile(video[0], output2)
    """
    # checking if 2d test project is available
    try:
        config = glob.glob(os.path.join(basepath, "TEST*", "config.yaml"))[-1]
    except:
        raise RuntimeError("Please run the testscript.py first before testing for 3d")

    dfolder = None

    print("CREATING 3-D PROJECT")
    path_config_file = deeplabcut.create_new_project_3d(task, scorer, num_cameras)

    try:
        cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
        cfg["config_file_camera-1"] = config
        cfg["shuffle_camera-1"] = 1

        cfg["config_file_camera-2"] = config
        cfg["shuffle_camera-2"] = 2

        cfg["skeleton"] = [["bodypart1", "bodypart2"], ["objectA", "bodypart3"]]
        deeplabcut.auxiliaryfunctions.write_config_3d(path_config_file, cfg)
    except:
        raise (
            "Please delete the project and re-try."
        )  # otherwise the cfg is an empty array!

    """
    # Creating the name of the project
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    project_name = '{pn}-{exp}-{date}-{triangulate}'.format(pn=task, exp=scorer, date=date,triangulate='3d')
    """
    project_name = path_config_file.split(os.sep)[-2]

    os.chdir(os.path.join(project_name, "calibration_images"))

    file_name = os.path.join(basepath,"stereo_example.zip")
    with zipfile.ZipFile(file_name) as zf:
        zf.extractall()

    # Deleting unnecessary images; the ones whose corners are not detected and .mat files
    cwd = os.getcwd()
    [os.remove(file) for file in os.listdir(cwd) if not file.endswith(".jpg")]

    # change the file names for calibration images to match the name of cameras in config.yaml file.i.e. camera-1 and camera-2
    cam1_images = glob.glob(os.path.join(cwd, "left*.jpg"))
    cam2_images = glob.glob(os.path.join(cwd, "right*.jpg"))
    # Sorting images
    cam1_images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    cam2_images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    for idx, name in enumerate(cam1_images):
        os.rename(
            name,
            os.path.join(cwd, str("camera-1_" + "{0:0=2d}".format(idx + 1) + ".jpg")),
        )

    for idx, name in enumerate(cam2_images):
        os.rename(
            name,
            os.path.join(cwd, str("camera-2_" + "{0:0=2d}".format(idx + 1) + ".jpg")),
        )

    # Removing some of the images where the corner was not detected
    [os.remove(file) for file in glob.glob(os.path.join(cwd, "*06.jpg"))]
    [os.remove(file) for file in glob.glob(os.path.join(cwd, "*01.jpg"))]

    print("CALIBRATING THE CAMERAS")
    deeplabcut.calibrate_cameras(path_config_file, calibrate=True)

    print("CHECKING FOR UNDISTORTION")
    deeplabcut.check_undistortion(path_config_file)

    print("TRIANGULATING")
    video_dir = os.path.join(os.path.dirname(basepath), folder)
    deeplabcut.auxiliaryfunctions.edit_config(
        path_config_file, edits={"pcutoff": 0.1}
    )  # otherwise get all-nan slices
    deeplabcut.triangulate(path_config_file, video_dir, save_as_csv=True)

    print("CREATING LABELED VIDEO 3-D")
    deeplabcut.create_labeled_video_3d(path_config_file, [video_dir], start=5, end=10, videotype=".avi")

    # output_path = [os.path.join(basepath,folder)]
    # deeplabcut.create_labeled_video_3d(path_config_file,output_path,start=5,end=10)

    print("ALL DONE!!! - default 3D cases are functional.")
