"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


def select_cropping_area(config, videos=None):
    """
    Interactively select the cropping area of all videos in the config.
    A user interface pops up with a frame to select the cropping parameters.
    Use the left click to draw a box and hit the button 'set cropping parameters'
    to store the cropping parameters for a video in the config.yaml file.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : optional (default=None)
        List of videos whose cropping areas are to be defined. Note that full paths are required.
        By default, all videos in the config are successively loaded.

    Returns
    -------
    cfg : dict
        Updated project configuration
    """
    from deeplabcut.utils import auxiliaryfunctions, auxfun_videos

    cfg = auxiliaryfunctions.read_config(config)
    if videos is None:
        videos = list(cfg.get("video_sets_original") or cfg["video_sets"])

    for video in videos:
        coords = auxfun_videos.draw_bbox(video)
        if coords:
            temp = {
                "crop": ", ".join(
                    map(
                        str,
                        [
                            int(coords[0]),
                            int(coords[2]),
                            int(coords[1]),
                            int(coords[3]),
                        ],
                    )
                )
            }
            try:
                cfg["video_sets"][video] = temp
            except KeyError:
                cfg["video_sets_original"][video] = temp

    auxiliaryfunctions.write_config(config, cfg)
    return cfg


def extract_frames(
    config,
    mode="automatic",
    algo="kmeans",
    crop=False,
    userfeedback=True,
    cluster_step=1,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=True,
    slider_width=25,
    config3d=None,
    extracted_cam=0,
):
    """
    Extracts frames from the videos in the config.yaml file. Only the videos in the config.yaml will be used to select the frames.\n
    Use the function ``add_new_video`` at any stage of the project to add new videos to the config file and extract their frames.

    The provided function either selects frames from the videos in a randomly and temporally uniformly distributed way (uniform), \n
    by clustering based on visual appearance (k-means), or by manual selection.

    Three important parameters for automatic extraction: numframes2pick, start and stop are set in the config file.

    After frames have been extracted from all videos from one camera, matched frames from other cameras can be extracted using mode = ``match``.
    This is necessary if you plan to use epipolar lines to improve labeling across multiple camera angles. It will overwrite previously extracted
    images from the second camera angle if necessary.

    Please refer to the user guide for more details on methods and parameters https://www.nature.com/articles/s41596-019-0176-0
    or the preprint: https://www.biorxiv.org/content/biorxiv/early/2018/11/24/476531.full.pdf

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    mode : string
        String containing the mode of extraction. It must be either ``automatic`` or ``manual`` to extract the inital set of frames. It can also be ``match`` to match frames between
        the cameras in preparation for the use of epipolar lines during labeling; namely, extract from camera_1 first, then run this to extact the matched frames in camera_2.
        WARNING: if you use match, and you previously extracted and labeled frames from the second camera, this will overwrite your data. This will require you deleting the
        collectdata.h5/.csv files before labeling.... Use with caution!

    algo : string
        String specifying the algorithm to use for selecting the frames. Currently, deeplabcut supports either ``kmeans`` or ``uniform`` based selection. This flag is
        only required for ``automatic`` mode and the default is ``uniform``. For uniform, frames are picked in temporally uniform way, kmeans performs clustering on downsampled frames (see user guide for details).
        Note: color information is discarded for kmeans, thus e.g. for camouflaged octopus clustering one might want to change this.

    crop : bool, optional
        If True, video frames are cropped according to the corresponding coordinates stored in the config.yaml.
        Alternatively, if cropping coordinates are not known yet, crop='GUI' triggers a user interface
        where the cropping area can be manually drawn and saved.

    userfeedback: bool, optional
        If this is set to false during automatic mode then frames for all videos are extracted. The user can set this to true, which will result in a dialog,
        where the user is asked for each video if (additional/any) frames from this video should be extracted. Use this, e.g. if you have already labeled
        some folders and want to extract data for new videos.

    cluster_resizewidth: number, default: 30
        For k-means one can change the width to which the images are downsampled (aspect ratio is fixed).

    cluster_step: number, default: 1
        By default each frame is used for clustering, but for long videos one could only use every nth frame (set by: cluster_step). This saves memory before clustering can start, however,
        reading the individual frames takes longer due to the skipping.

    cluster_color: bool, default: False
        If false then each downsampled image is treated as a grayscale vector (discarding color information). If true, then the color channels are considered. This increases
        the computational complexity.

    opencv: bool, default: True
        Uses openCV for loading & extractiong (otherwise moviepy (legacy))

    slider_width: number, default: 25
        Width of the video frames slider, in percent of window

    config3d: string, optional
        Path to the config.yaml file in the 3D project. This will be used to match frames extracted from all cameras present in the field 'camera_names' to the
        frames extracted from the camera given by the parameter 'extracted_cam'

    extracted_cam: number, default: 0
        The index of the camera that already has extracted frames. This will match frame numbers to extract for all other cameras.
        This parameter is necessary if you wish to use epipolar lines in the labeling toolbox. Only use if mode = 'match' and config3d is provided.

    Examples
    --------
    for selecting frames automatically with 'kmeans' and want to crop the frames.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans',True)
    --------
    for selecting frames automatically with 'kmeans' and defining the cropping area at runtime.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans','GUI')
    --------
    for selecting frames automatically with 'kmeans' and considering the color information.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans',cluster_color=True)
    --------
    for selecting frames automatically with 'uniform' and want to crop the frames.
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic',crop=True)
    --------
    for selecting frames manually,
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','manual')
    --------
    for selecting frames manually, with a 60% wide frames slider
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','manual', slider_width=60)
    --------
    for extracting frames from a second camera that match the frames extracted from the first
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml', mode='match', extracted_cam=0)

    While selecting the frames manually, you do not need to specify the ``crop`` parameter in the command. Rather, you will get a prompt in the graphic user interface to choose
    if you need to crop or not.
    --------

    """
    import os
    import sys
    import re
    import glob
    import numpy as np
    from pathlib import Path
    from skimage import io
    from skimage.util import img_as_ubyte
    from deeplabcut.utils import frameselectiontools
    from deeplabcut.utils import auxiliaryfunctions

    if mode == "manual":
        wd = Path(config).resolve().parents[0]
        os.chdir(str(wd))
        from deeplabcut.gui import frame_extraction_toolbox

        frame_extraction_toolbox.show(config, slider_width)

    elif mode == "automatic":
        config_file = Path(config).resolve()
        cfg = auxiliaryfunctions.read_config(config_file)
        print("Config file read successfully.")

        numframes2pick = cfg["numframes2pick"]
        start = cfg["start"]
        stop = cfg["stop"]

        # Check for variable correctness
        if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
            raise Exception(
                "Erroneous start or stop values. Please correct it in the config file."
            )
        if numframes2pick < 1 and not int(numframes2pick):
            raise Exception(
                "Perhaps consider extracting more, or a natural number of frames."
            )

        videos = cfg.get("video_sets_original") or cfg["video_sets"]
        if opencv:
            from deeplabcut.utils.auxfun_videos import VideoReader
        else:
            from moviepy.editor import VideoFileClip

        has_failed = []
        for video in videos:
            if userfeedback:
                print(
                    "Do you want to extract (perhaps additional) frames for video:",
                    video,
                    "?",
                )
                askuser = input("yes/no")
            else:
                askuser = "yes"

            if (
                askuser == "y"
                or askuser == "yes"
                or askuser == "Ja"
                or askuser == "ha"
                or askuser == "oui"
                or askuser == "ouais"
            ):  # multilanguage support :)

                if opencv:
                    cap = VideoReader(video)
                    nframes = len(cap)
                else:
                    # Moviepy:
                    clip = VideoFileClip(video)
                    fps = clip.fps
                    nframes = int(np.ceil(clip.duration * 1.0 / fps))
                if not nframes:
                    print("Video could not be opened. Skipping...")
                    continue

                indexlength = int(np.ceil(np.log10(nframes)))

                fname = Path(video)
                output_path = Path(config).parents[0] / "labeled-data" / fname.stem

                if output_path.exists():
                    if len(os.listdir(output_path)):
                        if userfeedback:
                            askuser = input(
                                "The directory already contains some frames. Do you want to add to it?(yes/no): "
                            )
                        if not (
                            askuser == "y"
                            or askuser == "yes"
                            or askuser == "Y"
                            or askuser == "Yes"
                        ):
                            sys.exit("Delete the frames and try again later!")

                if crop == "GUI":
                    cfg = select_cropping_area(config, [video])
                try:
                    coords = cfg["video_sets"][video]["crop"].split(",")
                except KeyError:
                    coords = cfg["video_sets_original"][video]["crop"].split(",")

                if crop and not opencv:
                    clip = clip.crop(
                        y1=int(coords[2]),
                        y2=int(coords[3]),
                        x1=int(coords[0]),
                        x2=int(coords[1]),
                    )
                elif not crop:
                    coords = None

                print("Extracting frames based on %s ..." % algo)
                if algo == "uniform":
                    if opencv:
                        frames2pick = frameselectiontools.UniformFramescv2(
                            cap, numframes2pick, start, stop
                        )
                    else:
                        frames2pick = frameselectiontools.UniformFrames(
                            clip, numframes2pick, start, stop
                        )
                elif algo == "kmeans":
                    if opencv:
                        frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
                            cap,
                            numframes2pick,
                            start,
                            stop,
                            crop,
                            coords,
                            step=cluster_step,
                            resizewidth=cluster_resizewidth,
                            color=cluster_color,
                        )
                    else:
                        frames2pick = frameselectiontools.KmeansbasedFrameselection(
                            clip,
                            numframes2pick,
                            start,
                            stop,
                            step=cluster_step,
                            resizewidth=cluster_resizewidth,
                            color=cluster_color,
                        )
                else:
                    print(
                        "Please implement this method yourself and send us a pull request! Otherwise, choose 'uniform' or 'kmeans'."
                    )
                    frames2pick = []

                if not len(frames2pick):
                    print("Frame selection failed...")
                    return

                output_path = (
                    Path(config).parents[0] / "labeled-data" / Path(video).stem
                )
                is_valid = []
                if opencv:
                    for index in frames2pick:
                        cap.set_to_frame(index)  # extract a particular frame
                        frame = cap.read_frame()
                        if frame is not None:
                            image = img_as_ubyte(frame)
                            img_name = (
                                str(output_path)
                                + "/img"
                                + str(index).zfill(indexlength)
                                + ".png"
                            )
                            if crop:
                                io.imsave(
                                    img_name,
                                    image[
                                        int(coords[2]) : int(coords[3]),
                                        int(coords[0]) : int(coords[1]),
                                        :,
                                    ],
                                )  # y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]
                            else:
                                io.imsave(img_name, image)
                            is_valid.append(True)
                        else:
                            print("Frame", index, " not found!")
                            is_valid.append(False)
                    cap.close()
                else:
                    for index in frames2pick:
                        try:
                            image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                            img_name = (
                                str(output_path)
                                + "/img"
                                + str(index).zfill(indexlength)
                                + ".png"
                            )
                            io.imsave(img_name, image)
                            if np.var(image) == 0:  # constant image
                                print(
                                    "Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
                                )
                            is_valid.append(True)
                        except FileNotFoundError:
                            print("Frame # ", index, " does not exist.")
                            is_valid.append(False)
                    clip.close()
                    del clip

                if not any(is_valid):
                    has_failed.append(True)
                else:
                    has_failed.append(False)

            else:  # NO!
                has_failed.append(False)

        if all(has_failed):
            print("Frame extraction failed. Video files must be corrupted.")
            return
        elif any(has_failed):
            print("Although most frames were extracted, some were invalid.")
        else:
            print("Frames were successfully extracted, for the videos of interest.")
        print(
            "\nYou can now label the frames using the function 'label_frames' "
            "(if you extracted enough frames for all videos)."
        )

    elif mode == "match":
        import cv2

        config_file = Path(config).resolve()
        cfg = auxiliaryfunctions.read_config(config_file)
        print("Config file read successfully.")
        videos = sorted(cfg["video_sets"].keys())
        project_path = Path(config).parents[0]
        labels_path = os.path.join(project_path, "labeled-data/")
        video_dir = os.path.join(project_path, "videos/")
        try:
            cfg_3d = auxiliaryfunctions.read_config(config3d)
        except:
            raise Exception(
                "You must create a 3D project and edit the 3D config file before extracting matched frames. \n"
            )
        cams = cfg_3d["camera_names"]
        extCam_name = cams[extracted_cam]
        del cams[extracted_cam]
        label_dirs = sorted(
            glob.glob(os.path.join(labels_path, "*" + extCam_name + "*"))
        )

        # select crop method
        crop_list = []
        for video in videos:
            if extCam_name not in video:
                if crop == "GUI":
                    cfg = select_cropping_area(config, [video])
                    print("in gui code")
                coords = cfg["video_sets"][video]["crop"].split(",")

                if crop and not opencv:
                    clip = clip.crop(
                        y1=int(coords[2]),
                        y2=int(coords[3]),
                        x1=int(coords[0]),
                        x2=int(coords[1]),
                    )
                elif not crop:
                    coords = None
                crop_list.append(coords)
        print(crop_list)

        for coords, dirPath in zip(crop_list, label_dirs):
            extracted_images = glob.glob(os.path.join(dirPath, "*png"))

            imgPattern = re.compile("[0-9]{1,10}")
            for cam in cams:
                output_path = re.sub(extCam_name, cam, dirPath)

                for fname in os.listdir(output_path):
                    if fname.endswith(".png"):
                        os.remove(os.path.join(output_path, fname))

                vid = os.path.join(video_dir, os.path.basename(output_path)) + ".avi"
                cap = cv2.VideoCapture(vid)
                print(
                    "\n extracting matched frames from "
                    + os.path.basename(output_path)
                    + ".avi"
                )
                for img in extracted_images:
                    imgNum = re.findall(imgPattern, os.path.basename(img))[0]
                    cap.set(1, int(imgNum))
                    ret, frame = cap.read()
                    if ret:
                        image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_name = str(output_path) + "/img" + imgNum + ".png"
                        if crop:
                            io.imsave(
                                img_name,
                                image[
                                    int(coords[2]) : int(coords[3]),
                                    int(coords[0]) : int(coords[1]),
                                    :,
                                ],
                            )
                        else:
                            io.imsave(img_name, image)
        print(
            "\n Done extracting matched frames. You can now begin labeling frames using the function label_frames\n"
        )

    else:
        print(
            "Invalid MODE. Choose either 'manual', 'automatic' or 'match'. Check ``help(deeplabcut.extract_frames)`` on python and ``deeplabcut.extract_frames?`` \
              for ipython/jupyter notebook for more details."
        )
