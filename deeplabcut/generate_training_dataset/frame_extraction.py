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

from collections.abc import Collection, Iterable
from pathlib import Path


def _filter_config_videos(
    configured_videos: Iterable[str | Path],
    selected_videos: Collection[str | Path] | None,
) -> list[str | Path]:
    """Return config video keys matching the selected video paths.

    The original config keys are returned so they remain valid for subsequent
    config dictionary lookups.
    """
    configured_videos = list(configured_videos)

    if selected_videos is None:
        return configured_videos

    selected = set(selected_videos)
    return [video for video in configured_videos if Path(video) in selected]


def select_cropping_area(config: str | Path, videos=None):
    """Interactively select the cropping area of all videos in the config. A user
    interface pops up with a frame to select the cropping parameters. Use the left click
    to draw a box and hit the button 'set cropping parameters' to store the cropping
    parameters for a video in the config.yaml file.

    Args:
        config (string): Full path of the config.yaml file as a string.
        videos (list, optional): List of videos whose cropping areas are to be defined.
            Full paths are required. By default, all videos in the config are loaded.
            Defaults to None.

    Returns:
        dict: Updated project configuration
    """
    from deeplabcut.utils import auxfun_videos, auxiliaryfunctions

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
    config: str | Path,
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
    videos_list: list[str | Path] | None = None,
):
    """Extract frames from videos in a DeepLabCut project.

    Videos are read from the project's ``config.yaml`` file. When
    ``videos_list`` is provided, only matching configured videos are processed.

    In ``"automatic"`` mode, frames are selected either at approximately
    uniform temporal intervals with ``algo="uniform"`` or by clustering
    downsampled frames by visual appearance with ``algo="kmeans"``. In
    ``"manual"`` mode, the first selected video is opened in the napari plugin for
    interactive frame selection and cropping.

    In ``"match"`` mode, frame numbers already extracted for one camera are
    used to extract corresponding frames from the other cameras in a 3D
    project. Existing PNG frames in the output directories for those cameras
    may be removed and replaced.

    Args:
        config: Path to the project ``config.yaml`` file.
        mode: Frame-extraction mode. Supported values are:

            * ``"automatic"``: Select and extract frames automatically.
            * ``"manual"``: Open the interactive frame-selection interface.
            * ``"match"``: Extract corresponding frames from the other
              cameras in a 3D project.

            Defaults to ``"automatic"``.
        algo: Selection algorithm. ``"uniform"`` selects
            frames at approximately uniform temporal intervals, while
            ``"kmeans"`` clusters downsampled frames by visual appearance.
            This parameter is only used in ``"automatic"`` mode. Defaults to
            ``"kmeans"``.
        crop: Cropping behavior. If ``True``, frames are cropped using the
            coordinates stored in the project configuration. If ``"GUI"``,
            an interface is opened to select and save cropping coordinates
            before extraction. If ``False``, frames are not cropped. Defaults
            to ``False``.
        userfeedback: Whether to ask before extracting frames from each video
            in ``"automatic"`` mode. If ``False``, all selected videos are
            processed without this prompt. Defaults to ``True``.
        cluster_step: Use every nth frame as input to k-means clustering.
            Increasing this value can reduce the number of frames held for
            clustering. This parameter is only used when
            ``mode="automatic"`` and ``algo="kmeans"``. Defaults to ``1``.
        cluster_resizewidth: Width, in pixels, to which frames are resized
            before k-means clustering. The aspect ratio is preserved. This
            parameter is only used when ``mode="automatic"`` and
            ``algo="kmeans"``. Defaults to ``30``.
        cluster_color: Whether k-means clustering uses color information. If
            ``False``, each downsampled frame is treated as a grayscale
            vector. If ``True``, its color channels are retained, increasing
            the computational cost. This parameter is only used when
            ``mode="automatic"`` and ``algo="kmeans"``. Defaults to ``False``.
        opencv: Whether to use OpenCV-based video loading and frame extraction.
            If ``False``, the legacy MoviePy implementation is used for
            automatic extraction. Defaults to ``True``.
        slider_width: Width of the frame-selection slider as a percentage of
            the window width. This parameter is used in ``"manual"`` mode.
            Defaults to ``25``.
        config3d: Path to the configuration file of the associated 3D project.
            Required in ``"match"`` mode to identify the project cameras.
            Defaults to ``None``.
        extracted_cam: Index in the 3D project's ``camera_names`` list of the
            camera for which frames have already been extracted. The
            corresponding frame numbers are extracted for the remaining
            cameras in ``"match"`` mode. Defaults to ``0``.
        videos_list: Full paths of the configured videos to process. Entries
            may be strings or `pathlib.Path` objects. The original
            configuration keys are retained after matching. If ``None``, all
            configured videos applicable to the selected mode are processed.
            Defaults to ``None``.

    Returns:
        In ``"automatic"`` mode, a list of booleans with one entry for each
        video considered by the extraction loop. ``True`` indicates that no
        valid selected frames were extracted from that video, and ``False``
        indicates success or that extraction was skipped by the user. An
        empty list may be returned if frame selection produces no frames.

        In ``"manual"`` and ``"match"`` modes, returns ``None``.

    Raises:
        ValueError: If ``videos_list`` is provided but none of its paths match
            the video paths in the project configuration, or if a required
            video cannot be found while matching cameras.
        RuntimeError: If no videos are processed.
        Exception: If automatic extraction settings in ``config.yaml`` are
            invalid, or if ``"match"`` mode cannot load a valid 3D project
            configuration.

    Warning:
        ``mode="match"`` may remove and replace previously extracted PNG
        frames for cameras other than ``extracted_cam``. If those frames have
        already been labeled, their associated annotation data may no longer
        correspond to the extracted images.

    Note:
        Automatic extraction reads ``numframes2pick``, ``start``, and ``stop``
        from the project configuration.

        Use `deeplabcut.add_new_videos` to add videos to the project
        configuration before extracting frames from them.

        In ``"manual"`` mode, cropping is selected through the interactive
        interface rather than through the ``crop`` argument.

    Examples:
        Extract frames automatically using k-means clustering:

        ```python
        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="automatic",
            algo="kmeans",
        )
        ```

        Select cropping coordinates interactively before automatic
        extraction:

        ```python
        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="automatic",
            algo="kmeans",
            crop="GUI",
        )
        ```

        Include color information during k-means clustering:

        ```python
        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="automatic",
            algo="kmeans",
            cluster_color=True,
        )
        ```

        Extract uniformly selected, cropped frames:

        ```python
        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="automatic",
            algo="uniform",
            crop=True,
        )
        ```

        Extract frames only from selected configured videos:

        ```python
        from pathlib import Path

        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="automatic",
            videos_list=[
                Path("/analysis/project/reaching-task/videos/reaching1.mp4"),
                Path("/analysis/project/reaching-task/videos/reaching2.mp4"),
            ],
        )
        ```

        Open the manual frame-selection interface:

        ```python
        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="manual",
            slider_width=60,
        )
        ```

        Extract frames from the other cameras that correspond to frames
        extracted from the first camera:

        ```python
        deeplabcut.extract_frames(
            "/analysis/project/reaching-task/config.yaml",
            mode="match",
            config3d="/analysis/project/reaching-3d/config.yaml",
            extracted_cam=0,
        )
        ```
    """
    import re
    import sys
    from pathlib import Path

    import numpy as np
    from skimage import io
    from skimage.util import img_as_ubyte

    from deeplabcut.utils import auxiliaryfunctions, frameselectiontools

    videos_list = None if videos_list is None else [Path(video) for video in videos_list]

    config_file = Path(config)
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    configured_videos = list(cfg.get("video_sets_original") or cfg["video_sets"])
    videos = _filter_config_videos(configured_videos, videos_list)

    if videos_list is not None and not videos:
        raise ValueError(
            "None of the selected videos matched the videos in the project "
            "configuration. Selected videos may use a different path representation."
        )

    if mode == "manual":
        from deeplabcut.gui.widgets import launch_napari

        _ = launch_napari(videos[0])
        return

    elif mode == "automatic":
        numframes2pick = cfg["numframes2pick"]
        start = cfg["start"]
        stop = cfg["stop"]

        # Check for variable correctness
        if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
            raise Exception("Erroneous start or stop values. Please correct it in the config file.")
        if numframes2pick < 1 and not int(numframes2pick):
            raise Exception("Perhaps consider extracting more, or a natural number of frames.")

        if opencv:
            from deeplabcut.utils.auxfun_videos import VideoWriter
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
                    cap = VideoWriter(video)
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
                    if any(output_path.iterdir()):
                        if userfeedback:
                            askuser = input(
                                "The directory already contains some frames. Do you want to add to it?(yes/no): "
                            )
                        if not (askuser == "y" or askuser == "yes" or askuser == "Y" or askuser == "Yes"):
                            sys.exit("Delete the frames and try again later!")

                if crop == "GUI":
                    cfg = select_cropping_area(config, [video])
                try:
                    coords = cfg["video_sets"][video]["crop"].split(",")
                except KeyError:
                    coords = cfg["video_sets_original"][video]["crop"].split(",")

                if crop:
                    if opencv:
                        cap.set_bbox(*map(int, coords))
                    else:
                        clip = clip.crop(
                            y1=int(coords[2]),
                            y2=int(coords[3]),
                            x1=int(coords[0]),
                            x2=int(coords[1]),
                        )
                else:
                    coords = None

                print(f"Extracting frames based on {algo} ...")
                if algo == "uniform":
                    if opencv:
                        frames2pick = frameselectiontools.UniformFramescv2(cap, numframes2pick, start, stop)
                    else:
                        frames2pick = frameselectiontools.UniformFrames(clip, numframes2pick, start, stop)
                elif algo == "kmeans":
                    if opencv:
                        frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
                            cap,
                            numframes2pick,
                            start,
                            stop,
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
                        "Please implement this method yourself and send us a pull "
                        "request! Otherwise, choose 'uniform' or 'kmeans'."
                    )
                    frames2pick = []

                if not len(frames2pick):
                    print("Frame selection failed...")
                    return []

                output_path = Path(config).parents[0] / "labeled-data" / Path(video).stem
                output_path.mkdir(parents=True, exist_ok=True)
                is_valid = []
                if opencv:
                    for index in frames2pick:
                        cap.set_to_frame(index)  # extract a particular frame
                        frame = cap.read_frame(crop=True)
                        if frame is not None:
                            image = img_as_ubyte(frame)
                            img_name = str(output_path) + "/img" + str(index).zfill(indexlength) + ".png"
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
                            img_name = str(output_path) + "/img" + str(index).zfill(indexlength) + ".png"
                            io.imsave(img_name, image)
                            if np.var(image) == 0:  # constant image
                                print(
                                    "Seems like black/constant images are extracted from your video."
                                    "Perhaps consider using opencv under the hood, by setting: opencv=True"
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

        if not has_failed:
            raise RuntimeError(
                "No frames were extracted. The project configuration lists no videos, or none could be opened."
            )
        elif all(has_failed):
            print("Frame extraction failed. Video files must be corrupted.")
            return has_failed
        elif any(has_failed):
            print("Although most frames were extracted, some were invalid.")
        else:
            print("Frames were successfully extracted, for the videos listed in the config.yaml file.")
        print(
            "\nYou can now label the frames using the function 'label_frames' "
            "(Note, you should label frames extracted from diverse videos "
            "(and many videos; we do not recommend training on single videos!))."
        )
        return has_failed

    elif mode == "match":
        import cv2

        config_file = Path(config)
        cfg = auxiliaryfunctions.read_config(config_file)
        print("Config file read successfully.")

        videos = _filter_config_videos(sorted(cfg["video_sets"]), videos_list)
        if videos_list is not None and not videos:
            raise ValueError(
                "None of the selected videos matched the videos in the project "
                "configuration. Selected videos may use a different path representation."
            )

        project_path = Path(config).parents[0]
        labels_path = project_path / "labeled-data"
        try:
            cfg_3d = auxiliaryfunctions.read_config(config3d)
        except Exception as e:
            raise Exception(
                "You must create a 3D project and edit the 3D config file before extracting matched frames. \n"
            ) from e
        cams = cfg_3d["camera_names"]
        extCam_name = cams[extracted_cam]
        del cams[extracted_cam]
        label_dirs = sorted(labels_path.glob("*" + extCam_name + "*"))

        # select crop method
        crop_list = []
        for video in videos:
            if extCam_name in video:
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

        for coords, dirPath in zip(crop_list, label_dirs, strict=False):
            extracted_images = list(dirPath.glob("*png"))

            imgPattern = re.compile("[0-9]{1,10}")
            for cam in cams:
                output_path = Path(re.sub(extCam_name, cam, str(dirPath)))

                for p in output_path.iterdir():
                    if p.name.endswith(".png"):
                        p.unlink()

                # Find the matching video from the config `video_sets`,
                # as it may be stored elsewhere than in the `videos` directory.
                video_name = output_path.name
                vid = ""
                for video in cfg["video_sets"]:
                    if video_name in video:
                        vid = video
                        break
                if not vid:
                    raise ValueError(f"Video {video_name} not found...")

                cap = cv2.VideoCapture(vid)
                print("\n extracting matched frames from " + video_name)
                for img in extracted_images:
                    imgNum = re.findall(imgPattern, img.name)[0]
                    cap.set(1, int(imgNum))
                    ret, frame = cap.read()
                    if ret:
                        image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_name = str(output_path / ("img" + imgNum + ".png"))
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
        print("\n Done extracting matched frames. You can now begin labeling frames using the function label_frames\n")

    else:
        print(
            "Invalid MODE. Choose either 'manual', 'automatic' or 'match'. "
            "Check ``help(deeplabcut.extract_frames)`` on python and ``deeplabcut.extract_frames?``"
            " for ipython/jupyter notebook for more details."
        )
