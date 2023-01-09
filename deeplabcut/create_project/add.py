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


def add_new_videos(
    config, videos, copy_videos=False, coords=None, extract_frames=False
):
    """
    Add new videos to the config file at any stage of the project.

    Parameters
    ----------
    config : string
        String containing the full path of the config file in the project.

    videos : list
        A list of strings containing the full paths of the videos to include in the project.

    copy_videos : bool, optional
        If this is set to True, the symlink of the videos are copied to the project/videos directory. The default is
        ``False``; if provided it must be either ``True`` or ``False``.

    coords: list, optional
        A list containing the list of cropping coordinates of the video. The default is set to None.

    extract_frames: bool, optional
        if this is set to True extract_frames will be run on the new videos

    Examples
    --------
    Video will be added, with cropping dimensions according to the frame dimensions of mouse5.avi
    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi'])

    Video will be added, with cropping dimensions [0,100,0,200]
    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi'],copy_videos=False,coords=[[0,100,0,200]])

    Two videos will be added, with cropping dimensions [0,100,0,200] and [0,100,0,250], respectively.
    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi','/data/videos/mouse6.avi'],copy_videos=False,coords=[[0,100,0,200],[0,100,0,250]])

    """
    import os
    import shutil
    from pathlib import Path

    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.utils.auxfun_videos import VideoReader
    from deeplabcut.generate_training_dataset import frame_extraction

    # Read the config file
    cfg = auxiliaryfunctions.read_config(config)

    video_path = Path(config).parents[0] / "videos"
    data_path = Path(config).parents[0] / "labeled-data"
    videos = [Path(vp) for vp in videos]

    dirs = [data_path / Path(i.stem) for i in videos]

    for p in dirs:
        """
        Creates directory under data & perhaps copies videos (to /video)
        """
        p.mkdir(parents=True, exist_ok=True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_videos:
        for src, dst in zip(videos, destinations):
            if dst.exists():
                pass
            else:
                print("Copying the videos")
                shutil.copy(os.fspath(src), os.fspath(dst))

    else:
        # creates the symlinks of the video and puts it in the videos directory.
        print("Attempting to create a symbolic link of the video ...")
        for src, dst in zip(videos, destinations):
            if dst.exists():
                print(f"Video {dst} already exists. Skipping...")
                continue
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
    for idx, video in enumerate(videos):
        try:
            # For windows os.path.realpath does not work and does not link to the real video.
            video_path = str(Path.resolve(Path(video)))
        #           video_path = os.path.realpath(video)
        except:
            video_path = os.readlink(video)

        vid = VideoReader(video_path)
        if coords is not None:
            c = coords[idx]
        else:
            c = vid.get_bbox()
        params = {video_path: {"crop": ", ".join(map(str, c))}}
        if "video_sets_original" not in cfg:
            cfg["video_sets"].update(params)
        else:
            cfg["video_sets_original"].update(params)
    videos_str = [str(video) for video in videos]
    if extract_frames:
        frame_extraction.extract_frames(
            config, userfeedback=False, videos_list=videos_str
        )
        print(
            "New videos were added to the project and frames have been extracted for labeling!"
        )
    else:
        print(
            "New videos were added to the project! Use the function 'extract_frames' to select frames for labeling."
        )
    auxiliaryfunctions.write_config(config, cfg)
