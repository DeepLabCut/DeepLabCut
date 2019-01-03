"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

def add_new_videos(config,videos,copy_videos=False,coords=None):
    """
    Add new videos to the config file at any stage of the project.

    Parameters
    ----------
    config : str
        Full path of the config.yaml file.
    videos : list of str
        List of full paths of the videos to include in the project.
    copy_videos : bool, optional
        Copy videos to videos directory (default False).
        If ``True``, the videos are copied to the project/videos directory.
        If ``False``, symlinks of the videos are copied to the project/videos directory.j
    coords : list of int or None, optional
        Cropping coordinates of the video (default None).

    Examples
    --------

    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi'],copy_videos=False,coords=[0,100,0,200])

    """
    import os
    import shutil
    import yaml
    from pathlib import Path

    from deeplabcut import DEBUG
    from deeplabcut.utils import auxiliaryfunctions
    import cv2

    # Read the config file
    cfg = auxiliaryfunctions.read_config(config)

    for idx,video in enumerate(videos):
        try:
           video_path = os.path.realpath(video)
        except:
           video_path = os.readlink(video)

        vcap = cv2.VideoCapture(video_path)
        if vcap.isOpened():
                  # get vcap property
           width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        else:
           print("Cannot open the video file!")
        if coords == None:
            cfg['video_sets'].update({video_path : {'crop': ', '.join(map(str, [0, width, 0, height]))}})
        else:
            c = coords[idx]
            cfg['video_sets'].update({video_path : {'crop': ', '.join(map(str, c))}})

    with open(str(config), 'w') as ymlfile:
        yaml.dump(cfg, ymlfile,default_flow_style=False)

    video_path = Path(config).parents[0] / 'videos'
    data_path = Path(config).parents[0] / 'labeled-data'
    videos = [Path(vp) for vp in videos]

    dirs = [data_path/Path(i.stem) for i in videos]

    for p in dirs:
        """
        Creates directory under data
        """
        p.mkdir(parents = True, exist_ok = True)
    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_videos==True:
        print("Copying the videos")
        for src, dst in zip(videos, destinations):
            shutil.copy(os.fspath(src),os.fspath(dst))
    else:
        print("Creating the symbolic link of the video")
        for src, dst in zip(videos, destinations):
            if dst.exists() and not DEBUG:
                raise FileExistsError('Video {} exists already!'.format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
            except shutil.SameFileError:
                if not DEBUG:
                    raise
    print("New video was added to the project! Use the function 'extract_frames' to select frames for labeling.")
