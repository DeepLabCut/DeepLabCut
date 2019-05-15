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
    config : string
        String containing the full path of the config file in the project.

    videos : list
        A list of string containing the full paths of the videos to include in the project.

    copy_videos : bool, optional
        If this is set to True, the symlink of the videos are copied to the project/videos directory. The default is
        ``False``; if provided it must be either ``True`` or ``False``.
    coords: list, optional
      A list containing the list of cropping coordinates of the video. The default is set to None.
    Examples
    --------
    Video will be added, with cropping dimenions according to the frame dimensinos of mouse5.avi
    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi'])
    
    Video will be added, with cropping dimenions [0,100,0,200]
    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi'],copy_videos=False,coords=[[0,100,0,200]])

    Two videos will be added, with cropping dimenions [0,100,0,200] and [0,100,0,250], respectively.
    >>> deeplabcut.add_new_videos('/home/project/reaching-task-Tanmay-2018-08-23/config.yaml',['/data/videos/mouse5.avi','/data/videos/mouse6.avi'],copy_videos=False,coords=[[0,100,0,200],[0,100,0,250]])

    """
    import os
    import shutil
    from pathlib import Path

    from deeplabcut import DEBUG
    from deeplabcut.utils import auxiliaryfunctions
    import cv2

    # Read the config file
    cfg = auxiliaryfunctions.read_config(config)

    video_path = Path(config).parents[0] / 'videos'
    data_path = Path(config).parents[0] / 'labeled-data'
    videos = [Path(vp) for vp in videos]

    dirs = [data_path/Path(i.stem) for i in videos]

    for p in dirs:
        """
        Creates directory under data & perhaps copies videos (to /video)
        """
        p.mkdir(parents = True, exist_ok = True)
    
    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_videos==True:
        for src, dst in zip(videos, destinations):
            if dst.exists():
                pass
            else:
                print("Copying the videos")
                shutil.copy(os.fspath(src),os.fspath(dst)) 
    else:
        for src, dst in zip(videos, destinations):
            if dst.exists():
                pass
            else:
                print("Creating the symbolic link of the video")
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
    
    if copy_videos==True:
        videos=destinations # in this case the *new* location should be added to the config file
    # adds the video list to the config.yaml file
    for idx,video in enumerate(videos):
        try:
# For windows os.path.realpath does not work and does not link to the real video. 
           video_path = str(Path.resolve(Path(video)))
#           video_path = os.path.realpath(video)
        except:
           video_path = os.readlink(video)

        vcap = cv2.VideoCapture(video_path)
        if vcap.isOpened():
            # get vcap property
           width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           if coords == None:
                cfg['video_sets'].update({video_path : {'crop': ', '.join(map(str, [0, width, 0, height]))}})
           else:
                c = coords[idx]
                cfg['video_sets'].update({video_path : {'crop': ', '.join(map(str, c))}})
        else:
           print("Cannot open the video file!")

    auxiliaryfunctions.write_config(config,cfg)
    print("New video was added to the project! Use the function 'extract_frames' to select frames for labeling.")
