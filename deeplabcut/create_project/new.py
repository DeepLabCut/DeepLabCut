"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

Boilerplate project creation inspired from DeepLabChop
by Ronny Eichler
"""
import os
import yaml
from pathlib import Path
import cv2
from deeplabcut import DEBUG
import shutil

def yaml_config_template(yaml_path, cfg):
    """Write a dictionary of configuration into a new yaml file.
    """
    yaml_path = str(yaml_path)
    with open(yaml_path, 'w') as cf:
        keys = list(cfg.keys())
        while len(keys):
            k = keys.pop()
            yaml.dump({k: cfg[k]}, cf, default_flow_style=False)

def create_new_project(project, experimenter, videos, working_directory=None, copy_videos=False,videotype='.avi'):
    """Creates a new project directory, sub-directories and a basic configuration file. The configuration file is loaded with the default values. Change its parameters to your projects need.

    Parameters
    ----------
    project : string
        String containing the name of the project.
        
    experimenter : string
        String containing the name of the experimenter.
        
    videos : list
        A list of string containing the full paths of the videos to include in the project. 
        Attention: Can also be a directory, then all videos of videotype will be imported. Do not pass it as a list!
        
    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string.
        
    copy_videos : bool, optional
        If this is set to True, the videos are copied to the ``videos`` directory. If it is False,symlink of the videos are copied to the project/videos directory. The default is ``False``; if provided it must be either 
        ``True`` or ``False``.

    Example
    --------
    Linux/MacOs
    >>> deeplabcut.create_new_project('reaching-task','Linus',['/data/videos/mouse1.avi','/data/videos/mouse2.avi','/data/videos/mouse3.avi'],'/analysis/project/')
    
    >>> deeplabcut.create_new_project('reaching-task','Linus','/data/videos',videotype='.mp4')
    
    Windows:
    >>> deeplabcut.create_new_project('reaching-task','Bill',['C:\\Users\\rig-95\\Videos\\reachingvideo1.avi'], copy_videos=True)

    """
    from datetime import datetime as dt

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    if working_directory == None:
        working_directory = '.'
    wd = Path(working_directory).resolve()
    project_name = '{pn}-{exp}-{date}'.format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return
    video_path = project_path / 'videos'
    data_path = project_path / 'labeled-data'
    shuffles_path = project_path / 'training-datasets'
    results_path = project_path / 'dlc-models'
    for p in [video_path, data_path, shuffles_path, results_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # Import all videos in a folder or if just one video withouth [] passed, then make it a list.
    if isinstance(videos,str):
        #there are two cases:
        if os.path.isdir(videos): # it is a path!
            path=videos
            videos=[os.path.join(path,vp) for vp in os.listdir(path) if videotype in vp]
            if len(videos)==0:
                print("No videos found in",path,os.listdir(path))
                print("Perhaps change the videotype, which is currently set to:", videotype)
            else:
                print("Directory entered, " , len(videos)," videos were found.")
        else:
            if os.path.isfile(videos):
                videos=[videos]

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
            shutil.copy(os.fspath(src),os.fspath(dst)) #https://www.python.org/dev/peps/pep-0519/
            #https://github.com/AlexEMG/DeepLabCut/issues/105 (for windows)
            
            #try:
            #    #shutil.copy(src,dst)
            #except OSError or TypeError: #https://github.com/AlexEMG/DeepLabCut/issues/105 (for windows)
            #    shutil.copy(os.fspath(src),os.fspath(dst))
    else:
      # creates the symlinks of the video and puts it in the videos directory.
        print("Creating the symbolic link of the video")
        for src, dst in zip(videos, destinations):
            if dst.exists() and not DEBUG:
                raise FileExistsError('Video {} exists already!'.format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
            except OSError:
                import subprocess
                subprocess.check_call('mklink %s %s' %(dst,src),shell = True)
            print('Created the symlink of {} to {}'.format(src, dst))
            videos = destinations
    
    # adds the video list to the config.yaml file
    video_sets = {}
    for video in videos:
        print(video)
        try:
           rel_video_path = os.path.realpath(video)
        except:
           rel_video_path = os.readlink(str(video))

        vcap = cv2.VideoCapture(rel_video_path)
        if vcap.isOpened():
           width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           video_sets[rel_video_path] = {'crop': ', '.join(map(str, [0, width, 0, height]))}
        else:
           print("Cannot open the video file!")
           video_sets=None

    # Configuration file templates
    cfg_dict = {'Task': project,
                'video_sets': video_sets,
                'cropping': False,
                'start': 0,
                'stop' : 1,
                'numframes2pick': 20,
                'bodyparts': ['hand', 'Finger1', 'Finger2',"Joystick"],
                'scorer': experimenter,
                'date':d,
                'TrainingFraction': [0.95],
                'resnet': 50,
                'snapshotindex':-1,
                'pcutoff':0.1,
                'corner2move2': (50,50),
                'move2corner': False,
                'x1':0,
                'x2':640,
                'y1':277,
                'y2':624,
                'dotsize':12,          #for plots size of dots
                'alphavalue':.5,          #for plots transparency of markers
                'colormap': 'jet',          #for plots type of colormap
                'iteration':0,
                'project_path': str(project_path),
                'batch_size': 4 #batch size during inference (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242
                }

    projconfigfile=os.path.join(str(project_path),'config.yaml')
    # Write dictionary to yaml  config file
    yaml_config_template(projconfigfile, cfg_dict)
    print('Generated "{}"'.format(project_path / 'config.yaml'))
    print("\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. Change the parameters in this file to adapt to your project's needs.\n Once you have changed the configuration file, use the function 'extract_frames' to select frames for labeling.\n. [OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)." %(project_name,str(wd)))
    return projconfigfile
