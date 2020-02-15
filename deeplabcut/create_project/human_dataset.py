"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import deeplabcut
import os
import subprocess
import yaml
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions, auxfun_models


def MakeTrain_pose_yaml(itemstochange,saveasconfigfile,defaultconfigfile):
    raw = open(defaultconfigfile).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc,Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]
    #docs[0]['init_weights'] = '../../pretrained/resnet_v1_101.ckpt'
    docs[0]['max_input_size'] = 1500
    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]

def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]
    dict_test['scoremap_dir'] = 'test'
    dict_test['global_scale'] = 1.0
    #dict_test['init_weights'] = 'models/mpii/snapshot-1030000'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)

def create_pretrained_human_project(project,experimenter,videos,working_directory=None,copy_videos=False,videotype='.avi',createlabeledvideo=True, analyzevideo=True):
    """
    Creates a demo human project and analyzes a video with ResNet 101 weights pretrained on MPII Human Pose. This is from the DeeperCut paper by Insafutdinov et al. https://arxiv.org/abs/1605.03170 Please make sure to cite it too if you use this code!

    Parameters
    ----------
    project : string
        String containing the name of the project.

    experimenter : string
        String containing the name of the experimenter.

    videos : list
        A list of string containing the full paths of the videos to include in the project.

    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string.

    copy_videos : bool, optional
        If this is set to True, the videos are copied to the ``videos`` directory. If it is False,symlink of the videos are copied to the project/videos directory. The default is ``False``; if provided it must be either
        ``True`` or ``False``.
    analyzevideo " bool, optional
        If true, then the video is analzyed and a labeled video is created. If false, then only the project will be created and the weights downloaded. You can then access them

    Example
    --------
    Linux/MacOs
    >>> deeplabcut.create_pretrained_human_project('human','Linus',['/data/videos/mouse1.avi'],'/analysis/project/',copy_videos=False)

    Windows:
    >>> deeplabcut.create_pretrained_human_project('human','Bill',[r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'],r'C:\yourusername\analysis\project' copy_videos=False)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    --------
    """

    cfg=deeplabcut.create_new_project(project,experimenter,videos,working_directory,copy_videos,videotype)

    config = auxiliaryfunctions.read_config(cfg)
    config['bodyparts'] = ['ankle1','knee1','hip1','hip2','knee2','ankle2','wrist1','elbow1','shoulder1','shoulder2','elbow2','wrist2','chin','forehead']
    config['skeleton'] = [['ankle1', 'knee1'],['ankle2', 'knee2'],['knee1', 'hip1'],['knee2', 'hip2'],['hip1', 'hip2'], ['shoulder1', 'shoulder2'], ['shoulder1', 'hip1'], ['shoulder2', 'hip2'], ['shoulder1', 'elbow1'], ['shoulder2', 'elbow2'], ['chin', 'forehead'], ['elbow1', 'wrist1'], ['elbow2', 'wrist2']]
    config['default_net_type']='resnet_101'
    auxiliaryfunctions.write_config(cfg,config)
    config = auxiliaryfunctions.read_config(cfg)

    train_dir = Path(os.path.join(config['project_path'],str(auxiliaryfunctions.GetModelFolder(trainFraction=config['TrainingFraction'][0],shuffle=1,cfg=config)),'train'))
    test_dir = Path(os.path.join(config['project_path'],str(auxiliaryfunctions.GetModelFolder(trainFraction=config['TrainingFraction'][0],shuffle=1,cfg=config)),'test'))

    # Create the model directory
    train_dir.mkdir(parents=True,exist_ok=True)
    test_dir.mkdir(parents=True,exist_ok=True)

    modelfoldername=auxiliaryfunctions.GetModelFolder(trainFraction=config['TrainingFraction'][0],shuffle=1,cfg=config)

    path_train_config = str(os.path.join(config['project_path'],Path(modelfoldername),'train','pose_cfg.yaml'))
    path_test_config = str(os.path.join(config['project_path'],Path(modelfoldername),'test','pose_cfg.yaml'))

    # Download the weights and put then in appropriate directory
    cwd = os.getcwd()
    os.chdir(train_dir)
    print("Checking if the weights are already available, otherwise I will download them!")
    weightfilename=auxfun_models.download_mpii_weigths(train_dir)
    os.chdir(cwd)

    # Create the pose_config.yaml files
    parent_path = Path(os.path.dirname(deeplabcut.__file__))
    defaultconfigfile = str(parent_path / 'pose_cfg.yaml')
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(config)
    datafilename,metadatafilename=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction=config['TrainingFraction'][0],shuffle=1,cfg=config)
    bodyparts = config['bodyparts']
    net_type ='resnet_101'
    num_shuffles= 1
    model_path,num_shuffles=auxfun_models.Check4weights(net_type,parent_path,num_shuffles)
    items2change = {"dataset": 'dataset-test.mat',#datafilename,
                        "metadataset": metadatafilename,
                        "num_joints": len(bodyparts),
                        "all_joints": [[i] for i in range(len(bodyparts))],
                        "all_joints_names": [str(bpt) for bpt in bodyparts],
                        "init_weights": weightfilename.split('.index')[0], #'models/mpii/snapshot-1030000',
                        "project_path": str(config['project_path']),
                        "net_type": net_type,
                        "dataset_type": "default"
                    }
    trainingdata = MakeTrain_pose_yaml(items2change,path_train_config,defaultconfigfile)

    keys2save = ["dataset", "dataset_type","num_joints", "all_joints", "all_joints_names",
                        "net_type", 'init_weights', 'global_scale', 'location_refinement',
                        'locref_stdev']
    MakeTest_pose_yaml(trainingdata, keys2save,path_test_config)

    video_dir = os.path.join(config['project_path'],'videos')

    if analyzevideo==True:
        # Analyze the videos
        deeplabcut.analyze_videos(cfg, [video_dir], videotype, save_as_csv=True)
    if createlabeledvideo==True:
        deeplabcut.create_labeled_video(cfg,[video_dir],videotype, draw_skeleton=True)
        deeplabcut.plot_trajectories(cfg, [video_dir], videotype)
    return cfg, path_train_config
