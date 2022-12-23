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

import os
from pathlib import Path

import yaml

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    MODELOPTIONS,
)

Modeloptions = MODELOPTIONS  # backwards compatability for COLAB NOTEBOOK


def MakeTrain_pose_yaml(itemstochange, saveasconfigfile, defaultconfigfile):
    raw = open(defaultconfigfile).read()
    docs = []
    for raw_doc in raw.split("\n---"):
        try:
            docs.append(yaml.load(raw_doc, Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]
    docs[0]["max_input_size"] = 1500
    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def UpdateTrain_pose_yaml(dict_train, dict2change, saveasfile):
    for key in dict2change.keys():
        dict_train[key] = dict2change[key]
    auxiliaryfunctions.write_plainconfig(saveasfile, dict_train)


def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]
    dict_test["scoremap_dir"] = "test"
    dict_test["global_scale"] = 1.0
    auxiliaryfunctions.write_plainconfig(saveasfile, dict_test)
    # with open(saveasfile, "w") as f:
    #    yaml.dump(dict_test, f)


def create_pretrained_human_project(
    project,
    experimenter,
    videos,
    working_directory=None,
    copy_videos=False,
    videotype="",
    createlabeledvideo=True,
    analyzevideo=True,
):
    """
    LEGACY FUNCTION will be deprecated.

    Use deeplabcut.create_pretrained_project(project, experimenter, videos, model='full_human', ..)

    For now just calls that function....

    Creates a demo human project and analyzes a video with ResNet 101 weights pretrained on
    MPII Human Pose. This is from the DeeperCut paper by Insafutdinov et al. https://arxiv.org/abs/1605.03170
    Please make sure to cite it too if you use this code!
    """
    print(
        "LEGACY FUNCTION will be deprecated.... use  deeplabcut.create_pretrained_project(project, experimenter, videos, model='full_human', ..) in the future!"
    )
    create_pretrained_project(
        project,
        experimenter,
        videos,
        model="full_human",
        working_directory=working_directory,
        copy_videos=copy_videos,
        videotype=videotype,
        createlabeledvideo=createlabeledvideo,
        analyzevideo=analyzevideo,
    )


def create_pretrained_project(
    project,
    experimenter,
    videos,
    model="full_human",
    working_directory=None,
    copy_videos=False,
    videotype="",
    analyzevideo=True,
    filtered=True,
    createlabeledvideo=True,
    trainFraction=None,
):
    """
    Creates a new project directory, sub-directories and a basic configuration file.
    Change its parameters to your projects need.

    The project will also be initialized with a pre-trained model from the DeepLabCut model zoo!

    http://modelzoo.deeplabcut.org

    Parameters
    ----------
    project : string
        String containing the name of the project.

    experimenter : string
        String containing the name of the experimenter.

    model: string, options see  http://www.mousemotorlab.org/dlc-modelzoo
        Current option and default: 'full_human'  Creates a demo human project and analyzes a video with ResNet 101 weights pretrained on MPII Human Pose. This is from the DeeperCut paper
        by Insafutdinov et al. https://arxiv.org/abs/1605.03170 Please make sure to cite it too if you use this code!

    videos : list
        A list of string containing the full paths of the videos to include in the project.

    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string.

    copy_videos : bool, optional  ON WINDOWS: TRUE is often necessary!
        If this is set to True, the videos are copied to the ``videos`` directory. If it is False,symlink of the videos are copied to the project/videos directory. The default is ``False``; if provided it must be either
        ``True`` or ``False``.

    analyzevideo " bool, optional
        If true, then the video is analyzed and a labeled video is created. If false, then only the project will be created and the weights downloaded. You can then access them

    filtered: bool, default false
        Boolean variable indicating if filtered pose data output should be plotted rather than frame-by-frame predictions.
        Filtered version can be calculated with deeplabcut.filterpredictions

    trainFraction: By default value from *new* projects. (0.95)
            Fraction that will be used in dlc-model/trainingset folder name.

    Example
    --------
    Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi
    >>> deeplabcut.create_pretrained_project('humanstrokestudy','Linus',['/data/videos/homosapiens1.avi'], copy_videos=False)

    Loading full_cat model and analyzing video "felixfeliscatus3.avi"
    >>> deeplabcut.create_pretrained_project('humanstrokestudy','Linus',['/data/videos/felixfeliscatus3.avi'], model='full_cat')

    Windows:
    >>> deeplabcut.create_pretrained_project('humanstrokestudy','Bill',[r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'],r'C:\yourusername\analysis\project' copy_videos=True)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )

    """
    if model in MODELOPTIONS:
        cwd = os.getcwd()

        cfg = deeplabcut.create_new_project(
            project, experimenter, videos, working_directory, copy_videos, videotype
        )
        if trainFraction is not None:
            auxiliaryfunctions.edit_config(cfg, {"TrainingFraction": [trainFraction]})

        config = auxiliaryfunctions.read_config(cfg)
        if model == "full_human":
            config["bodyparts"] = [
                "ankle1",
                "knee1",
                "hip1",
                "hip2",
                "knee2",
                "ankle2",
                "wrist1",
                "elbow1",
                "shoulder1",
                "shoulder2",
                "elbow2",
                "wrist2",
                "chin",
                "forehead",
            ]
            config["skeleton"] = [
                ["ankle1", "knee1"],
                ["ankle2", "knee2"],
                ["knee1", "hip1"],
                ["knee2", "hip2"],
                ["hip1", "hip2"],
                ["shoulder1", "shoulder2"],
                ["shoulder1", "hip1"],
                ["shoulder2", "hip2"],
                ["shoulder1", "elbow1"],
                ["shoulder2", "elbow2"],
                ["chin", "forehead"],
                ["elbow1", "wrist1"],
                ["elbow2", "wrist2"],
            ]
            config["default_net_type"] = "resnet_101"
        else:  # just make a case and put the stuff you want.
            # TBD: 'partaffinityfield_graph' >> use to set skeleton!
            pass

        auxiliaryfunctions.write_config(cfg, config)
        config = auxiliaryfunctions.read_config(cfg)

        train_dir = Path(
            os.path.join(
                config["project_path"],
                str(
                    auxiliaryfunctions.get_model_folder(
                        trainFraction=config["TrainingFraction"][0],
                        shuffle=1,
                        cfg=config,
                    )
                ),
                "train",
            )
        )
        test_dir = Path(
            os.path.join(
                config["project_path"],
                str(
                    auxiliaryfunctions.get_model_folder(
                        trainFraction=config["TrainingFraction"][0],
                        shuffle=1,
                        cfg=config,
                    )
                ),
                "test",
            )
        )

        # Create the model directory
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        modelfoldername = auxiliaryfunctions.get_model_folder(
            trainFraction=config["TrainingFraction"][0], shuffle=1, cfg=config
        )
        path_train_config = str(
            os.path.join(
                config["project_path"], Path(modelfoldername), "train", "pose_cfg.yaml"
            )
        )
        path_test_config = str(
            os.path.join(
                config["project_path"], Path(modelfoldername), "test", "pose_cfg.yaml"
            )
        )

        # Download the weights and put then in appropriate directory
        print("Downloading weights...")
        download_huggingface_model(model, train_dir)

        pose_cfg = deeplabcut.auxiliaryfunctions.read_plainconfig(path_train_config)
        pose_cfg["dataset_type"] = "imgaug"
        print(path_train_config)
        # Updating config file:
        dict_ = {
            "default_net_type": pose_cfg["net_type"],
            "default_augmenter": pose_cfg["dataset_type"],
            "bodyparts": pose_cfg["all_joints_names"],
            "dotsize": 6,
        }
        auxiliaryfunctions.edit_config(cfg, dict_)

        # downloading base encoder / not required unless on re-trains (but when a training set is created this happens anyway)
        # model_path, num_shuffles=auxfun_models.Check4weights(pose_cfg['net_type'], parent_path, num_shuffles= 1)

        # Updating training and test pose_cfg:
        snapshotname = [fn for fn in os.listdir(train_dir) if ".meta" in fn][0].split(
            ".meta"
        )[0]
        dict2change = {
            "init_weights": str(os.path.join(train_dir, snapshotname)),
            "project_path": str(config["project_path"]),
        }

        UpdateTrain_pose_yaml(pose_cfg, dict2change, path_train_config)
        keys2save = [
            "dataset",
            "dataset_type",
            "num_joints",
            "all_joints",
            "all_joints_names",
            "net_type",
            "init_weights",
            "global_scale",
            "location_refinement",
            "locref_stdev",
        ]

        MakeTest_pose_yaml(pose_cfg, keys2save, path_test_config)

        video_dir = os.path.join(config["project_path"], "videos")
        if analyzevideo == True:
            print("Analyzing video...")
            deeplabcut.analyze_videos(cfg, [video_dir], videotype, save_as_csv=True)

        if createlabeledvideo == True:
            if filtered:
                deeplabcut.filterpredictions(cfg, [video_dir], videotype)

            print("Plotting results...")
            deeplabcut.create_labeled_video(
                cfg, [video_dir], videotype, draw_skeleton=True, filtered=filtered
            )
            deeplabcut.plot_trajectories(cfg, [video_dir], videotype, filtered=filtered)

        os.chdir(cwd)
        return cfg, path_train_config

    else:
        return "N/A", "N/A"
