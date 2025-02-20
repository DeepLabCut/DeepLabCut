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
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    MODELOPTIONS,
)

import deeplabcut
from deeplabcut import Engine
from deeplabcut.generate_training_dataset.metadata import TrainingDatasetMetadata, ShuffleMetadata, DataSplit
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.utils import auxiliaryfunctions

Modeloptions = MODELOPTIONS  # backwards compatibility for COLAB NOTEBOOK


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
    project: str,
    experimenter: str,
    videos: list[str],
    model: str | None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    videotype: str = "",
    analyzevideo: bool = True,
    filtered: bool = True,
    createlabeledvideo: bool = True,
    trainFraction: float | None = None,
    engine: Engine = Engine.PYTORCH,
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

    model: string | None, default = None,
        The model / dataset to use as basis for the project.
        If None, the default model / dataset for the selected engine will be used.

    videos : list[string]
        A list of string containing the full paths of the videos to include in the project.

    working_directory : string, optional, default = None
        The directory where the project will be created. If None - the current working directory will be used.

    copy_videos : bool, optional, default = False,
        If this is set to True, the videos are copied to the ``videos`` directory.
        If it is False, symlink of the videos are copied to the project/videos directory.
        Note: on Windows: True is often necessary!

    analyzevideo: bool, optional
        If true, then the video is analyzed and a labeled video is created.
        If false, then only the project will be created and the weights downloaded.

    filtered: bool, default True
        Indicates if filtered pose data output should be plotted rather than frame-by-frame predictions.
        Filtered version can be calculated with deeplabcut.filterpredictions()

    createlabeledvideo: bool, default True,
        Specifies if a labeled video needs to be created.

    trainFraction: float|None, default = None.
            Fraction that will be used in dlc-model/trainingset folder name.
            If None - default value (0.95) from new projects will be used.

    engine: Engine, default Engine.PYTORCH,
        engine on which the pretrained weights are based

    Example
    --------
    Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi
    >>> deeplabcut.create_pretrained_project("humanstrokestudy", "Linus", ["/data/videos/homosapiens1.avi"], copy_videos=False)

    Loading full_cat model and analyzing video "felixfeliscatus3.avi"
    >>> deeplabcut.create_pretrained_project("humanstrokestudy", "Linus", ["/data/videos/felixfeliscatus3.avi"], model="full_cat", engine=Engine.TF)

    Windows:
    >>> deeplabcut.create_pretrained_project("humanstrokestudy", "Bill", [r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'], r'C:\yourusername\analysis\project', copy_videos=True)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    """
    if engine == Engine.TF:
        return create_pretrained_project_tensorflow(
            project=project,
            experimenter=experimenter,
            videos=videos,
            model=model,
            working_directory=working_directory,
            copy_videos=copy_videos,
            videotype=videotype,
            analyzevideo=analyzevideo,
            filtered=filtered,
            createlabeledvideo=createlabeledvideo,
            trainFraction=trainFraction,
        )
    elif engine == Engine.PYTORCH:
        return create_pretrained_project_pytorch(
            project=project,
            experimenter=experimenter,
            videos=videos,
            dataset=model,
            working_directory=working_directory,
            copy_videos=copy_videos,
            video_type=videotype,
            analyze_video=analyzevideo,
            filtered=filtered,
            create_labeled_video=createlabeledvideo,
            train_fraction=trainFraction,
        )

    raise NotImplementedError(f"This function is not implemented for {engine}")


def create_pretrained_project_pytorch(
    project: str,
    experimenter: str,
    videos: list[str],
    dataset: str | None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    video_type: str | None = None,
    analyze_video: bool = True,
    filtered: bool = True,
    create_labeled_video: bool = True,
    train_fraction: float | None = None,
):
    """
    Method used specifically for Pytorch-based ModelZoo models.

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

    dataset: string|None, default = None,
        The superanimal dataset to use as basis for the project.
        If not specified - superanimal_quadruped will be used by default.

    videos : list[string]
        A list of string containing the full paths of the videos to include in the project.

    working_directory : string, optional, default = None
        The directory where the project will be created. If None - the current working directory will be used.

    copy_videos : bool, optional, default = False,
        If this is set to True, the videos are copied to the ``videos`` directory.
        If it is False, symlink of the videos are copied to the project/videos directory.
        Note: on Windows: True is often necessary!

    analyze_video: bool, optional
        If true, then the video is analyzed and a labeled video is created.
        If false, then only the project will be created and the weights downloaded.

    filtered: bool, default True
        Indicates if filtered pose data output should be plotted rather than frame-by-frame predictions.
        Filtered version can be calculated with deeplabcut.filterpredictions()

    create_labeled_video: bool, default True
        Specifies if a labeled video needs to be created.

    train_fraction: float|None, default = None.
            Fraction that will be used in dlc-model/trainingset folder name.
            If None - default value (0.95) from new projects will be used.

    Example
    --------
    Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi
    >>> deeplabcut.create_pretrained_project_pytorch("humanstrokestudy", "Linus", ["/data/videos/homosapiens1.avi"], copy_videos=False)

    Loading full_cat model and analyzing video "felixfeliscatus3.avi"
    >>> deeplabcut.create_pretrained_project_pytorch("humanstrokestudy", "Linus", ["/data/videos/felixfeliscatus3.avi"], model="full_cat", engine=Engine.TF)

    Windows:
    >>> deeplabcut.create_pretrained_project_pytorch("humanstrokestudy", "Bill", [r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'], r'C:\yourusername\analysis\project', copy_videos=True)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    """
    pass


def create_pretrained_project_tensorflow(
    project: str,
    experimenter: str,
    videos: list[str],
    model: str|None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    videotype: str = "",
    analyzevideo: bool = True,
    filtered: bool = True,
    createlabeledvideo: bool = True,
    trainFraction: float | None = None,
):
    """
    Method used specifically for Tensorflow-based ModelZoo models.

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

    model: string|None, default = None,
        The model / dataset to use as basis for the project.
        If not specified - full_human will be used by default.

    videos : list[string]
        A list of string containing the full paths of the videos to include in the project.

    working_directory : string, optional, default = None
        The directory where the project will be created. If None - the current working directory will be used.

    copy_videos : bool, optional, default = False,
        If this is set to True, the videos are copied to the ``videos`` directory.
        If it is False, symlink of the videos are copied to the project/videos directory.
        Note: on Windows: True is often necessary!

    analyzevideo: bool, optional
        If true, then the video is analyzed and a labeled video is created.
        If false, then only the project will be created and the weights downloaded.

    filtered: bool, default True
        Indicates if filtered pose data output should be plotted rather than frame-by-frame predictions.
        Filtered version can be calculated with deeplabcut.filterpredictions()

    createlabeledvideo: bool, default True
        Specifies if a labeled video needs to be created.

    trainFraction: float|None, default = None.
            Fraction that will be used in dlc-model/trainingset folder name.
            If None - default value (0.95) from new projects will be used.

    Example
    --------
    Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi
    >>> deeplabcut.create_pretrained_project_tensorflow("humanstrokestudy", "Linus", ["/data/videos/homosapiens1.avi"], copy_videos=False)

    Loading full_cat model and analyzing video "felixfeliscatus3.avi"
    >>> deeplabcut.create_pretrained_project_tensorflow("humanstrokestudy", "Linus", ["/data/videos/felixfeliscatus3.avi"], model="full_cat", engine=Engine.TF)

    Windows:
    >>> deeplabcut.create_pretrained_project_tensorflow("humanstrokestudy", "Bill", [r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'], r'C:\yourusername\analysis\project', copy_videos=True)
    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    """
    if not model:
        model = "full_human"

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
        # model_path = auxfun_models.check_for_weights(pose_cfg['net_type'], parent_path)

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

        _create_training_datasets_metadata(config, modelfoldername.name, Engine.TF)

        _process_videos(
            cfg_path=cfg,
            video_type=videotype,
            analyze_video=analyzevideo,
            filtered=filtered,
            create_labeled_video=createlabeledvideo,
        )

        os.chdir(cwd)
        return cfg, path_train_config

    else:
        return "N/A", "N/A"

def _create_training_datasets_metadata(config: dict, shuffle_dir_name: str, engine: Engine):
    # First create the metadata object
    metadata = TrainingDatasetMetadata.create(config)

    # Create a new shuffle with TensorFlow engine
    new_shuffle = ShuffleMetadata(
        name=shuffle_dir_name,
        train_fraction=config["TrainingFraction"][0],
        index=1,
        engine=engine,
        split=DataSplit(train_indices=(),test_indices=())
    )

    # Add the shuffle to metadata
    metadata = metadata.add(new_shuffle)

    # Save the metadata
    metadata.save()

    return metadata


def _process_videos(
    cfg_path: str | Path,
    video_type: str = "",
    analyze_video: bool = True,
    filtered: bool = True,
    create_labeled_video: bool = True,
):
    cfg_path = str(cfg_path)
    video_dir = Path(cfg_path).parent / "videos"

    if analyze_video:
        print("Analyzing video...")
        deeplabcut.analyze_videos(cfg_path, [video_dir], videotype=video_type, save_as_csv=True)

    if create_labeled_video:
        if filtered:
            deeplabcut.filterpredictions(cfg_path, [video_dir], video_type)

        print("Plotting results...")
        deeplabcut.create_labeled_video(
            cfg_path, [video_dir], video_type, draw_skeleton=True, filtered=filtered
        )
        deeplabcut.plot_trajectories(cfg_path, [video_dir], video_type, filtered=filtered)