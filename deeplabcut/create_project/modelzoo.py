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

from collections.abc import Sequence
from pathlib import Path

from dlclibrary import get_available_detectors
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    get_available_datasets,
    get_available_models,
)

import deeplabcut
from deeplabcut.core.config import ProjectConfig
from deeplabcut.core.deprecation import deprecated, renamed_parameter
from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset.metadata import (
    DataSplit,
    ShuffleMetadata,
    TrainingDatasetMetadata,
)
from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    MakeInference_yaml,
)
from deeplabcut.modelzoo.utils import get_super_animal_project_cfg
from deeplabcut.pose_estimation_pytorch.config import (
    PoseMetadata,
    make_pytorch_test_config,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import load_super_animal_config
from deeplabcut.utils import auxiliaryfunctions


@deprecated(replacement="deeplabcut.create_pretrained_project(..., model='full_human')", since="3.0.0")
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def create_pretrained_human_project(
    project,
    experimenter,
    videos,
    working_directory=None,
    copy_videos=False,
    video_extensions: str | Sequence[str] | None = None,
    createlabeledvideo=True,
    analyzevideo=True,
):
    """Creates a demo human project and analyzes a video with ResNet 101 weights pretrained on
    MPII Human Pose. This is from the DeeperCut paper by Insafutdinov et al. https://arxiv.org/abs/1605.03170
    Please make sure to cite it too if you use this code!
    """
    deeplabcut.create_pretrained_project(
        project,
        experimenter,
        videos,
        model="full_human",
        working_directory=working_directory,
        copy_videos=copy_videos,
        video_extensions=video_extensions,
        createlabeledvideo=createlabeledvideo,
        analyzevideo=analyzevideo,
        engine=Engine.TF,
    )


@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def create_pretrained_project(
    project: str,
    experimenter: str,
    videos: list[str],
    model: str | None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    video_extensions: str | Sequence[str] | None = None,
    analyzevideo: bool = True,
    filtered: bool = True,
    createlabeledvideo: bool = True,
    trainFraction: float | None = None,
    multi_animal: bool = False,
    individuals: list[str] | None = None,
    net_name: str | None = None,
    detector_name: str | None = None,
):
    r"""Creates a new project directory, sub-directories and a basic configuration file.
    Change its parameters to your projects need.

    The project will also be initialized with a pre-trained model from the DeepLabCut model zoo!

    http://modelzoo.deeplabcut.org

    Args:
        project (string): String containing the name of the project.
        experimenter (string): String containing the name of the experimenter.
        model (string | None, optional): The model / dataset to use as basis for the
            project. If None, the default model / dataset for the selected engine will
            be used. Defaults to None.
        videos (list[string]): A list of string containing the full paths of the videos
            to include in the project.
        working_directory (string, optional): The directory where the project will be
            created. If None - the current working directory will be used. Defaults to
            None.
        copy_videos (bool, optional): If this is set to True, the videos are copied to
            the ``videos`` directory. If it is False, symlink of the videos are copied
            to the project/videos directory.
            Note: on Windows, True is necessary when not running in Administrator mode.
            The same applies whenever symlinks are disabled or unsupported.
            Defaults to False.
        analyzevideo (bool, optional): If true, then the video is analyzed and a labeled
            video is created. If false, then only the project will be created and the
            weights downloaded.
        filtered (bool, optional): Indicates if filtered pose data output should be
            plotted rather than frame-by-frame predictions. Filtered version can be
            calculated with deeplabcut.filterpredictions(). Defaults to True.
        createlabeledvideo (bool, optional): Specifies if a labeled video needs to be
            created. Defaults to True.
        trainFraction (float | None, optional): Fraction that will be used in
            dlc-model/trainingset folder name. If None - default value (0.95) from new
            projects will be used. Defaults to None.
        multi_animal (bool, optional): Specifies if the project is single or
            multi-animal. Implemented only for Pytorch-based models. Defaults to False.
        individuals (list[str] | None, optional): Only if multianimal is True. Defines
            the names of the individuals. Defaults to None.
        net_name (str | None, optional): Valid only if using Pytorch engine. Name of the
            pose model on which the superanimal dataset has been trained on. If None -
            "hrnet_w32" will be used as default. Defaults to None.
        detector_name (str | None, optional): Valid only if using Pytorch engine. Name
            of the detector model on which the superanimal dataset has been trained on.
            If None - "fasterrcnn_resnet50_fpn_v2" will be used as default. Defaults to
            None.

    Examples:
        Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi:

            deeplabcut.create_pretrained_project(
                "humanstrokestudy", "Linus", ["/data/videos/homosapiens1.avi"], copy_videos=False
            )

        Loading full_cat model and analyzing video "felixfeliscatus3.avi":

            deeplabcut.create_pretrained_project(
                "humanstrokestudy", "Linus", ["/data/videos/felixfeliscatus3.avi"], model="full_cat", engine=Engine.TF
            )

        Windows:

            deeplabcut.create_pretrained_project(
                "humanstrokestudy",
                "Bill",
                [r"C:\yourusername\rig-95\Videos\reachingvideo1.avi"],
                r"C:\yourusername\analysis\project",
                copy_videos=True,
            )

        On Windows, paths should be formatted as ``r`"C:\"`` or ``"C:\\"`` (i.e. a double backslash).
    """
    return create_pretrained_project_pytorch(
        project=project,
        experimenter=experimenter,
        videos=videos,
        dataset=model,
        working_directory=working_directory,
        copy_videos=copy_videos,
        video_extensions=video_extensions,
        analyze_video=analyzevideo,
        filtered=filtered,
        create_labeled_video=createlabeledvideo,
        train_fraction=trainFraction,
        multi_animal=multi_animal,
        individuals=individuals,
        net_name=net_name,
        detector_name=detector_name,
    )


def create_pretrained_project_pytorch(
    project: str,
    experimenter: str,
    videos: list[str],
    dataset: str | None = None,
    working_directory: str | None = None,
    copy_videos: bool = False,
    video_extensions: str | None = None,
    analyze_video: bool = True,
    filtered: bool = True,
    create_labeled_video: bool = True,
    train_fraction: float | None = None,
    multi_animal: bool = False,
    individuals: list[str] | None = None,
    net_name: str | None = None,
    detector_name: str | None = None,
):
    r"""Method used specifically for Pytorch-based ModelZoo models.

    Creates a new project directory, sub-directories and a basic configuration file.
    Change its parameters to your projects need.

    The project will also be initialized with a pre-trained model from the DeepLabCut model zoo!

    http://modelzoo.deeplabcut.org

    Args:
        project (string): String containing the name of the project.
        experimenter (string): String containing the name of the experimenter.
        dataset (string | None, optional): The superanimal dataset to use as basis for
            the project. If not specified - superanimal_quadruped will be used by
            default. Defaults to None.
        videos (list[string]): A list of string containing the full paths of the videos
            to include in the project.
        working_directory (string, optional): The directory where the project will be
            created. If None - the current working directory will be used. Defaults to
            None.
        copy_videos (bool, optional): If this is set to True, the videos are copied to
            the ``videos`` directory. If it is False, symlink of the videos are copied
            to the project/videos directory. Note: on Windows: True is often necessary!
            Defaults to False.
        analyze_video (bool, optional): If true, then the video is analyzed and a
            labeled video is created. If false, then only the project will be created
            and the weights downloaded.
        filtered (bool, optional): Indicates if filtered pose data output should be
            plotted rather than frame-by-frame predictions. Filtered version can be
            calculated with deeplabcut.filterpredictions(). Defaults to True.
        create_labeled_video (bool, optional): Specifies if a labeled video needs to be
            created. Defaults to True.
        train_fraction (float | None, optional): Fraction that will be used in
            dlc-model/trainingset folder name. If None - default value (0.95) from new
            projects will be used. Defaults to None.
        multi_animal (bool, optional): Specifies if the project is single or
            multi-animal. Defaults to False.
        individuals (list[str] | None, optional): Only if multianimal is True. Defines
            the names of the individuals. Defaults to None.
        net_name (str | None, optional): Valid only if using Pytorch engine. Name of the
            pose model on which the superanimal dataset has been trained on. If None -
            "hrnet_w32" will be used as default. Defaults to None.
        detector_name (str | None, optional): Valid only if using Pytorch engine. Name
            of the detector model on which the superanimal dataset has been trained on.
            If None - "fasterrcnn_resnet50_fpn_v2" will be used as default. Defaults to
            None.

    Examples:
        Linux/MacOs loading full_human model and analyzing video /homosapiens1.avi:

            deeplabcut.create_pretrained_project_pytorch(
                "humanstrokestudy", "Linus", ["/data/videos/homosapiens1.avi"], copy_videos=False
            )

        Loading full_cat model and analyzing video "felixfeliscatus3.avi":

            deeplabcut.create_pretrained_project_pytorch(
                "humanstrokestudy", "Linus", ["/data/videos/felixfeliscatus3.avi"], model="full_cat", engine=Engine.TF
            )

        Windows:

            deeplabcut.create_pretrained_project_pytorch(
                "humanstrokestudy",
                "Bill",
                [r"C:\yourusername\rig-95\Videos\reachingvideo1.avi"],
                r"C:\yourusername\analysis\project",
                copy_videos=True,
            )

        On Windows, paths should be formatted as ``r`"C:\"`` or ``"C:\\"`` (i.e. a double backslash).
    """
    # Check arguments
    if not dataset:
        dataset = "superanimal_quadruped"

    if not net_name:
        net_name = "hrnet_w32"

    # Currently, all Pytorch Superanimal models are Top-Down.
    if not detector_name:
        detector_name = "fasterrcnn_resnet50_fpn_v2"

    if dataset not in get_available_datasets():
        raise ValueError(f"Invalid dataset '{dataset}'. Available datasets are: {get_available_datasets()}")

    if net_name not in get_available_models(dataset):
        raise ValueError(
            f"Invalid net_name '{net_name}' for dataset {dataset}. "
            f"The following net types are available: {get_available_models(dataset)}"
        )

    if detector_name not in get_available_detectors(dataset):
        raise ValueError(
            f"Invalid detector_name '{detector_name}' for dataset {dataset}. "
            f"The following detectors are available: {get_available_detectors(dataset)}"
        )

    # Create project
    cfg_path = deeplabcut.create_new_project(
        project=project,
        experimenter=experimenter,
        videos=videos,
        working_directory=working_directory,
        copy_videos=copy_videos,
        video_extensions=video_extensions,
        multianimal=multi_animal,
        individuals=individuals,
    )

    # Edits to do to the project config
    cfg_edits = {}
    if train_fraction is not None:
        cfg_edits["TrainingFraction"] = [train_fraction]
    super_animal_project_cfg = get_super_animal_project_cfg(dataset)
    super_animal_bodyparts = super_animal_project_cfg.get("bodyparts")
    super_animal_skeleton = super_animal_project_cfg.get("skeleton")
    cfg_edits["skeleton"] = super_animal_skeleton
    if multi_animal:
        cfg_edits["multianimalbodyparts"] = super_animal_bodyparts
    else:
        cfg_edits["bodyparts"] = super_animal_bodyparts
    config = ProjectConfig.from_yaml(cfg_path)
    config.update(cfg_edits)
    config.to_yaml(cfg_path, log_changes=True, mark_clean=True)

    # Create the shuffle train and test directories
    shuffle_dir = Path(cfg_path).parent / auxiliaryfunctions.get_model_folder(
        trainFraction=config["TrainingFraction"][0],
        shuffle=1,
        cfg=config,
        engine=Engine.PYTORCH,
    )
    train_dir = shuffle_dir / "train"
    test_dir = shuffle_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Download the weights and put them into appropriate directory
    print("Downloading weights...")
    super_animal_detector_name = f"{dataset}_{detector_name}"
    new_detector_name = "snapshot-detector-000.pt"
    download_huggingface_model(
        model_name=super_animal_detector_name,
        target_dir=str(train_dir),
        rename_mapping={f"{super_animal_detector_name}.pt": new_detector_name},
    )
    super_animal_model_name = f"{dataset}_{net_name}"
    new_snapshot_name = "snapshot-000.pt"
    download_huggingface_model(
        model_name=super_animal_model_name,
        target_dir=str(train_dir),
        rename_mapping={f"{super_animal_model_name}.pt": new_snapshot_name},
    )

    # Create pytorch_config.yaml
    train_cfg_path = train_dir / "pytorch_config.yaml"
    pytorch_config = load_super_animal_config(
        super_animal=dataset,
        model_name=net_name,
        detector_name=detector_name,
    )
    pytorch_config["metadata"] = PoseMetadata.build(config, pose_config_path=train_cfg_path).to_dict()
    pytorch_config["resume_training_from"] = str(train_dir / new_snapshot_name)
    pytorch_config["detector"]["resume_training_from"] = str(train_dir / new_detector_name)
    pytorch_config.to_yaml(train_cfg_path)

    # Create test pose_cfg.yaml
    test_cfg_path = test_dir / "pose_cfg.yaml"
    make_pytorch_test_config(model_config=pytorch_config, test_config_path=test_cfg_path, save=True)

    # Create inference_cfg.yaml if needed
    if multi_animal:
        inference_cfg_path = test_dir / "inference_cfg.yaml"
        _create_inference_config(inference_cfg_path, config)

    # Create metadata.yaml with shuffle info in training-data directory
    _create_training_datasets_metadata(config, shuffle_dir.name, Engine.PYTORCH)

    # Process the videos
    _process_videos(
        cfg_path=cfg_path,
        video_extensions=video_extensions,
        analyze_video=analyze_video,
        filtered=filtered,
        create_labeled_video=create_labeled_video,
    )
    return cfg_path, str(train_cfg_path)


def _create_inference_config(inference_cfg_path: str | Path, project_cfg: dict):
    inf_updates = dict(
        minimalnumberofconnections=int(len(project_cfg["multianimalbodyparts"]) / 2),
        topktoretain=len(project_cfg["individuals"]),
        withid=project_cfg.get("identity", False),
    )
    default_inf_path = auxiliaryfunctions.get_deeplabcut_path() / "inference_cfg.yaml"
    MakeInference_yaml(inf_updates, inference_cfg_path, default_inf_path)


def _create_training_datasets_metadata(config: dict, shuffle_dir_name: str, engine: Engine):
    # First create the metadata object
    metadata = TrainingDatasetMetadata.create(config)

    # Create a new shuffle with TensorFlow engine
    new_shuffle = ShuffleMetadata(
        name=shuffle_dir_name,
        train_fraction=config["TrainingFraction"][0],
        index=1,
        engine=engine,
        split=DataSplit(train_indices=(), test_indices=()),
    )

    # Add the shuffle to metadata
    metadata = metadata.add(new_shuffle)

    # Save the metadata
    metadata.save()

    return metadata


def _process_videos(
    cfg_path: str | Path,
    video_extensions: str | Sequence[str] | None = None,
    analyze_video: bool = True,
    filtered: bool = True,
    create_labeled_video: bool = True,
):
    cfg_path = str(cfg_path)
    video_dir = Path(cfg_path).parent / "videos"

    if analyze_video:
        print("Analyzing video...")
        deeplabcut.analyze_videos(cfg_path, [video_dir], video_extensions=video_extensions, save_as_csv=True)

    if create_labeled_video:
        if filtered:
            deeplabcut.filterpredictions(cfg_path, [video_dir], video_extensions)

        print("Plotting results...")
        deeplabcut.create_labeled_video(cfg_path, [video_dir], video_extensions, draw_skeleton=True, filtered=filtered)
        deeplabcut.plot_trajectories(cfg_path, [video_dir], video_extensions, filtered=filtered)
