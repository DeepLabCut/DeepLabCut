#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import json
import os
from pathlib import Path
from typing import Optional, Union

from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model
from ruamel.yaml import YAML

from deeplabcut.modelzoo.utils import parse_project_model_name
from deeplabcut.pose_estimation_pytorch.modelzoo.train_from_coco import adaptation_train
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_config_model_paths,
    update_config,
)
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path
from deeplabcut.utils.pseudo_label import (
    dlc3predictions_2_annotation_from_video,
    video_to_frames,
)


def video_inference_superanimal(
    videos: Union[str, list],
    superanimal_name: str,
    scale_list: list = [],
    videotype: str = "mp4",
    dest_folder: Optional[str] = None,
    video_adapt: bool = False,
    plot_trajectories: bool = False,
    pcutoff: float = 0.1,
    adapt_iterations: int = 1000,
    pseudo_threshold: float = 0.1,
    bbox_threshold: float = 0.9,
    detector_epochs: int = 4,
    pose_epochs: int = 4,
    max_individuals: int = 10,
    video_adapt_batch_size: int = 8,
    device: Optional[str] = None,
    customized_pose_checkpoint: Optional[str] = None,
    customized_detector_checkpoint: Optional[str] = None,
):
    """
    This function performs inference on videos using a superanimal model. The model is downloaded from hugginface to the `modelzoo/checkpoints` folder.

    Args:

        videos (str or list): The path to the video or a list of paths to videos.
        superanimal_name (str): The name of the superanimal model.
                                The name should be in the format: {project_name}_{modelname}.
                                For example: `superanimal_topviewmouse_dlcrnet` or `superanimal_quadruped_hrnet`.
        scale_list (list): A list of different resolutions for the spatial pyramid. Used only for bottom up models.

        videotype (str): Checks for the extension of the video in case the input to the video is a directory.
                         Only videos with this extension are analyzed. The default is ``.mp4``.
        dest_folder (str): The path to the folder where the results should be saved.

        video_adapt (bool): Whether to perform video adaptation. The default is False.
                            Current we do not support video adaptation in pytorch models.
        plot_trajectories (bool): Whether to plot the trajectories. The default is False.

        pcutoff (float): The p-value cutoff for the confidence of the prediction. The default is 0.1.

        adapt_iterations (int): Number of iterations for adaptation training. Empirically 1000 is sufficient.

        bbox_threshold (float): The pseudo-label threshold for the confidence of the detector. The default is 0.9

        detector_epochs (int): Used in the pytorch engine. The number of epochs for training the detector. The default is 4.

        pose_epochs (int): Used in the pytorch engine. The number of epochs for training the pose estimator. The default is 4.

        pseudo_threshold (float): The pseudo-label threshold for the confidence of the prediction. The default is 0.1.

        max_individuals (int): The maximum number of individuals in the video. The default is 30. Used only for top down models.

        video_adapt_batch_size (int): The batch size to use for video adaptation.

        device (str): The device to use for inference. The default is None (CPU). Used only for pytorch models.

        customized_pose_checkpoint (str): Used in the pytorch engine. If specified, replacing the default pose checkpoint.

        customized_detector_checkpoint (str): Used in the pytorch engine. If specified, replacing the default detector checkpoint.

    Raises:
        NotImplementedError: If the model is not found in the modelzoo.
        Warning: If the superanimal_name will be deprecated in the future.

    """
    project_name, model_name = parse_project_model_name(superanimal_name)

    dlc_root_path = get_deeplabcut_path()
    modelzoo_path = os.path.join(dlc_root_path, "modelzoo")
    available_architectures = json.load(
        open(os.path.join(modelzoo_path, "models_to_framework.json"), "r")
    )
    framework = available_architectures[model_name]
    print(f"Using {framework} for model {model_name}")

    weight_folder = os.path.join(modelzoo_path, "checkpoints")

    redownload = False
    if framework == "pytorch":
        pose_model_name = f"{project_name}_{model_name}.pth"
        detector_name = f"{project_name}_fasterrcnn.pt"
        rename_mapping = {
            "pose_model.pth": pose_model_name,
            "detector.pt": detector_name,
        }
        if customized_pose_checkpoint is None:
            pose_model_path = os.path.join(weight_folder, pose_model_name)
        else:
            pose_model_path = customized_pose_checkpoint
        detector_model_path = os.path.join(weight_folder, detector_name)
        if not (
            os.path.exists(pose_model_path) and os.path.exists(detector_model_path)
        ):
            redownload = True
    elif framework == "tensorflow":
        weight_folder = os.path.join(weight_folder, f"{project_name}_{model_name}")
        redownload = not os.path.isdir(weight_folder)
        rename_mapping = {}

    if redownload:
        download_huggingface_model(
            superanimal_name, target_dir=weight_folder, rename_mapping=rename_mapping
        )

    if framework == "tensorflow":
        from deeplabcut.pose_estimation_tensorflow.modelzoo.api.superanimal_inference import (
            _video_inference_superanimal,
        )

        if isinstance(videos, str):
            videos = [videos]
        _video_inference_superanimal(
            videos,
            project_name,
            model_name,
            scale_list,
            videotype,
            video_adapt,
            plot_trajectories,
            pcutoff,
            adapt_iterations,
            pseudo_threshold,
        )
    elif framework == "pytorch":
        from deeplabcut.pose_estimation_pytorch.modelzoo.inference import (
            _video_inference_superanimal,
        )

        if video_adapt:
            # the users can pass in many videos. For now, we only use one video for
            # video adaptation. As reported in Ye et al. 2023, one video should be
            # sufficient for video adaptation.
            video_path = Path(videos[0])
            print(f"using {video_path} for video adaptation training")

            # video inference to get pseudo label
            _video_inference_superanimal(
                [str(video_path)],
                project_name,
                model_name,
                max_individuals,
                pcutoff,
                device,
                dest_folder,
                customized_pose_checkpoint=None,
                customized_detector_checkpoint=None,
            )

            (
                model_config,
                project_config,
                pose_model_path,
                detector_path,
            ) = get_config_model_paths(project_name, model_name)
            config = {**project_config, **model_config}
            config = update_config(config, max_individuals, device)

            # we need config to fetch the correct keypoints to dlc3predictions_2_annotation_from_video
            bodyparts = config["metadata"]["bodyparts"]

            # we prepare the pseudo dataset in the same folder of the target video
            pseudo_dataset_folder = video_path.with_name(f"pseudo_{video_path.stem}")
            pseudo_dataset_folder.mkdir(exist_ok=True)
            model_folder = pseudo_dataset_folder / "checkpoints"
            model_folder.mkdir(exist_ok=True)

            image_folder = pseudo_dataset_folder / "images"
            if image_folder.exists():
                print(f"{image_folder} exists, skipping the frame extraction")
            else:
                image_folder.mkdir()
                video_to_frames(video_path, pseudo_dataset_folder)

            anno_folder = pseudo_dataset_folder / "annotations"
            if anno_folder.exists():
                print(f"{anno_folder} exists, skipping the annotation construction")
            else:
                anno_folder.mkdir()

                if dest_folder is None:
                    pseudo_anno_dir = video_path.parent
                else:
                    pseudo_anno_dir = Path(dest_folder)
                dlc_scorer = f"{project_name}_{model_name}"
                pseudo_anno_name = f"{video_path.stem}_{dlc_scorer}_before_adapt.json"
                with open(pseudo_anno_dir / pseudo_anno_name, "r") as f:
                    predictions = json.load(f)

                # make sure we tune parameters inside this function such as pseudo
                # threshold etc.
                dlc3predictions_2_annotation_from_video(
                    predictions,
                    pseudo_dataset_folder,
                    bodyparts,
                    superanimal_name,
                    pose_threshold=pseudo_threshold,
                    bbox_threshold=bbox_threshold,
                )

            # this is probably needed for video generation
            individuals = [f"animal{i}" for i in range(max_individuals)]
            config["metadata"]["individuals"] = individuals

            # the model config's parameters need to be updated for adaptation training
            model_config_path = model_folder / "pytorch_config.yaml"
            with open(model_config_path, "w") as f:
                yaml = YAML()
                yaml.dump(config, f)

            adapted_detector_checkpoint = (
                model_folder / f"snapshot-detector-{detector_epochs:03}.pt"
            )
            adapted_pose_checkpoint = model_folder / f"snapshot-{pose_epochs:03}.pt"

            if (
                adapted_detector_checkpoint.exists()
                and adapted_pose_checkpoint.exists()
            ):
                print(
                    f"Video adaptation already ran; pose ({adapted_pose_checkpoint}) "
                    f"and detector ({adapted_detector_checkpoint}) already exist. To "
                    "rerun video adaptation training, delete the checkpoints or select"
                    "a different number of adaptation epochs. Continuing with the"
                    "existing checkpoints.")
            else:
                adaptation_train(
                    project_root=pseudo_dataset_folder,
                    model_folder=model_folder,
                    train_file="train.json",
                    test_file="test.json",
                    model_config_path=model_config_path,
                    device=device,
                    epochs=pose_epochs,
                    save_epochs=1,
                    detector_epochs=detector_epochs,
                    detector_save_epochs=1,
                    snapshot_path=pose_model_path,
                    detector_path=detector_path,
                    batch_size=video_adapt_batch_size,
                    detector_batch_size=video_adapt_batch_size,
                )

            # Set the customized checkpoint paths
            customized_pose_checkpoint = str(adapted_pose_checkpoint)
            customized_detector_checkpoint = str(adapted_detector_checkpoint)

        return _video_inference_superanimal(
            videos,
            project_name,
            model_name,
            max_individuals,
            pcutoff,
            device,
            dest_folder,
            customized_pose_checkpoint=customized_pose_checkpoint,
            customized_detector_checkpoint=customized_detector_checkpoint,
        )
