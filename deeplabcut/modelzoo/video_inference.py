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
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

import torch
from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model
from ruamel.yaml import YAML

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.modelzoo.utils import get_super_animal_scorer
from deeplabcut.pose_estimation_pytorch.modelzoo.train_from_coco import adaptation_train
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_snapshot_folder_path,
    get_super_animal_snapshot_path,
    load_super_animal_config,
    update_config,
)
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path
from deeplabcut.utils.pseudo_label import (
    dlc3predictions_2_annotation_from_video,
    video_to_frames,
)


def get_checkpoint_epoch(checkpoint_path):
    """
    Load a PyTorch checkpoint and return the current epoch number.

    Args:
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        int: Current epoch number, or 0 if not found
    """
    # Use CUDA if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "metadata" in checkpoint and "epoch" in checkpoint["metadata"]:
        return checkpoint["metadata"]["epoch"]
    else:
        return 0


def video_inference_superanimal(
    videos: Union[str, list],
    superanimal_name: str,
    model_name: str,
    detector_name: str | None = None,
    scale_list: Optional[list] = None,
    videotype: str = ".mp4",
    dest_folder: Optional[str] = None,
    cropping: list[int] | None = None,
    video_adapt: bool = False,
    plot_trajectories: bool = False,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    pcutoff: float = 0.1,
    adapt_iterations: int = 1000,
    pseudo_threshold: float = 0.1,
    bbox_threshold: float = 0.9,
    detector_epochs: int = 4,
    pose_epochs: int = 4,
    max_individuals: int = 10,
    video_adapt_batch_size: int = 8,
    device: Optional[str] = "auto",
    customized_pose_checkpoint: Optional[str] = None,
    customized_detector_checkpoint: Optional[str] = None,
    customized_model_config: Optional[str] = None,
    plot_bboxes: bool = True,
):
    """
    This function performs inference on videos using a pretrained SuperAnimal model.

    IMPORTANT: Note that since we have both TensorFlow and PyTorch Engines, we will
    route the engine based on the model you select:

        * dlcrnet -> TensorFlow
        * all others - > PyTorch

    Parameters
    ----------

    videos (str or list):
        The path to the video or a list of paths to videos.

    superanimal_name (str):
        The name of the SuperAnimal dataset for which to load a pre-trained model.

    model_name (str):
        The model architecture to use for inference.

    detector_name (str):
        For top-down models (only available with the PyTorch framework), the type of
        object detector to use for inference.

    scale_list (list):
        A list of different resolutions for the spatial pyramid. Used only for bottom up models.

    videotype (str):
        Checks for the extension of the video in case the input to the video is a directory.
        Only videos with this extension are analyzed. The default is ``.mp4``.

    dest_folder (str): The path to the folder where the results should be saved.

    cropping: list or None, optional, default=None
        Only for SuperAnimal models running with the PyTorch engine.
        List of cropping coordinates as [x1, x2, y1, y2].
        Note that the same cropping parameters will then be used for all videos.
        If different video crops are desired, run ``video_inference_superanimal`` on
        individual videos with the corresponding cropping coordinates.

    video_adapt (bool):
        Whether to perform video adaptation. The default is False.
        You only need to perform it on one video because the adaptation generalizes to all videos that are similar.

    plot_trajectories (bool):
        Whether to plot the trajectories. The default is False.

    batch_size (int):
        The batch size to use for video inference. Only for PyTorch models.

    detector_batch_size (int):
        The batch size to use for the detector during video inference. Only for PyTorch.

    pcutoff (float):
        The p-value cutoff for the confidence of the prediction. The default is 0.1.

    adapt_iterations (int):
        Number of iterations for adaptation training. Empirically 1000 is sufficient.

    bbox_threshold (float):
        The pseudo-label threshold for the confidence of the detector. The default is 0.9

    detector_epochs (int):
        Used in the PyTorch engine. The number of epochs for training the detector. The default is 4.

    pose_epochs (int):
        Used in the PyTorch engine. The number of epochs for training the pose estimator. The default is 4.

    pseudo_threshold (float):
        The pseudo-label threshold for the confidence of the prediction. The default is 0.1.

    max_individuals (int):
        The maximum number of individuals in the video. The default is 30. Used only for top down models.

    video_adapt_batch_size (int):
        The batch size to use for video adaptation.

    device (str):
        The device to use for inference. The default is None (CPU). Used only for PyTorch models.

    customized_pose_checkpoint (str):
        Used in the PyTorch engine. If specified, it replaces the default pose checkpoint.

    customized_detector_checkpoint (str):
        Used in the PyTorch engine. If specified, it replaces the default detector checkpoint.

    customized_model_config (str):
        Used for loading customized model config. Only supported in Pytorch

    plot_bboxes (bool):
        If using Top-Down approach, whether to plot the detector's bounding boxes. The default is True.

    Raises:
        NotImplementedError:
        If the model is not found in the modelzoo.
        Warning: If the superanimal_name will be deprecated in the future.

    (Model Explanation) SuperAnimal-Quadruped:
    `superanimal_quadruped` models aim to work across a large range of quadruped
    animals, from horses, dogs, sheep, rodents, to elephants. The camera perspective is
    orthogonal to the animal ("side view"), and most of the data includes the animals
    face (thus the front and side of the animal). You will note we have several variants
    that differ in speed vs. performance, so please do test them out on your data to see
    which is best suited for your application. Also note we have a "video adaptation"
    feature, which lets you adapt your data to the model in a self-supervised way.
    No labeling needed!

    All model snapshots are automatically downloaded to modelzoo/checkpoints when used.

    - PLEASE SEE THE FULL DATASHEET: https://zenodo.org/records/10619173
    - MORE DETAILS ON THE MODELS (detector, pose estimators):
        https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped
    - We provide several models:
        - `hrnet_w32` (Top-Down pose estimation model, PyTorch engine)
            An `hrnet_w32` is a top-down model that is paired with a detector. That
            means it takes a cropped image from an object detector and predicts the
            keypoints. When selecting this variant, a `detector_name` must be set with
            one of the provided object detectors.
        - `dlcrnet` (TensorFlow engine)
            This is a bottom-up model that predicts all keypoints then groups them into
            individuals. This can be faster, but more error prone.
    - We provide one object detector (only for the PyTorch engine):
        - `fasterrcnn_resnet50_fpn_v2`
            This is a FasterRCNN model with a ResNet backbone, see
            https://pytorch.org/vision/stable/models/faster_rcnn.html

    (Model Explanation) SuperAnimal-TopViewMouse:
    `superanimal_topviewmouse` aims to work across lab mice in different lab settings
    from a top-view perspective; this is very polar in many behavioral assays in freely
    moving mice.

    All model snapshots are automatically downloaded to modelzoo/checkpoints when used.

    - [PLEASE SEE THE FULL DATASHEET HERE](https://zenodo.org/records/10618947)
    - [MORE DETAILS ON THE MODELS (detector, pose estimators)](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
    - We provide several models:
        - `hrnet_w32` (Top-Down pose estimation model, PyTorch engine)
            An `hrnet_w32` is a top-down model that is paired with a detector. That
            means it takes a cropped image from an object detector and predicts the
            keypoints. When selecting this variant, a `detector_name` must be set with
            one of the provided object detectors.
        - `dlcrnet` (TensorFlow engine)
            This is a bottom-up model that predicts all keypoints then groups them into
            individuals. This can be faster, but more error prone.
    - We provide one object detector (only for the PyTorch engine):
        - `fasterrcnn_resnet50_fpn_v2`
            This is a FasterRCNN model with a ResNet backbone, see
            https://pytorch.org/vision/stable/models/faster_rcnn.html

    (Model Explanation) SuperAnimal-Bird:
    `superanimal_superbird` model aims to work on various bird species. It was developed 
    during the 2024 DLC AI Residency Program. More info can be 
    [found here](https://deeplabcut.medium.com/deeplabcut-ai-residency-2024-recap-working-with-the-superanimal-bird-model-and-dlc-3-0-live-e55807ca2c7c)

    (Model Explanation) SuperAnimal-HumanBody:
    `superanimal_humanbody` models aim to work across human body pose estimation
    from various camera perspectives and environments. The models are designed to
    handle different human poses, activities, and lighting conditions commonly
    found in human motion analysis, sports analysis, and behavioral studies.

    All model snapshots are automatically downloaded to modelzoo/checkpoints when used.

    - We provide:
        - `rtmpose_x` (Top-Down pose estimation model, PyTorch engine)
            An `rtmpose_x` is a top-down model that is paired with a detector. That
            means it takes a cropped image from an object detector and predicts the
            keypoints. When selecting this variant, a `detector_name` must be set with
            one of the provided object detectors. This model uses 17 body parts in
            the COCO body7 format.
    - We provide an object detector (PyTorch engine):
        - `fasterrcnn_mobilenet_v3_large_fpn`
            This is a FasterRCNN model with a MobileNet backbone, see
            https://pytorch.org/vision/stable/models/faster_rcnn.html

    Examples (PyTorch Engine)
    --------
    >>> import deeplabcut.modelzoo.video_inference.video_inference_superanimal as video_inference_superanimal
    >>> video_inference_superanimal(
        videos=["/mnt/md0/shaokai/DLCdev/3mice_video1_short.mp4"],
        superanimal_name="superanimal_topviewmouse",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        video_adapt=True,
        max_individuals=3,
        pseudo_threshold=0.1,
        bbox_threshold=0.9,
        detector_epochs=4,
        pose_epochs=4,
    )

    Tips:
    * max_individuals: make sure you correctly give the number of individuals. Our
        inference api will only give up to max_individuals number of predictions.
    * pseudo_threshold: the higher you set, the more aggressive you filter low
        confidence predictions during video adaptation.
    * bbox_threshold: the higher you set, the more aggressive you filter low confidence
        bounding boxes during video adaptation. Different from our paper, we now add
        video adaptation to the object detector as well.
    * detector_epochs and pose_epochs do not need to be to high as video adaptation does
        not require too much training. However, you can make them higher if you see a
        substaintial gain in the training logs.

    Examples
    --------

    >>> from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    >>> videos = ["/path/to/my/video.mp4"]
    >>> superanimal_name = "superanimal_topviewmouse"
    >>> videotype = "mp4"
    >>> scale_list = [200, 300, 400]
    >>> video_inference_superanimal(
            videos,
            superanimal_name,
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            scale_list = scale_list,
            videotype = videotype,
            video_adapt = True,
        )

    Tips:
    scale_list: it's recommended to leave this as empty list []. Empirically
    [200, 300, 400] works well. We needed to do this as bottom-up models in TensorFlow
    are sensitive to the scales of the image.
    If you find your predictions not good without scale_list or it's too hard to find
    the right scale_list, you can try to use the PyTorch engine.
    """
    if scale_list is None:
        scale_list = []

    print(f"Running video inference on {videos} with {superanimal_name}_{model_name}")
    dlc_root_path = get_deeplabcut_path()
    modelzoo_path = os.path.join(dlc_root_path, "modelzoo")
    available_architectures = json.load(
        open(os.path.join(modelzoo_path, "models_to_framework.json"), "r")
    )
    framework = available_architectures[model_name]
    print(f"Using {framework} for model {model_name}")
    if framework == "tensorflow":
        from deeplabcut.pose_estimation_tensorflow.modelzoo.api.superanimal_inference import (
            _video_inference_superanimal,
        )

        weight_folder = get_snapshot_folder_path() / f"{superanimal_name}_{model_name}"
        if not weight_folder.exists():
            download_huggingface_model(
                superanimal_name, target_dir=str(weight_folder), rename_mapping=None
            )

        if isinstance(videos, str):
            videos = [videos]
        _video_inference_superanimal(
            videos,
            superanimal_name,
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
        if detector_name is None:
            raise ValueError(
                "You have to specify a detector_name when using the Pytorch framework."
            )

        # Special handling for superanimal_humanbody - use dedicated implementation
        if superanimal_name == "superanimal_humanbody":
            from deeplabcut.pose_estimation_pytorch.modelzoo.superanimal_humanbody_video_inference import (
                analyze_videos_superanimal_humanbody,
            )
            
            # Convert videos to list if needed
            if isinstance(videos, str):
                videos = [videos]
            
            # Set destination folder
            if dest_folder is None:
                dest_folder = Path(videos[0]).parent
            else:
                dest_folder = Path(dest_folder)
            
            if not dest_folder.exists():
                dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Map parameters to the dedicated function
            # Note: analyze_videos_superanimal_humanbody has its own parameter set
            # Handle device parameter - convert "auto" to actual device
            if device == "auto":
                import torch
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device
            
            dedicated_kwargs = {
                "videotype": videotype,
                "destfolder": str(dest_folder),
                "bbox_threshold": bbox_threshold,
                "pose_threshold": pcutoff,
                "device": actual_device,
                "cropping": cropping,
                "batch_size": batch_size,
                "detector_batch_size": detector_batch_size,
            }
            
            # Use a dummy config path since the dedicated function loads its own config
            dummy_config = "superanimal_humanbody"
            
            results = analyze_videos_superanimal_humanbody(
                dummy_config,
                videos,
                **dedicated_kwargs,
            )
            
            return results

        # Standard PyTorch implementation for other models
        from deeplabcut.pose_estimation_pytorch.modelzoo.inference import (
            _video_inference_superanimal,
        )

        if customized_model_config is not None:
            config = read_config_as_dict(customized_model_config)
        else:
            config = load_super_animal_config(
                super_animal=superanimal_name,
                model_name=model_name,
                detector_name=detector_name,
            )

        pose_model_path = customized_pose_checkpoint
        if pose_model_path is None:
            pose_model_path = get_super_animal_snapshot_path(
                dataset=superanimal_name,
                model_name=model_name,
            )

        detector_path = customized_detector_checkpoint
        if detector_path is None:
            detector_path = get_super_animal_snapshot_path(
                dataset=superanimal_name,
                model_name=detector_name,
            )

        dlc_scorer = get_super_animal_scorer(
            superanimal_name, pose_model_path, detector_path
        )

        # Add superanimal_name to config metadata for all superanimal models (needed for detector routing)
        if "metadata" not in config:
            config["metadata"] = {}
        config["metadata"]["superanimal_name"] = superanimal_name
        
        config = update_config(config, max_individuals, device)
        
        output_suffix = "_before_adapt"
        if video_adapt:
            # the users can pass in many videos. For now, we only use one video for
            # video adaptation. As reported in Ye et al. 2024, one video should be
            # sufficient for video adaptation.
            video_path = Path(videos[0])
            print(f"Using {video_path} for video adaptation training")

            # video inference to get pseudo label
            _video_inference_superanimal(
                [str(video_path)],
                superanimal_name,
                model_cfg=config,
                model_snapshot_path=pose_model_path,
                detector_snapshot_path=detector_path,
                max_individuals=max_individuals,
                pcutoff=pcutoff,
                batch_size=batch_size,
                detector_batch_size=detector_batch_size,
                cropping=cropping,
                dest_folder=dest_folder,
                output_suffix=output_suffix,
                plot_bboxes=plot_bboxes,
                bboxes_pcutoff=bbox_threshold,
            )

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
                print(
                    f"Video frames being extracted to {image_folder} for video "
                    f"adaptation."
                )
                video_to_frames(video_path, pseudo_dataset_folder, cropping=cropping)

            anno_folder = pseudo_dataset_folder / "annotations"
            if (anno_folder / "train.json").exists() and (
                anno_folder / "test.json"
            ).exists():
                print(
                    f"{anno_folder} exists, skipping the annotation construction. "
                    f"Delete the folder if you want to re-construct pseudo annotations"
                )
            else:
                anno_folder.mkdir()

                if dest_folder is None:
                    pseudo_anno_dir = video_path.parent
                else:
                    pseudo_anno_dir = Path(dest_folder)

                pseudo_anno_name = f"{video_path.stem}_{dlc_scorer}_before_adapt.json"
                with open(pseudo_anno_dir / pseudo_anno_name, "r") as f:
                    predictions = json.load(f)

                # make sure we tune parameters inside this function such as pseudo
                # threshold etc.
                print(f"Constructing pseudo dataset at {pseudo_dataset_folder}")
                dlc3predictions_2_annotation_from_video(
                    predictions,
                    pseudo_dataset_folder,
                    config["metadata"]["bodyparts"],
                    superanimal_name,
                    pose_threshold=pseudo_threshold,
                    bbox_threshold=bbox_threshold,
                )

            model_snapshot_prefix = f"snapshot-{model_name}"
            detector_snapshot_prefix = f"snapshot-{detector_name}"

            config["runner"]["snapshot_prefix"] = model_snapshot_prefix
            config["detector"]["runner"]["snapshot_prefix"] = detector_snapshot_prefix

            # the model config's parameters need to be updated for adaptation training
            model_config_path = model_folder / "pytorch_config.yaml"
            with open(model_config_path, "w") as f:
                yaml = YAML()
                yaml.dump(config, f)

            # get the current epoch of the detector and pose model
            current_pose_epoch = get_checkpoint_epoch(pose_model_path)
            current_detector_epoch = get_checkpoint_epoch(detector_path)
            # update the checkpoint path with the current epoch, if the checkpoint does not exist, use the best checkpoint
            adapted_detector_checkpoint = (
                model_folder
                / f"{detector_snapshot_prefix}-{current_detector_epoch + detector_epochs:03}.pt"
            )
            adapted_pose_checkpoint = (
                model_folder
                / f"{model_snapshot_prefix}-{current_pose_epoch + pose_epochs:03}.pt"
            )
            if not Path(adapted_detector_checkpoint).exists():
                adapted_detector_checkpoint = (
                    model_folder
                    / f"{detector_snapshot_prefix}-best-{current_detector_epoch + detector_epochs:03}.pt"
                )
            if not Path(adapted_pose_checkpoint).exists():
                adapted_pose_checkpoint = (
                    model_folder
                    / f"{model_snapshot_prefix}-best-{current_pose_epoch + pose_epochs:03}.pt"
                )

            if (
                adapted_detector_checkpoint.exists()
                and adapted_pose_checkpoint.exists()
            ):
                print(
                    f"Video adaptation already ran; pose ({adapted_pose_checkpoint}) "
                    f"and detector ({adapted_detector_checkpoint}) already exist. To "
                    "rerun video adaptation training, delete the checkpoints or select"
                    "a different number of adaptation epochs. Continuing with the"
                    "existing checkpoints."
                )
            else:
                print(
                    "Running video adaptation with following parameters:\n"
                    f"  (pose training) pose_epochs: {pose_epochs}\n"
                    "  (pose) save_epochs: 1\n"
                    f"  detector_epochs: {detector_epochs}\n"
                    "  detector_save_epochs: 1\n"
                    f"  video adaptation batch size: {video_adapt_batch_size}\n"
                )
                train_file = pseudo_dataset_folder / "annotations" / "train.json"
                with open(train_file, "r") as f:
                    temp_obj = json.load(f)

                annotations = temp_obj["annotations"]
                if len(annotations) == 0:
                    print(
                        f"No valid predictions from {str(video_path)}. Check the "
                        "quality of the video"
                    )
                    return

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

            # after video adaptation, re-update the adapted checkpoint path, if the checkpoint does not exist, use the best checkpoint
            adapted_detector_checkpoint = (
                model_folder
                / f"{detector_snapshot_prefix}-{current_detector_epoch + detector_epochs:03}.pt"
            )
            adapted_pose_checkpoint = (
                model_folder
                / f"{model_snapshot_prefix}-{current_pose_epoch + pose_epochs:03}.pt"
            )
            if not Path(adapted_detector_checkpoint).exists():
                adapted_detector_checkpoint = (
                    model_folder
                    / f"{detector_snapshot_prefix}-best-{current_detector_epoch + detector_epochs:03}.pt"
                )
            if not Path(adapted_pose_checkpoint).exists():
                adapted_pose_checkpoint = (
                    model_folder
                    / f"{model_snapshot_prefix}-best-{current_pose_epoch + pose_epochs:03}.pt"
                )

            # Set the customized checkpoint paths and
            output_suffix = "_after_adapt"
            detector_path = adapted_detector_checkpoint
            pose_model_path = adapted_pose_checkpoint

        return _video_inference_superanimal(
            videos,
            superanimal_name,
            model_cfg=config,
            model_snapshot_path=pose_model_path,
            detector_snapshot_path=detector_path,
            max_individuals=max_individuals,
            pcutoff=pcutoff,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            cropping=cropping,
            dest_folder=dest_folder,
            output_suffix=output_suffix,
            plot_bboxes=plot_bboxes,
            bboxes_pcutoff=bbox_threshold,
        )
