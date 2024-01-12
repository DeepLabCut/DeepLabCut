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
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.postprocessor import (
    build_bottom_up_postprocessor,
    build_detector_postprocessor,
    build_top_down_postprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
    build_bottom_up_preprocessor,
    build_top_down_preprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.transforms import (
    CoarseDropout,
    ElasticTransform,
    Grayscale,
    KeepAspectRatioResize,
    KeypointAwareCrop,
)
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.runners import (
    build_inference_runner,
    InferenceRunner,
    Task,
)
from deeplabcut.utils import auxfun_videos


def build_optimizer(optimizer_cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Builds an optimizer from configuration file

    Args:
        optimizer_cfg: the optimizer configuration
        model: the model to optimize

    Returns:
        the optimizer
    """
    get_optimizer = getattr(torch.optim, optimizer_cfg["type"])
    return get_optimizer(params=model.parameters(), **optimizer_cfg["params"])


def build_transforms(aug_cfg: dict, augment_bbox: bool = False) -> A.BaseCompose:
    """
    Returns the transformation pipeline based on config

    Args:
        aug_cfg : dict containing all transforms information
        augment_bbox : whether the returned augmentation pipelines should keep track of bboxes or not
    Returns:
        transform: callable element that can augment images, keypoints and bboxes
    """
    transforms = []

    crop_sampling = aug_cfg.get("crop_sampling", False)
    if crop_sampling:
        # Add smart, keypoint-aware image cropping
        transforms.append(
            A.PadIfNeeded(
                min_height=crop_sampling["height"],
                min_width=crop_sampling["width"],
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        )
        transforms.append(
            KeypointAwareCrop(
                crop_sampling["width"],
                crop_sampling["height"],
                crop_sampling["max_shift"],
                crop_sampling["method"],
            )
        )

    if resize_aug := aug_cfg.get("resize", False):
        transforms += build_resize_transforms(resize_aug)

    if aug_cfg.get("hflip"):
        warnings.warn(
            "Be careful! Do not train pose models with horizontal flips if you have"
            " symmetric keypoints!"
        )
        hflip_proba = 0.5
        if isinstance(aug_cfg["hflip"], float):
            hflip_proba = aug_cfg["hflip"]
        transforms.append(A.HorizontalFlip(p=hflip_proba))

    # TODO code again this augmentation to match the symmetric_pair syntax in original dlc
    # if aug_cfg.get('flipr', False) and aug_cfg.get('symmetric_pair', False):
    #     opt = aug_cfg.get("fliplr", False)
    #     if type(opt) == int:
    #         p = opt
    #     else:
    #         p = 0.5
    #     transforms.append(
    #         CustomHorizontalFlip(

    #             symmetric_pairs = aug_cfg['symmetric_pairs'],
    #             p=p
    #         )
    #     )
    scale_jitter_lo, scale_jitter_up = aug_cfg.get("scale_jitter", (1, 1))
    transforms.append(A.Affine(scale=(scale_jitter_lo, scale_jitter_up), p=1))
    if rotation := aug_cfg.get("rotation", 0) != 0:
        transforms.append(
            A.Affine(
                rotate=(-rotation, rotation),
                p=0.5,
            )
        )
    if aug_cfg.get("hist_eq", False):
        transforms.append(A.Equalize(p=0.5))
    if aug_cfg.get("motion_blur", False):
        transforms.append(A.MotionBlur(p=0.5))
    if aug_cfg.get("covering", False):
        transforms.append(
            CoarseDropout(
                max_holes=10,
                max_height=0.05,
                min_height=0.01,
                max_width=0.05,
                min_width=0.01,
                p=0.5,
            )
        )
    if aug_cfg.get("elastic_transform", False):
        transforms.append(ElasticTransform(sigma=5, p=0.5))
    if aug_cfg.get("grayscale", False):
        transforms.append(Grayscale(alpha=(0.5, 1.0)))
    if aug_cfg.get("gaussian_noise", False):
        opt = aug_cfg.get("gaussian_noise", False)  # std
        # TODO inherit custom gaussian transform to support per_channel = 0.5
        if type(opt) == int or type(opt) == float:
            transforms.append(
                A.GaussNoise(
                    var_limit=(0, opt**2),
                    mean=0,
                    per_channel=True,  # Albumentations doesn't support per_channel = 0.5
                    p=0.5,
                )
            )
        else:
            transforms.append(
                A.GaussNoise(
                    var_limit=(0, (0.05 * 255) ** 2), mean=0, per_channel=True, p=0.5
                )
            )

    if aug_cfg.get("auto_padding"):
        transforms.append(build_auto_padding(**aug_cfg["auto_padding"]))

    if aug_cfg.get("normalize_images"):
        transforms.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    bbox_params = None
    if augment_bbox:
        bbox_params = A.BboxParams(format="coco", label_fields=["bbox_labels"])

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            "xy", remove_invisible=False, label_fields=["class_labels"]
        ),
        bbox_params=bbox_params,
    )


def build_inference_transform(
    transform_cfg: dict, augment_bbox: bool = True
) -> A.BasicTransform | A.BaseCompose:
    """Build transform pipeline for inference

    Mainly about normalising the images a giving them a specific shape

    Args:
        transform_cfg (dict): dict containing information about the transforms to apply
            should be the same as the one used for build_transforms to ensure matching
            distributions between train and test
        augment_bbox (bool): should always be True for inference

    Returns:
        Union[A.BasicTransform, A.BaseCompose]: the transformation pipeline
    """
    list_transforms = []
    if resize_aug := transform_cfg.get("resize"):
        list_transforms += build_resize_transforms(resize_aug)

    if transform_cfg.get("auto_padding"):
        list_transforms.append(build_auto_padding(**transform_cfg["auto_padding"]))

    if transform_cfg.get("normalize_images"):
        list_transforms.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    bbox_params = None
    if augment_bbox:
        bbox_params = A.BboxParams(format="coco", label_fields=["bbox_labels"])

    return A.Compose(
        list_transforms,
        keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        bbox_params=bbox_params,
    )


def get_model_snapshots(model_folder: Path) -> list[Path]:
    """
    Assumes that all snapshots are named using the pattern "snapshot-{idx}.pt"

    Args:
        model_folder: the path to the folder containing the snapshots

    Returns:
        the paths of snapshots in the folder, sorted by index in ascending order
    """
    return sorted(
        [
            file
            for file in model_folder.iterdir()
            if ((file.suffix == ".pt") and ("detector" not in str(file)))
        ],
        key=lambda p: int(p.stem.split("-")[-1]),
    )


def get_detector_snapshots(model_folder: Path) -> list[Path]:
    """
    Assumes that all snapshots are named using the pattern "detector-snapshot-{idx}.pt"

    Args:
        model_folder: the path to the folder containing the snapshots

    Returns:
        the paths of detector snapshots in the folder, sorted by index in ascending order
    """
    return sorted(
        [
            file
            for file in model_folder.iterdir()
            if ((file.suffix == ".pt") and ("detector" in str(file)))
        ],
        key=lambda p: int(p.stem.split("-")[-1]),
    )


def list_videos_in_folder(
    data_path: str | list[str], video_type: str | None
) -> list[Path]:
    """
    TODO
    """
    video_path = Path(data_path)
    if video_path.is_dir():
        if video_type is None:
            video_suffixes = ["." + ext for ext in auxfun_videos.SUPPORTED_VIDEOS]
        else:
            video_suffixes = [video_type]

        video_suffixes = [s if s.startswith(".") else "." + s for s in video_suffixes]
        return [file for file in video_path.iterdir() if file.suffix in video_suffixes]

    assert (
        video_path.exists()
    ), f"Could not find the video: {video_path}. Check access rights."
    return [video_path]


def build_auto_padding(
    min_height: int | None = None,
    min_width: int | None = None,
    pad_height_divisor: int | None = 1,
    pad_width_divisor: int | None = 1,
    position: str = "random",  # TODO: Which default to set?
    border_mode: str = "reflect_101",  # TODO: Which default to set?
    border_value: float | None = None,
    border_mask_value: float | None = None,
) -> A.PadIfNeeded:
    """
    Create an albumentations PadIfNeeded transform from a config

    Args:
        min_height: the minimum height of the image
        min_width: the minimum width of the image
        pad_height_divisor: if not None, ensures height is dividable by value of this argument
        pad_width_divisor: if not None, ensures width is dividable by value of this argument
        position: position of the image, one of the possible PadIfNeeded
        border_mode: 'constant' or 'reflect_101' (see cv2.BORDER modes)
        border_value: padding value if border_mode is 'constant'
        border_mask_value: padding value for mask if border_mode is 'constant'

    Raises:
        ValueError:
            Only one of 'min_height' and 'pad_height_divisor' parameters must be set
            Only one of 'min_width' and 'pad_width_divisor' parameters must be set

    Returns:
        the auto-padding transform
    """
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect_101": cv2.BORDER_REFLECT_101,
    }
    if border_mode not in border_modes:
        raise ValueError(
            f"Unknown border mode for auto_padding: {border_mode} "
            f"(valid values are: {border_modes.keys()})"
        )

    return A.PadIfNeeded(
        min_height=min_height,
        min_width=min_width,
        pad_height_divisor=pad_height_divisor,
        pad_width_divisor=pad_width_divisor,
        position=position,
        border_mode=border_modes[border_mode],
        value=border_value,
        mask_value=border_mask_value,
    )


def build_resize_transforms(resize_cfg: dict) -> list[A.BasicTransform]:
    height, width = resize_cfg["height"], resize_cfg["width"]

    transforms = []
    if resize_cfg.get("keep_ratio", True):
        transforms.append(KeepAspectRatioResize(width=width, height=height, mode="pad"))
        transforms.append(
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                position=A.PadIfNeeded.PositionType.TOP_LEFT,
            )
        )
    else:
        transforms.append(A.Resize(height, width))

    return transforms


def ensure_multianimal_df_format(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to 'multianimal' format (with an "individuals" columns index)

    Args:
        df_predictions: the dataframe to convert

    Returns:
        the dataframe in MA format
    """
    df_predictions_ma = df_predictions.copy()
    try:
        df_predictions_ma.columns.get_level_values("individuals").unique().tolist()
    except KeyError:
        new_cols = pd.MultiIndex.from_tuples(
            [(col[0], "animal", col[1], col[2]) for col in df_predictions_ma.columns],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        df_predictions_ma.columns = new_cols
    return df_predictions_ma


def build_predictions_dataframe(
    scorer: str,
    predictions: dict[str, dict[str, np.ndarray]],
    parameters: PoseDatasetParameters,
    image_name_to_index: Callable[[str], tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """

    Args:
        scorer:
        predictions
        parameters:
        image_name_to_index:

    Returns:

    """
    kpt_entries = ["x", "y", "likelihood"]
    col_names = ["scorer", "individuals", "bodyparts", "coords"]

    col_values = []
    for i in parameters.individuals:
        for b in parameters.bodyparts:
            col_values += [(scorer, i, b, entry) for entry in kpt_entries]
    for unique_bpt in parameters.unique_bpts:
        col_values += [(scorer, "single", unique_bpt, entry) for entry in kpt_entries]

    prediction_data = []
    index_data = []
    for image, image_predictions in predictions.items():
        image_data = image_predictions["bodyparts"][..., :3].reshape(-1)
        if "unique_bodyparts" in image_predictions:
            image_data = np.concatenate(
                [image_data, image_predictions["unique_bodyparts"][..., :3].reshape(-1)]
            )

        prediction_data.append(image_data)
        if image_name_to_index is not None:
            index_data.append(image_name_to_index(image))

    if len(index_data) > 0:
        index = pd.MultiIndex.from_tuples(index_data)
    else:
        index = list(predictions.keys())

    return pd.DataFrame(
        prediction_data,
        index=index,
        columns=pd.MultiIndex.from_tuples(col_values, names=col_names),
    )


def get_runners(
    pytorch_config: dict,
    snapshot_path: str,
    max_individuals: int,
    num_bodyparts: int,
    num_unique_bodyparts: int,
    with_identity: bool = False,
    transform: A.BaseCompose | None = None,
    detector_path: str | None = None,
    detector_transform: A.BaseCompose | None = None,
) -> tuple[InferenceRunner, InferenceRunner | None]:
    """Builds the runners for pose estimation

    Args:
        pytorch_config: the pytorch configuration file
        snapshot_path: the path of the snapshot from which to load the weights
        max_individuals: the maximum number of individuals per image
        num_bodyparts: the number of bodyparts predicted by the model
        num_unique_bodyparts: the number of unique_bodyparts predicted by the model
        with_identity: whether the pose model has an identity head
        transform: the transform for pose estimation. if None, uses the transform
            defined in the config.
        detector_path: the path to the detector snapshot from which to load weights,
            for top-down models (if a detector runner is needed)
        detector_transform: the transform for object detection. if None, uses the
            transform defined in the config.

    Returns:
        a runner for pose estimation
        a runner for detection, if detector_path is not None
    """
    pose_task = Task(pytorch_config.get("method", "bu"))
    device = pytorch_config["device"]
    if transform is None:
        transform = build_inference_transform(pytorch_config["data"])

    detector_runner = None
    if pose_task == Task.BOTTOM_UP:
        pose_preprocessor = build_bottom_up_preprocessor(
            color_mode="RGB", transform=transform  # TODO: read from Loader
        )
        pose_postprocessor = build_bottom_up_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
            with_identity=with_identity,
        )
    else:
        pose_preprocessor = build_top_down_preprocessor(
            color_mode="RGB",  # TODO: read from Loader
            transform=transform,
            cropped_image_size=(256, 256),
        )
        pose_postprocessor = build_top_down_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
        )

        if detector_path is not None:
            if detector_transform is None:
                detector_transform = build_inference_transform(
                    pytorch_config["data_detector"]
                )

            detector_runner = build_inference_runner(
                task=Task.DETECT,
                model=DETECTORS.build(pytorch_config["detector"]["model"]),
                device=device,
                snapshot_path=detector_path,
                preprocessor=build_bottom_up_preprocessor(
                    color_mode="RGB",  # TODO: read from Loader
                    transform=detector_transform,
                ),
                postprocessor=build_detector_postprocessor(
                    max_individuals=max_individuals,
                ),
            )

    pose_runner = build_inference_runner(
        task=pose_task,
        model=PoseModel.build(pytorch_config["model"]),
        device=device,
        snapshot_path=snapshot_path,
        preprocessor=pose_preprocessor,
        postprocessor=pose_postprocessor,
    )
    return pose_runner, detector_runner
