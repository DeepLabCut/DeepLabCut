from pathlib import Path
from typing import Dict, List, Optional, Union

import albumentations as A
import torch
import yaml
from deeplabcut.utils import auxfun_videos

from deeplabcut.pose_estimation_pytorch.models import (
    PoseModel,
    BACKBONES,
    NECKS,
    HEADS,
    LOSSES,
    DETECTORS,
)
from deeplabcut.pose_estimation_pytorch.solvers import LOGGER, SOLVERS
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS
from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    TARGET_GENERATORS,
)
from deeplabcut.pose_estimation_pytorch.solvers.schedulers import LRListScheduler
from deeplabcut.pose_estimation_pytorch.solvers.base import Solver


def build_pose_model(cfg: Dict, pytorch_cfg: Dict) -> PoseModel:
    """
        Returns a pytorch pose model based on pytorch config

    Args:
        cfg : sub dict of the pytorch config that contains all information about the model
        pytorch_cfg : entire pytorch config"""

    # TODO not sure why exactly we would need those two dicts as entries
    backbone = BACKBONES.build(dict(cfg["backbone"]))
    heads = []
    for head_config in cfg["heads"]:
        heads.append(HEADS.build(dict(head_config)))
    target_generator = TARGET_GENERATORS.build(dict(cfg["target_generator"]))
    if cfg.get("neck"):
        neck = NECKS.build(dict(cfg["neck"]))
    else:
        neck = None
    pose_model = PoseModel(
        cfg=pytorch_cfg,
        backbone=backbone,
        heads=heads,
        target_generator=target_generator,
        neck=neck,
        **cfg["pose_model"],
    )

    return pose_model


def build_detector(detector_cfg: Dict):
    """Builds detector related objects : detector, its optimizer and its scheduler

    Args:
        detector_cfg (Dict): detector config dictionary

    Returns:
        detector, detector_optimizer, detector_scheduler
    """
    detector = DETECTORS.build(detector_cfg["detector_model"])

    get_optimizer = getattr(torch.optim, detector_cfg["detector_optimizer"]["type"])
    detector_optimizer = get_optimizer(
        params=detector.parameters(), **detector_cfg["detector_optimizer"]["params"]
    )

    if detector_cfg.get("detector_scheduler"):
        if detector_cfg["detector_scheduler"]["type"] == "LRListScheduler":
            _scheduler = LRListScheduler
        else:
            _scheduler = getattr(
                torch.optim.lr_scheduler, detector_cfg["detector_scheduler"]["type"]
            )
        detector_scheduler = _scheduler(
            optimizer=detector_optimizer, **detector_cfg["detector_scheduler"]["params"]
        )
    else:
        detector_scheduler = None

    return detector, detector_optimizer, detector_scheduler


def build_solver(pytorch_cfg: Dict) -> Solver:
    """
        Build the solver object to run training

    Args:
        pytorch_cfg: config dictionary to build the solver
    Returns:
        solver : solver to train the model
    """
    pose_model = build_pose_model(pytorch_cfg["model"], pytorch_cfg)

    get_optimizer = getattr(torch.optim, pytorch_cfg["optimizer"]["type"])
    optimizer = get_optimizer(
        params=pose_model.parameters(), **pytorch_cfg["optimizer"]["params"]
    )

    criterion = LOSSES.build(pytorch_cfg["criterion"])

    predictor = PREDICTORS.build(dict(pytorch_cfg["predictor"]))

    if pytorch_cfg.get("scheduler"):
        if pytorch_cfg["scheduler"]["type"] == "LRListScheduler":
            _scheduler = LRListScheduler
        else:
            _scheduler = getattr(
                torch.optim.lr_scheduler, pytorch_cfg["scheduler"]["type"]
            )
        scheduler = _scheduler(
            optimizer=optimizer, **pytorch_cfg["scheduler"]["params"]
        )
    else:
        scheduler = None

    if pytorch_cfg.get("logger"):
        logger = LOGGER.build(dict(**pytorch_cfg["logger"], model=pose_model))
    else:
        logger = None

    if pytorch_cfg.get("method", "bu") == "bu":
        solver = SOLVERS.build(
            dict(
                **pytorch_cfg["solver"],
                model=pose_model,
                criterion=criterion,
                optimizer=optimizer,
                predictor=predictor,
                cfg=pytorch_cfg,
                device=pytorch_cfg["device"],
                scheduler=scheduler,
                logger=logger,
            )
        )
    elif pytorch_cfg.get("method", "bu") == "td":
        detector, detector_optimizer, detector_scheduler = build_detector(
            pytorch_cfg["detector"]
        )

        solver = SOLVERS.build(
            dict(
                **pytorch_cfg["solver"],
                model=pose_model,
                criterion=criterion,
                optimizer=optimizer,
                predictor=predictor,
                cfg=pytorch_cfg,
                device=pytorch_cfg["device"],
                scheduler=scheduler,
                logger=logger,
                detector=detector,
                detector_optimizer=detector_optimizer,
                detector_scheduler=detector_scheduler,
            )
        )
    else:
        raise ValueError(
            "The method in your pytorch config is invalid, possible values are "
            "'bu' (Bottom Up) or 'td' (Top Down)."
        )
    return solver


def build_transforms(
    aug_cfg: dict, augment_bbox: bool = False
) -> Union[A.BasicTransform, A.BaseCompose]:
    """
    Returns the transformation pipeline based on config

    Args:
        aug_cfg : dict containing all transforms information
        augment_bbox : whether the returned augmentation pipelines should keep track of bboxes or not
    Returns:
        transform: callable element that can augment images, keypoints and bboxes
    """
    transforms = []

    if aug_cfg.get("resize", False):
        input_size = aug_cfg.get("resize", False)
        transforms.append(A.Resize(input_size[0], input_size[1]))

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
    rotation = aug_cfg.get("rotation", 0)
    translation = aug_cfg.get("translation", 0)
    transforms.append(
        A.Affine(
            scale=(scale_jitter_lo, scale_jitter_up),
            rotate=(-rotation, rotation),
            translate_px=(-translation, translation),
            p=0.5,
        )
    )
    if aug_cfg.get("hist_eq", False):
        transforms.append(A.Equalize(p=0.5))
    if aug_cfg.get("motion_blur", False):
        transforms.append(A.MotionBlur(p=0.5))
    # TODO Coarse dropout can mask a keypoint which messes up the training, implement new augmentation
    # if aug_cfg.get('covering', False):
    #     transforms.append(
    #         A.CoarseDropout(
    #             max_holes=10,
    #             max_height=0.05,
    #             min_height=0.01,
    #             max_width=0.05,
    #             min_width=0.01,
    #             p=0.5
    #         )
    #     )
    # TODO implement elastic transform apply_to_keypoints in albumentations
    # if aug_cfg.get('elastic_transform', False):
    #     transforms.append(A.ElasticTransform(sigma=5, p=0.5))
    # TODO implement iia grayscale augmentation with albumentation
    # if aug_cfg.get('grayscale', False):
    if aug_cfg.get("gaussian_noise", False):
        opt = aug_cfg.get("gaussian_noise", False)  # std
        # TODO inherit custom gaussian transform to support per_channel = 0.5
        if type(opt) == int or type(opt) == float:
            transforms.append(
                A.GaussNoise(
                    var_limit=(0, opt**2),
                    mean=0,
                    per_channel=True,  # Albumentations doesn't support per_cahnnel = 0.5
                    p=0.5,
                )
            )
        else:
            transforms.append(
                A.GaussNoise(
                    var_limit=(0, (0.05 * 255) ** 2),
                    mean=0,
                    per_channel=True,
                    p=0.5,
                )
            )

    if aug_cfg.get("auto_padding"):
        params = aug_cfg.get("auto_padding")
        pad_height_divisor = params.get("pad_height_divisor", 1)
        pad_width_divisor = params.get("pad_width_divisor", 1)
        transforms.append(
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=pad_height_divisor,
                pad_width_divisor=pad_width_divisor,
            )
        )
    if aug_cfg.get("normalize_images"):
        transforms.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    if augment_bbox:
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams("xy", remove_invisible=False),
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
        )
    else:
        return A.Compose(
            transforms, keypoint_params=A.KeypointParams("xy", remove_invisible=False)
        )


def build_inference_transform(
    transform_cfg: dict, augment_bbox: bool = True
) -> Union[A.BasicTransform, A.BaseCompose]:
    """Build transform pipeline for inference

    Mainly about normalising the images a giving them a specific shape

    Args:
        transform_cfg (dict): dict containing information about the transforms to apply
                                should be the same as the one used for build_transforms to
                                ensure matching distributions between train and test
        augment_bbox (bool): should always be True for inference

    Returns:
        Union[A.BasicTransform, A.BaseCompose]: the transformation pipeline
    """

    list_transforms = []
    if transform_cfg.get("normalize_images"):
        list_transforms.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    if augment_bbox:
        return A.Compose(
            list_transforms,
            keypoint_params=A.KeypointParams("xy", remove_invisible=False),
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
        )
    else:
        return A.Compose(
            list_transforms,
            keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        )


def read_yaml(path):
    with open(path) as f:
        file = yaml.safe_load(f)
    return file


def get_model_snapshots(model_folder: Path) -> List[Path]:
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


def get_detector_snapshots(model_folder: Path) -> List[Path]:
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


def videos_in_folder(
    data_path: Union[str, List[str]],
    video_type: Optional[str],
) -> List[Path]:
    """
    TODO
    """
    video_path = Path(data_path)
    if video_path.is_dir():
        if video_type is None:
            video_suffixes = auxfun_videos.SUPPORTED_VIDEOS
        else:
            video_suffixes = [video_type]

        return [
            file for file in video_path.iterdir() if video_path.stem in video_suffixes
        ]

    assert (
        video_path.exists()
    ), f"Could not find the video: {video_path}. Check access rights."
    return [video_path]


def update_config_parameters(pytorch_config: dict, **kwargs) -> None:
    """
    Overwrites the pytorch config dictionary to correspond to the command line input keys

    Args:
        pytorch_config
        **kwargs : any arguments that can be found as entry for the pytorch config

    Return:
        None
    """
    for key in kwargs.keys():
        pytorch_config[key] = kwargs[key]

    return
