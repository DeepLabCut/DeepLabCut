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

import warnings
from typing import Any, Iterable, Sequence

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as F
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform


def build_transforms(augmentations: dict) -> A.BaseCompose:
    transforms = []

    if crop_sampling := augmentations.get("crop_sampling"):
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

    if resize_aug := augmentations.get("resize", False):
        transforms += build_resize_transforms(resize_aug)

    if (lms_cfg := augmentations.get("longest_max_size")) is not None:
        transforms.append(A.LongestMaxSize(lms_cfg))

    if hflip_cfg := augmentations.get("hflip"):
        hflip_proba = 0.5
        symmetries = None
        if isinstance(hflip_cfg, float):
            hflip_proba = hflip_cfg
        elif isinstance(hflip_cfg, dict):
            if "p" in hflip_cfg:
                hflip_proba = float(hflip_cfg["p"])

            if "symmetries" in hflip_cfg:
                symmetries = []
                for kpt_a, kpt_b in hflip_cfg["symmetries"]:
                    symmetries.append((int(kpt_a), int(kpt_b)))

        if symmetries is not None:
            transforms.append(HFlip(symmetries=symmetries, p=hflip_proba))
        else:
            warnings.warn(
                "Be careful! Do not train pose models with horizontal flips if you have"
                " symmetric keypoints!"
            )
            transforms.append(A.HorizontalFlip(p=hflip_proba))

    if (affine := augmentations.get("affine")) is not None:
        scaling = affine.get("scaling")
        rotation = affine.get("rotation")
        translation = affine.get("translation")
        if rotation is not None:
            rotation = (-rotation, rotation)
        if translation is not None:
            translation = (0, translation)

        transforms.append(
            A.Affine(
                scale=scaling,
                rotate=rotation,
                translate_px=translation,
                p=affine.get("p", 0.9),
                keep_ratio=True,
            )
        )

    if augmentations.get("hist_eq", False):
        transforms.append(A.Equalize(p=0.5))
    if augmentations.get("motion_blur", False):
        transforms.append(A.MotionBlur(p=0.5))
    if augmentations.get("covering", False):
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
    if augmentations.get("elastic_transform", False):
        transforms.append(ElasticTransform(sigma=5, p=0.5))
    if augmentations.get("grayscale", False):
        transforms.append(Grayscale(alpha=(0.5, 1.0)))
    if noise := augmentations.get("gaussian_noise", False):
        # TODO inherit custom gaussian transform to support per_channel = 0.5
        if not isinstance(noise, (int, float)):
            noise = 0.05 * 255
        transforms.append(
            A.GaussNoise(
                var_limit=(0, noise ** 2),
                mean=0,
                per_channel=True,
                # Albumentations doesn't support per_channel = 0.5
                p=0.5,
            )
        )

    if augmentations.get("auto_padding"):
        transforms.append(build_auto_padding(**augmentations["auto_padding"]))

    if augmentations.get("normalize_images"):
        transforms.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            "xy", remove_invisible=False, label_fields=["class_labels"]
        ),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )


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


class HFlip(A.HorizontalFlip):
    """Horizontal Flip which swaps symmetric keypoints"""

    def __init__(self, symmetries: list[tuple[int, int]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._symmetries = {}
        for i, j in symmetries:
            self._symmetries[i] = j
            self._symmetries[j] = i

    def apply_to_keypoints(self, keypoints, **params):
        swapped_keypoints = [
            keypoints[self._symmetries.get(kpt_idx, kpt_idx)]
            for kpt_idx in range(len(keypoints))
        ]
        return super().apply_to_keypoints(swapped_keypoints, **params)


class KeypointAwareCrop(A.RandomCrop):
    """Random crop for an image around keypoints

    Args:
        width: Crop images down to this maximum width.
        height: Crop images down to this maximum height.
        max_shift: Maximum allowed shift of the cropping center position
            as a fraction of the crop size.
        crop_sampling: Crop centers sampling method. Must be either:
            "uniform" (randomly over the image),
            "keypoints" (randomly over the annotated keypoints),
            "density" (weighing preferentially dense regions of keypoints),
            "hybrid" (alternating randomly between "uniform" and "density").
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_shift: float = 0.4,
        crop_sampling: str = "hybrid",
    ):
        super().__init__(height, width, always_apply=True)
        # Clamp to 40% of crop size to ensure that at least
        # the center keypoint remains visible after the offset is applied.
        self.max_shift = max(0.0, min(max_shift, 0.4))
        if crop_sampling not in ("uniform", "keypoints", "density", "hybrid"):
            raise ValueError(
                f"Invalid sampling {crop_sampling}. Must be "
                f"either 'uniform', 'keypoints', 'density', or 'hybrid."
            )
        self.crop_sampling = crop_sampling

    @staticmethod
    def calc_n_neighbors(xy: NDArray, radius: float) -> NDArray:
        d = pdist(xy, "sqeuclidean")
        mat = squareform(d <= radius * radius, checks=False)
        return np.sum(mat, axis=0)

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", "keypoints"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        kpts = params["keypoints"]
        shift_factors = np.random.random(2)
        shift = self.max_shift * shift_factors * np.array([self.width, self.height])
        sampling = self.crop_sampling
        if self.crop_sampling == "hybrid":
            sampling = np.random.choice(["uniform", "density"])
        if sampling == "uniform":
            center = np.random.random(2)
        else:
            h, w = img.shape[:2]
            kpts = np.array([[k[0], k[1]] for k in kpts])
            kpts = kpts[~np.isnan(kpts).all(axis=1)]
            n_kpts = kpts.shape[0]
            inds = np.arange(n_kpts)
            if sampling == "density":
                # Points located close to one another are sampled preferentially
                # in order to augment crowded regions.
                radius = 0.1 * min(h, w)
                n_neighbors = self.calc_n_neighbors(kpts, radius)
                # Include keypoints in the count to avoid null probabilities
                n_neighbors += 1
                p = n_neighbors / n_neighbors.sum()
            else:
                p = np.ones_like(inds) / n_kpts
            center = kpts[np.random.choice(inds, p=p)]
            # Shift the crop center in both dimensions by random amounts
            # and normalize to the original image dimensions.
            center = (center + shift) / [w, h]
            center = np.clip(center, 0, np.nextafter(1, 0))  # Clip to 1 exclusive
        return {"h_start": center[1], "w_start": center[0]}

    def apply_to_keypoints(
        self,
        keypoints,
        **params,
    ) -> list[float]:
        keypoints = super().apply_to_keypoints(keypoints, **params)
        new_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            if not (0 <= x < self.width and 0 <= y < self.height):
                kp = list(kp)
                kp[:2] = np.nan, np.nan
                kp = tuple(kp)
            new_keypoints.append(kp)
        return new_keypoints

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "width", "height", "max_shift", "crop_sampling"


class KeepAspectRatioResize(A.DualTransform):
    """Resizes images while preserving their aspect ratio

    In 'pad' mode, the image will be rescaled to the largest possible size such that
    it can be padded to the correct size (with PadIfNeeded). So we'll have:
        output_width <= width, output_height <= height

    In 'crop' mode, the image will be rescaled to the smallest possible size such that
    it can be cropped to the correct size (with any random crop you want), so:
        output_width >= width, output_height >= height
    """

    def __init__(
        self,
        width: int,
        height: int,
        mode: str = "pad",
        interpolation: Any = cv2.INTER_LINEAR,
        p: float = 1.0,
        always_apply: bool = True,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.mode = mode
        self.interpolation = interpolation

    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        return A.scale(img, scale, interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, scale=0, **params):
        keypoint = A.keypoint_scale(keypoint, scale, scale)
        return keypoint

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        h, w, _ = params["image"].shape
        if self.mode == "pad":
            scale = min(self.height / h, self.width / w)
        else:
            scale = max(self.height / h, self.width / w)

        return {"scale": scale}

    def get_transform_init_args_names(self):
        return "height", "width", "mode", "interpolation"


class Grayscale(A.ToGray):
    def __init__(
        self,
        alpha: float | int | tuple[float, float] = 1.0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        """
        Args:
            alpha: int, float or tuple of floats, optional
            The alpha value of the new colorspace when overlaid over the
            old one. A value close to 1.0 means that mostly the new
            colorspace is visible. A value close to 0.0 means that mostly the
            old image is visible.

            * If a float, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
        """
        super().__init__(always_apply, p)
        if isinstance(alpha, (float, int)):
            self._alpha = self._validate_alpha(alpha)
        elif isinstance(alpha, tuple):
            if len(alpha) != 2:
                raise ValueError("`alpha` must be a tuple of two numbers.")
            self._alpha = tuple([self._validate_alpha(val) for val in alpha])
        else:
            raise ValueError("")

    @staticmethod
    def _validate_alpha(val: float) -> float:
        if not 0.0 <= val <= 1.0:
            warnings.warn("`alpha` will be clipped to the interval [0.0, 1.0].")
        return min(1.0, max(0.0, val))

    @property
    def alpha(self) -> float:
        if isinstance(self._alpha, float):
            return self._alpha
        return np.random.uniform(*self._alpha)

    def apply(self, img: NDArray, **params) -> NDArray:
        img_gray = super().apply(img, **params)
        alpha = self.alpha
        img_blend = img * (1 - alpha) + img_gray * alpha
        return img_blend.astype(img.dtype)


class ElasticTransform(A.ElasticTransform):
    def __init__(
        self,
        alpha: float = 20.0,
        sigma: float = 5.0,  # As in DLC TF
        alpha_affine: float = 0.0,  # Deactivate affine prior to elastic deformation
        interpolation: int = cv2.INTER_CUBIC,  # As in imgaug
        border_mode: int = cv2.BORDER_CONSTANT,  # As in imgaug
        value: float | None = None,
        mask_value: float | None = None,
        always_apply: bool = False,
        approximate: bool = True,  # Faster by a factor of 2
        same_dxdy: bool = True,  # Here too
        p: float = 0.5,
    ):
        super().__init__(
            alpha,
            sigma,
            alpha_affine,
            interpolation,
            border_mode,
            value,
            mask_value,
            always_apply,
            approximate,
            same_dxdy,
            p,
        )
        self._neighbor_dist = 3
        self._neighbor_dist_square = self._neighbor_dist ** 2

    def apply_to_keypoints(
        self, keypoints: Sequence[float], random_state: int | None = None, **params
    ) -> list[float]:
        heatmaps = np.zeros(
            (params["rows"], params["cols"], len(keypoints)), dtype=np.float32
        )
        grid = np.mgrid[: params["rows"], : params["cols"]].transpose((1, 2, 0))
        kpts = np.array([(k[1], k[0]) for k in keypoints])
        valid_kpts = np.all(kpts > 0.0, axis=1)
        dist = ((grid - kpts[:, None, None]) ** 2).sum(axis=3)
        mask = (dist <= self._neighbor_dist_square) & valid_kpts[:, None, None]
        heatmaps[mask.transpose(1, 2, 0)] = 1

        heatmaps_aug = F.elastic_transform(
            heatmaps,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )

        inds = np.indices(heatmaps_aug.shape[:2])[::-1]
        mask = np.transpose(heatmaps_aug == 1, (2, 0, 1))
        # Let's compute the average, rather than the median, coordinates
        div = np.sum(mask, axis=(1, 2))
        sum_indices = np.sum(inds[:, None] * mask[None], axis=(2, 3)).T
        xy = sum_indices / div[:, None]
        new_keypoints = []
        for kp, new_coords in zip(keypoints, xy):
            kp = list(kp)
            kp[:2] = new_coords
            new_keypoints.append(tuple(kp))
        return new_keypoints


class CoarseDropout(A.CoarseDropout):
    def __init__(
        self,
        max_holes: int = 8,
        max_height: int | float = 8,
        max_width: int | float = 8,
        min_holes: int | None = None,
        min_height: int | float | None = None,
        min_width: int | float | None = None,
        fill_value: int = 0,
        mask_fill_value: int | None = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            max_holes,
            max_height,
            max_width,
            min_holes,
            min_height,
            min_width,
            fill_value,
            mask_fill_value,
            always_apply,
            p,
        )

    def apply_to_bboxes(self, bboxes: Sequence[float], **params) -> list[float]:
        return list(bboxes)

    def apply_to_keypoints(
        self,
        keypoints: Sequence[float],
        holes: Iterable[tuple[int, int, int, int]] = (),
        **params,
    ) -> list[float]:
        new_keypoints = []
        for kp in keypoints:
            in_hole = False
            for hole in holes:
                if self._keypoint_in_hole(kp, hole):
                    in_hole = True
                    break
            if in_hole:
                kp = list(kp)
                kp[:2] = np.nan, np.nan
                kp = tuple(kp)
            new_keypoints.append(kp)
        return new_keypoints

    def _keypoint_in_hole(self, keypoint, hole: tuple[int, int, int, int]) -> bool:
        """Reimplemented from Albumentations as was removed in v1.4.0"""
        x1, y1, x2, y2 = hole
        x, y = keypoint[:2]
        return x1 <= x < x2 and y1 <= y < y2
