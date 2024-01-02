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

from typing import Any, Iterable, Sequence

import albumentations as A
import cv2
import numpy as np
import warnings
from albumentations.augmentations.geometric import functional as F
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform


class KeypointAwareCrop(A.RandomCrop):
    def __init__(
        self,
        width: int,
        height: int,
        max_shift: float = 0.4,
        crop_sampling: str = "hybrid",
    ):
        """
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
            kpts = np.asarray(kpts)[:, :2]
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
        alpha: float | int | tuple[float] = 1.0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        """
        Args:
            alpha: int, float or tuple of floats, optional
            The alpha value of the new colorspace when overlayed over the
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
        alpha_affine: float = 0.0,  # Deactivate affine transformation prior to elastic deformation
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
        max_height: int = 8,
        max_width: int = 8,
        min_holes: int | None = None,
        min_height: int | None = None,
        min_width: int | None = None,
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
