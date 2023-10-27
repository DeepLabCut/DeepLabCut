from __future__ import annotations

import numpy as np
from albumentations.augmentations.crops import RandomCrop
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from typing import Any


class KeypointAwareCrop(RandomCrop):
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

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "width", "height", "max_shift", "crop_sampling"
