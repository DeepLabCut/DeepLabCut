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
import imgaug.augmenters as iaa
import numpy as np
from imgaug import KeypointsOnImage
from scipy.spatial.distance import pdist, squareform
from typing import List, Union, Tuple


class KeypointFliplr(iaa.Fliplr):
    def __init__(
        self,
        keypoints: List[str],
        symmetric_pairs: List[Union[Tuple, List]],
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.keypoints = keypoints
        self.symmetric_pairs = symmetric_pairs

    @property
    def n_keypoints(self):
        return len(self.keypoints)

    def _augment_batch_(self, batch, random_state, parents, hooks):
        batch = super()._augment_batch_(batch, random_state, parents, hooks)
        keypoints = []
        for kpts in batch.keypoints:
            kpts_ = list(kpts)
            n_kpts = len(kpts_)
            for i1, i2 in self.symmetric_pairs:
                for j in range(0, n_kpts, self.n_keypoints):
                    kpts_[i1 + j], kpts_[i2 + j] = kpts_[i2 + j], kpts_[i1 + j]
            keypoints.append(KeypointsOnImage(kpts_, kpts.shape))
        batch.keypoints = keypoints
        return batch


class KeypointAwareCropToFixedSize(iaa.CropToFixedSize):
    def __init__(
        self,
        width,
        height,
        max_shift=0.4,
        crop_sampling="hybrid",
    ):
        """
        Parameters
        ----------
        width : int
            Crop images down to this maximum width.

        height : int
            Crop images down to this maximum height.

        max_shift : float, optional (default=0.25)
            Maximum allowed shift of the cropping center position
            as a fraction of the crop size.

        crop_sampling : str, optional (default="hybrid")
            Crop centers sampling method. Must be either:
            "uniform" (randomly over the image),
            "keypoints" (randomly over the annotated keypoints),
            "density" (weighing preferentially dense regions of keypoints),
            or "hybrid" (alternating randomly between "uniform" and "density").
        """
        super(KeypointAwareCropToFixedSize, self).__init__(
            width,
            height,
            name="kptscrop",
        )
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
    def calc_n_neighbors(xy, radius):
        d = pdist(xy, "sqeuclidean")
        mat = squareform(d <= radius * radius, checks=False)
        return np.sum(mat, axis=0)

    def _draw_samples(self, batch, random_state):
        n_samples = batch.nb_rows
        offsets = np.empty((n_samples, 2), dtype=np.float32)
        rngs = random_state.duplicate(2)
        shift_x = self.max_shift * self.size[0] * rngs[0].uniform(-1, 1, n_samples)
        shift_y = self.max_shift * self.size[1] * rngs[1].uniform(-1, 1, n_samples)
        sampling = self.crop_sampling
        for n in range(batch.nb_rows):
            if self.crop_sampling == "hybrid":
                sampling = random_state.choice(["uniform", "density"])
            if sampling == "uniform":
                center = random_state.uniform(size=2)
            else:
                h, w = batch.images[n].shape[:2]
                kpts = batch.keypoints[n].to_xy_array()
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
                center = kpts[random_state.choice(inds, p=p)]
                # Shift the crop center in both dimensions by random amounts
                # and normalize to the original image dimensions.
                center[0] += shift_x[n]
                center[0] /= w
                center[1] += shift_y[n]
                center[1] /= h
            offsets[n] = center
        offsets = np.clip(offsets, 0, 1)
        return [self.size] * n_samples, offsets[:, 0], offsets[:, 1]


def update_crop_size(pipeline, width, height):
    aug = pipeline.find_augmenters_by_name("kptscrop")
    if not aug:
        return
    aug[0].size = width, height
