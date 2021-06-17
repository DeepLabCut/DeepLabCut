import imgaug.augmenters as iaa
import numpy as np
from scipy.spatial.distance import pdist, squareform


class KeypointAwareCropToFixedSize(iaa.CropToFixedSize):
    def __init__(self, width, height, max_shift=0.4):
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
        """
        super(KeypointAwareCropToFixedSize, self).__init__(
            width, height, name="kptscrop",
        )
        # Clamp to 40% of crop size to ensure that at least
        # the center keypoint remains visible after the offset is applied.
        self.max_shift = max(0., min(max_shift, 0.4))

    @staticmethod
    def calc_n_neighbors(xy, radius):
        d = pdist(xy, 'sqeuclidean')
        mat = squareform(d <= radius * radius, checks=False)
        return np.sum(mat, axis=0)

    def _draw_samples(self, batch, random_state):
        n_samples = batch.nb_rows
        offsets = np.empty((n_samples, 2), dtype=np.float32)
        rngs = random_state.duplicate(2)
        shift_x = self.max_shift * self.size[0] * rngs[0].uniform(-1, 1, n_samples)
        shift_y = self.max_shift * self.size[1] * rngs[1].uniform(-1, 1, n_samples)
        for n in range(batch.nb_rows):
            h, w = batch.images[n].shape[:2]
            kpts = batch.keypoints[n].to_xy_array()
            inds = np.arange(kpts.shape[0])
            # Points located close to one another are sampled preferentially
            # in order to augment crowded regions.
            radius = 0.1 * min(h, w)
            n_neighbors = self.calc_n_neighbors(kpts, radius)
            # Include keypoints in the count to avoid null probabilities
            n_neighbors += 1
            p = n_neighbors / n_neighbors.sum()
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
    aug = pipeline.find_augmenters_by_name("kptscrop")[0]
    aug.size = width, height
