import imgaug.augmenters as iaa
import numpy as np
from scipy.spatial.distance import pdist, squareform


class KeypointAwareCropsToFixedSize(iaa.CropToFixedSize):
    def __init__(self, width, height, n_crops=10, max_shift=0.40):
        """
        Parameters
        ----------
        width : int
            Crop images down to this maximum width.

        height : int
            Crop images down to this maximum height.

        n_crops : int, optional (default=10)
            Number of crops to produce.

        max_shift : float, optional (default=0.25)
            Maximum allowed shift of the cropping center position
            as a fraction of the crop size.
        """
        super(KeypointAwareCropsToFixedSize, self).__init__(width, height)
        self.n_crops = n_crops
        # Clamp to 40% of crop size to ensure that at least
        # the center keypoint remains visible after the offset is applied.
        self.max_shift = max(0., min(max_shift, 0.4))

    def augment_batch_(self, batch, parents=None, hooks=None):
        try:
            batch = SizeVaryingUnnormalizedBatch(
                images=batch.images_unaug,
                keypoints=batch.keypoints_unaug,
            )
        except AttributeError:
            pass
        return super(KeypointAwareCropsToFixedSize, self).augment_batch_(
            batch, parents, hooks,
        )

    @staticmethod
    def calc_n_neighbors(xy, radius):
        d = pdist(xy, 'sqeuclidean')
        mat = squareform(d <= radius * radius, checks=False)
        return np.sum(mat, axis=0)

    def _draw_samples(self, batch, random_state):
        rngs = random_state.duplicate(2)
        offsets = []
        max_shift_x = self.max_shift * self.size[0]
        max_shift_y = self.max_shift * self.size[1]
        for n in range(batch.nb_rows):
            h, w = batch.images[n].shape[:2]
            kpts = batch.keypoints[n].to_xy_array()
            inds = np.arange(kpts.shape[0])
            # Points with a higher number of neighbors are sampled preferentially
            # to favor the augmentation of denser (harder) scene regions.
            radius = 0.1 * min(h, w)
            n_neighbors = self.calc_n_neighbors(kpts, radius)
            p = n_neighbors / n_neighbors.sum()
            centers = kpts[random_state.choice(inds, self.n_crops, p=p)]
            centers[:, 0] += max_shift_x * rngs[0].uniform(-1, 1, self.n_crops)
            centers[:, 0] /= w
            centers[:, 1] += max_shift_y * rngs[1].uniform(-1, 1, self.n_crops)
            centers[:, 1] /= h
            offsets.append(centers)
        offsets = np.clip(np.stack(offsets), 0, 1)
        return [self.size] * batch.nb_rows, offsets

    def _augment_batch_(self, batch, random_state, parents, hooks):
        images = []
        keypoints = []
        sizes, offsets = self._draw_samples(batch, random_state)
        for i, (image, kpts, (w, h)) in enumerate(
            zip(batch.images, batch.keypoints, sizes)
        ):
            height_image, width_image = image.shape[:2]
            for x, y in offsets[i].tolist():
                croppings = self._calculate_crop_amounts(
                    height_image, width_image, h, w, y, x,
                )
                image_cropped = iaa.size._crop_and_pad_arr(
                    image,
                    croppings,
                    paddings=(0, 0, 0, 0),
                    keep_size=False,
                )
                # Deepcopy to avoid shifting points in place
                kpts_cropped = iaa.size._crop_and_pad_kpsoi_(
                    kpts.deepcopy(),
                    croppings,
                    paddings_img=(0, 0, 0, 0),
                    keep_size=False,
                )
                images.append(image_cropped)
                keypoints.append(kpts_cropped)
        batch.images = images
        batch.keypoints = keypoints
        return batch


class Sequential(iaa.Sequential):
    def augment_batch_(self, batch, parents=None, hooks=None):
        try:
            batch = SizeVaryingUnnormalizedBatch(
                images=batch.images_unaug,
                keypoints=batch.keypoints_unaug,
                heatmaps=batch.heatmaps_unaug,
                segmentation_maps=batch.segmentation_maps_unaug,
                bounding_boxes=batch.bounding_boxes_unaug,
                polygons=batch.polygons_unaug,
                line_strings=batch.line_strings_unaug,
            )
        except AttributeError:
            pass
        return super(Sequential, self).augment_batch_(
            batch, parents, hooks,
        )


class SizeVaryingUnnormalizedBatch(iaa.UnnormalizedBatch):
    def fill_from_augmented_normalized_batch_(self, batch_aug_norm):
        super(SizeVaryingUnnormalizedBatch, self).fill_from_augmented_normalized_batch_(
            batch_aug_norm,
        )
        self.images_aug = batch_aug_norm.images_aug
        return self
