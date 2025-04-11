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
"""Modules to dynamically crop individuals out of videos to improve video analysis"""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchvision.transforms.functional as F


@dataclass
class DynamicCropper:
    """
    If the state is true, then dynamic cropping will be performed. That means that
    if an object is detected (i.e. any body part > detection threshold), then object
    boundaries are computed according to the smallest/largest x position and
    smallest/largest y position of all body parts. This window is expanded by the
    margin and from then on only the posture within this crop is analyzed (until the
    object is lost, i.e. < detection threshold). The current position is utilized for
    updating the crop window for the next frame (this is why the margin is important
    and should be set large enough given the movement of the animal).

    Attributes:
        threshold: float
            The threshold score for bodyparts above which an individual is deemed to
            have been detected.
        margin: int
            The margin used to expand an individuals bounding box before cropping it.

    Examples:
        >>> import deeplabcut.pose_estimation_pytorch.models as models
        >>>
        >>> model: models.PoseModel
        >>> frames: torch.Tensor  # shape (num_frames, 3, H, W)
        >>>
        >>> dynamic = DynamicCropper(threshold=0.6, margin=25)
        >>> predictions = []
        >>> for image in frames:
        >>>     image = dynamic.crop(image)
        >>>
        >>>     outputs = model(image)
        >>>     preds = model.get_predictions(outputs)
        >>>     pose = preds["bodypart"]["poses"]
        >>>
        >>>     dynamic.update(pose)
        >>>     predictions.append(pose)
        >>>
    """
    threshold: float
    margin: int
    _crop: tuple[int, int, int, int] | None = field(default=None, repr=False)
    _shape: tuple[int, int] | None = field(default=None, repr=False)

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crops an input image according to the dynamic cropping parameters.

        Args:
            image: The image to crop, of shape (1, C, H, W).

        Returns:
            The cropped image of shape (1, C, H', W'), where [H', W'] is the size of
            the crop.

        Raises:
            RuntimeError: if there is not exactly one image in the batch to crop, or if
                `crop` was previously called with an image of a different width or
                height.
        """
        if len(image) != 1:
            raise RuntimeError(
                "DynamicCropper can only be used with batch size 1 (found image "
                f"shape: {image.shape})"
            )

        if self._shape is None:
            self._shape = image.shape[3], image.shape[2]

        if image.shape[3] != self._shape[0] or image.shape[2] != self._shape[1]:
            raise RuntimeError(
                "All frames must have the same shape; The first frame had (W, H) "
                f"{self._shape} but the current frame has shape {image.shape}."
            )

        if self._crop is None:
            return image

        x0, y0, x1, y1 = self._crop
        return image[:, :, y0:y1, x0:x1]

    def update(self, pose: torch.Tensor) -> torch.Tensor:
        """Updates the dynamic crop according to the pose model output.

        Uses the pose predicted by the model to update the dynamic crop parameters for
        the next frame. Scales the pose predicted in the cropped image back to the
        original image space and returns it.

        This method modifies the pose tensor in-place; so pass a copy of the tensor if
        you need to keep the original values.

        Args:
            pose: The pose that was predicted by the pose estimation model in the
                cropped image coordinate space.

        Returns:
            The pose, with coordinates updated to the full image space.
        """
        if self._shape is None:
            raise RuntimeError(f"You must call `crop` before calling `update`.")

        # offset the pose to the original image space
        offset_x, offset_y = 0, 0
        if self._crop is not None:
            offset_x, offset_y = self._crop[:2]
        pose[..., 0] = pose[..., 0] + offset_x
        pose[..., 1] = pose[..., 1] + offset_y

        # check whether keypoints can be used for dynamic cropping
        keypoints = pose[..., :3].reshape(-1, 3)
        keypoints = keypoints[~torch.any(torch.isnan(keypoints), dim=1)]
        if len(keypoints) == 0:
            self.reset()
            return pose

        mask = keypoints[:, 2] >= self.threshold
        if torch.all(~mask):
            self.reset()
            return pose

        # set the crop coordinates
        x0 = self._min_value(keypoints[:, 0], self._shape[0])
        x1 = self._max_value(keypoints[:, 0], self._shape[0])
        y0 = self._min_value(keypoints[:, 1], self._shape[1])
        y1 = self._max_value(keypoints[:, 1], self._shape[1])
        crop_w, crop_h = x1 - x0, y1 - y0
        if crop_w == 0 or crop_h == 0:
            self.reset()
        else:
            self._crop = x0, y0, x1, y1

        return pose

    def reset(self) -> None:
        """Resets the DynamicCropper to not crop the next frame"""
        self._crop = None

    @staticmethod
    def build(
        dynamic: bool, threshold: float, margin: int
    ) -> Optional["DynamicCropper"]:
        """Builds the DynamicCropper based on the given parameters

        Args:
            dynamic: Whether dynamic cropping should be used
            threshold: The threshold score for bodyparts above which an individual is
                deemed to have been detected.
            margin: The margin used to expand an individuals bounding box before
                cropping it.

        Returns:
            None if dynamic is False
            DynamicCropper to use if dynamic is True
        """
        if not dynamic:
            return None

        return DynamicCropper(threshold, margin)

    def _min_value(self, coordinates: torch.Tensor, maximum: int) -> int:
        """Returns: min(coordinates - margin), clipped to [0, maximum]"""
        return self._clip(
            int(math.floor(torch.min(coordinates).item() - self.margin)),
            maximum,
        )

    def _max_value(self, coordinates: torch.Tensor, maximum: int) -> int:
        """Returns: max(coordinates + margin), clipped to [0, maximum]"""
        return self._clip(
            int(math.ceil(torch.max(coordinates).item() + self.margin)),
            maximum,
        )

    def _clip(self, value: int, maximum: int) -> int:
        """Returns: The value clipped to [0, maximum]"""
        return min(max(value, 0), maximum)


class TopDownDynamicCropper(DynamicCropper):
    """Dynamic cropping for top-down models used on single animal videos.

    The `TopDownDynamicCropper` can be used instead of an object detector to analyze
    videos **containing a single animal** with top-down models.

    At frame 0, the full frame is split into (n, m) image patches, with a given overlap
    between the patches. Patches are then
        - Resized to the input size required by the model with a top-down crop.
        - Stacked into a batch and given to the pose estimation model
        - The output poses for each patch are post-processed: the patch containing the
            highest average score prediction is selected as the patch containing the
            individual, and the pose from that patch is selected as the predicted pose.

    At frame n, one of two things can happen:
        - If the individual was successfully detected at frame n - 1, a bounding box
            is generated from the predicted pose and used as the bounding box for the
            next frame.
        - If the individual was not detected at frame n - 1, patches are cropped as in
            frame 0 and the pose selected as in frame 0

    An individual is considered to be successfully detected if:
        - at least `min_hq_keypoints` keypoint have scores above the `threshold`

    The bounding box is generated from the keypoints (either from all keypoints or only
    the ones above the threshold) with a margin around the keypoints. If the bounding
    box is smaller than a set minimum size, it is expanded to that size.

    Args:
        top_down_crop_size: The (width, height) of to resize crops to.
        patch_counts: The number of patches along the (width, height) of the images when
            no crop is found.
        patch_overlap: The amount of overlapping pixels between adjacent patches.
        min_bbox_size: The minimum (width, height) for a detected bounding box. If the
            bounding box computed from the keypoints is smaller than this value, it
            will be expanded to these values.
        threshold: The threshold score for bodyparts above which an individual is
            considered to be detected.
        margin: The margin to add around keypoints when generating bounding boxes.
        min_hq_keypoints: The minimum number of keypoints above the threshold required
            for the individual to be considered detected and a bounding box to be
            computed from the pose.
        bbox_from_hq: If True, only keypoints above the score threshold will be used
            to compute the bounding boxes.
        store_crops: Useful for debugging. When True, all crops are stored in the
            `crop_history` attribute.
        **kwargs: Key-word arguments passed to the DynamicCropper base class.

    Attributes:
        min_bbox_size: tuple[int, int]. The minimum (width, height) for a detected
            bounding box. If the bounding box computed from the keypoints is smaller
            than this value, it will be expanded to these values.
        min_hq_keypoints: int. The minimum number of keypoints above the threshold
            required for the individual to be considered detected and a bounding box to
            be computed from the pose.
        bbox_from_hq: bool. If True, only keypoints above the score threshold will be
            used to compute the bounding boxes.
        store_crops: bool. Useful for debugging. When True, all crops are stored in the
            `crop_history` attribute.
        crop_history: list[list[tuple[int, int, int, int]]. Empty list if `store_crops`
            is False. Every time `crop` is called, a list is appended to the
            `crop_history` attribute. This list is empty if no crop was used for the
            frame, otherwise a list containing a single (x, y, w, h) tuple is appended.
    """

    def __init__(
        self,
        top_down_crop_size: tuple[int, int],
        patch_counts: tuple[int, int] = (3, 2),
        patch_overlap: int = 50,
        min_bbox_size: tuple[int, int] = (50, 50),
        threshold: float = 0.6,
        margin: int = 25,
        min_hq_keypoints: int = 2,
        bbox_from_hq: bool = False,
        store_crops: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(threshold=threshold, margin=margin, **kwargs)
        self.min_bbox_size = min_bbox_size
        self.min_hq_keypoints = min_hq_keypoints
        self.bbox_from_hq = bbox_from_hq

        self._patch_counts = patch_counts
        self._patch_overlap = patch_overlap
        self._patches = []
        self._patch_offsets = []
        self._td_crop_size = top_down_crop_size
        self._td_ratio = self._td_crop_size[0] / self._td_crop_size[1]

        self.crop_history = []
        self.store_crops = store_crops

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crops an input image according to the dynamic cropping parameters.

        Args:
            image: The image to crop, of shape (1, C, H, W).

        Returns:
            The cropped image of shape (B, C, H', W'), where [H', W'] is the size of
            the crop.

        Raises:
            RuntimeError: if there is not exactly one image in the batch to crop, or if
                `crop` was previously called with an image of a different W or H.
        """
        if len(image) != 1:
            raise RuntimeError(
                "DynamicCropper can only be used with batch size 1 (found image "
                f"shape: {image.shape})"
            )

        if self._shape is None:
            self._shape = image.shape[3], image.shape[2]
            self._patches = self.generate_patches()

        if image.shape[3] != self._shape[0] or image.shape[2] != self._shape[1]:
            raise RuntimeError(
                "All frames must have the same shape; The first frame had (W, H) "
                f"{self._shape} but the current frame has shape {image.shape}."
            )

        if self._crop is None:
            if self.store_crops:
                self.crop_history.append([])
            return self._crop_patches(image)

        if self.store_crops:
            self.crop_history.append([self._crop])

        return self._crop_bounding_box(image, self._crop)

    def update(self, pose: torch.Tensor) -> torch.Tensor:
        """Updates the dynamic crop according to the pose model output.

        Uses the pose predicted by the model to update the dynamic crop parameters for
        the next frame. Scales the pose predicted in the cropped image back to the
        original image space and returns it.

        This method modifies the pose tensor in-place; so pass a copy of the tensor if
        you need to keep the original values.

        Args:
            pose: The pose that was predicted by the pose estimation model in the
                cropped image coordinate space.

        Returns:
            The pose, with coordinates updated to the full image space.
        """
        if self._shape is None:
            raise RuntimeError(f"You must call `crop` before calling `update`.")

        # check whether this was a patched crop
        batch_size = pose.shape[0]
        if batch_size > 1:
            pose = self._extract_best_patch(pose)

        if self._crop is None:
            raise RuntimeError(
                "The _crop should never be `None` when `update` is called. Ensure you "
                "always alternate between `crop` and `update`."
            )

        # offset and rescale the pose to the original image space
        out_w, out_h = self._td_crop_size
        offset_x, offset_y, w, h = self._crop
        scale_x, scale_y = w / out_w, h / out_h
        pose[..., 0] = (pose[..., 0] * scale_x) + offset_x
        pose[..., 1] = (pose[..., 1] * scale_y) + offset_y
        pose[..., 0] = torch.clip(pose[..., 0], 0, self._shape[0])
        pose[..., 1] = torch.clip(pose[..., 1], 0, self._shape[1])

        # check whether keypoints can be used for dynamic cropping
        keypoints = pose[..., :3].reshape(-1, 3)
        keypoints = keypoints[~torch.any(torch.isnan(keypoints), dim=1)]
        if len(keypoints) == 0:
            self.reset()
            return pose

        mask = keypoints[:, 2] >= self.threshold
        if torch.sum(mask) < self.min_hq_keypoints:
            self.reset()
            return pose

        if self.bbox_from_hq:
            keypoints = keypoints[mask]

        # set the crop coordinates
        x0 = self._min_value(keypoints[:, 0], self._shape[0])
        x1 = self._max_value(keypoints[:, 0], self._shape[0])
        y0 = self._min_value(keypoints[:, 1], self._shape[1])
        y1 = self._max_value(keypoints[:, 1], self._shape[1])
        crop_w, crop_h = x1 - x0, y1 - y0
        if crop_w == 0 or crop_h == 0:
            self.reset()
        else:
            self._crop = self._prepare_bounding_box(x0, y0, x1, y1)

        return pose

    def patch_counts(self) -> tuple[int, int]:
        """Returns: the number of patches created for an image."""
        return self._patch_counts

    def num_patches(self) -> int:
        """Returns: the total number of patches created for an image."""
        return self._patch_counts[0] * self._patch_counts[1]

    def _prepare_bounding_box(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> tuple[int, int, int, int]:
        """Prepares the bounding box for cropping.

        Adds a margin around the bounding box, then transforms it into the target aspect
        ratio required for crops given as inputs to the model.

        Args:
            x1: The x coordinate for the top-left corner of the bounding box.
            y1: The y coordinate for the top-left corner of the bounding box.
            x2: The x coordinate for the bottom-right corner of the bounding box.
            y2: The y coordinate for the bottom-right corner of the bounding box.

        Returns:
            The (x, y, w, h) coordinates for the prepared bounding box.
        """
        x1 -= self.margin
        x2 += self.margin
        y1 -= self.margin
        y2 += self.margin
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2

        input_ratio = w / h
        if input_ratio > self._td_ratio:  # h/w < h0/w0 => h' = w * h0/w0
            h = w /  self._td_ratio
        elif input_ratio < self._td_ratio:  # w/h < w0/h0 => w' = h * w0/h0
            w = h *  self._td_ratio

        x1, y1 = int(round(cx - (w / 2))), int(round(cy - (h / 2)))
        w, h = max(int(w), self.min_bbox_size[0]), max(int(h), self.min_bbox_size[1])
        return x1, y1, w, h

    def _crop_bounding_box(
        self, image: torch.Tensor, bbox: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Applies a top-down crop to an image given a bounding box.

        Args:
            image: The image to crop, of shape (1, C, H, W).
            bbox: The bounding box to crop out of the image.

        Returns:
            The cropped and resized image.
        """
        x1, y1, w, h = bbox
        out_w, out_h = self._td_crop_size
        return F.resized_crop(image, y1, x1, h, w, [out_h, out_w])

    def _crop_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Crops patches from the image.

        Args:
            image: The image to crop patches from, of shape (1, C, H, W).

        Returns:
            The patches, of shape (B, C, H', W'), where [H', W'] is the crop size.
        """
        patches = [self._crop_bounding_box(image, patch) for patch in self._patches]
        return torch.cat(patches, dim=0)

    def _extract_best_patch(self, pose: torch.Tensor) -> torch.Tensor:
        """Extracts the best pose prediction from patches.

        Args:
            pose: The predicted pose, of shape (b, num_idv, num_kpt, 3). The number of
                individuals must be 1.

        Returns:
            The selected pose, of shape [1, N, K, 3]
        """
        # check that only 1 prediction was made in each image
        if pose.shape[1] != 1:
            raise ValueError(
                "The TopDownDynamicCropper can only be used with models predicting "
                f"a single individual per image. Found {pose.shape[0]} "
                f"predictions."
            )

        # compute the score for each individual
        idv_scores = torch.mean(pose[:, 0, :, 2], dim=1)

        # get the index of the best patch
        best_patch = torch.argmax(idv_scores)

        # set the crop to the one used for the best patch
        self._crop = self._patches[best_patch]

        return pose[best_patch:best_patch + 1]

    def generate_patches(self) -> list[tuple[int, int, int, int]]:
        """Generates patch coordinates for splitting an image.

        Returns:
            A list of patch coordinates as tuples (x0, y0, x1, y1).
        """
        patch_xs = self.split_array(
            self._shape[0], self._patch_counts[0], self._patch_overlap
        )
        patch_ys = self.split_array(
            self._shape[1], self._patch_counts[1], self._patch_overlap
        )

        patches = []
        for y0, y1 in patch_ys:
            for x0, x1 in patch_xs:
                patches.append(self._prepare_bounding_box(x0, y0, x1, y1))

        return patches

    @staticmethod
    def split_array(size: int, n: int, overlap: int) -> list[tuple[int, int]]:
        """
        Splits an array into n segments of equal size, where the overlap between each
        segment is at least a given value.

        Args:
            size: The size of the array.
            n: The number of segments to split the array into.
            overlap: The minimum overlap between each segment.

        Returns:
            (start_index, end_index) pairs for each segment. The end index is exclusive.
        """
        if n < 1:
            raise ValueError(f"Array must be split into at least 1 segment. Found {n}.")

        # FIXME - auto-correct the overlap to spread it out more evenly
        padded_size = size + (n - 1) * overlap
        segment_size = (padded_size // n) + (padded_size % n > 0)
        segments = []
        end = overlap
        for i in range(n):
            start = end - overlap
            end = start + segment_size
            if end > size:
                end = size
                start = end - segment_size

            segments.append((start, end))

        return segments
