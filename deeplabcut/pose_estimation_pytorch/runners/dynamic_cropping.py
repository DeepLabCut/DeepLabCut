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

    def update(self, pose: torch.Tensor) -> None:
        """Updates the dynamic crop according to the pose model output.

        Uses the pose predicted by the model to update the dynamic crop parameters for
        the next frame. Scales the pose predicted in the cropped image back to the
        original image space and returns it.

        This method modifies the pose tensor in-place; so pass a copy of the tensor if
        you need to keep the original values.

        Args:
            pose: The pose that was predicted by the pose estimation model in the
                cropped image coordinate space.
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
            return

        mask = keypoints[:, 2] >= self.threshold
        if torch.all(~mask):
            self.reset()
            return

        # set the crop coordinates
        x0 = self._min_value(keypoints[:, 0], self._shape[0])
        x1 = self._max_value(keypoints[:, 0], self._shape[0])
        y0 = self._min_value(keypoints[:, 1], self._shape[1])
        y1 = self._max_value(keypoints[:, 1], self._shape[1])
        crop_w, crop_h = x1 - x0, y1 - y0
        if crop_w == 0 or crop_h == 0:
            self.reset()
            return

        self._crop = x0, y0, x1, y1

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
