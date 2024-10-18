from __future__ import annotations

import copy

import numpy as np
import torch
import torchvision.transforms.functional as F

from deeplabcut.pose_estimation_pytorch.data.utils import _compute_crop_bounds


def resize_and_random_crop(
    image: np.ndarray,
    targets: dict,
    size: int | tuple[int, int],
    max_size: int | None = None,
    max_shift: int | None = None,
) -> tuple[torch.tensor, dict]:
    """Resizes images while preserving their aspect ratio

    If size is an integer: resizes to square images.
        First, resizes the image so that it's short side is equal to `size`. If this
        makes its long side greater than `max_size`, resizes the long side to `max_size`
        and the short side to the corresponding value to preserve the aspect ratio.

        Then, the image is cropped to a size-by-size square with a random crop.

    If size is a tuple, resize images to (w=size[1], h=size[0])
        First, rescales the image while preserving the aspect ratio such that both its
        width and height are greater or equal to the target width/height for the image
        (where either the width/height is the target width/height). If this makes its
        long side greater than `max_size`, resizes the long side to `max_size`.

        Then, the image is cropped to (w=size[1], h=size[0]) with a random crop.

    Args:
        image: an image of shape (C, H, W)
        targets: the dictionary containing targets
        size: the size of the output image (it will be square)
        max_size: if defined, the maximum size of any side of the output image
        max_shift: the maximum shift for the crop after resizing

    Returns: image, targets
        the resized image as a PyTorch tensor
        the updated targets in the resized image
    """

    def get_resize_hw(
        original_size: tuple[int, int], tgt_short_side: int, max_long_side: int | None
    ) -> tuple[int, int]:
        short_side, long_side = min(*original_size), max(*original_size)
        tgt_long_side = int((tgt_short_side / short_side) * long_side)

        # if the image's long side will be too big, make the image smaller
        if max_long_side is not None and tgt_long_side > max_long_side:
            tgt_long_side = max_long_side
            tgt_short_side = int((tgt_long_side / long_side) * short_side)

        # height is the short side
        if original_size[0] < original_size[1]:
            return tgt_short_side, tgt_long_side

        # width is the short side
        return tgt_long_side, tgt_short_side

    def get_resize_preserve_ratio(
        oh: int, ow: int, tgt_h: int, tgt_w: int, max_long_side: int | None
    ) -> tuple[int, int]:
        w_scale = ow / tgt_w
        h_scale = oh / tgt_h
        if h_scale <= w_scale:
            h = tgt_h
            w = int(ow * (tgt_h / oh))
        else:
            h = int(oh * (tgt_w / ow))
            w = tgt_w

        # if the image's long side will be too big, make the image smaller
        long_side = max(h, w)
        if max_long_side is not None and long_side > max_long_side:
            if h <= w:
                w = max_long_side
                h = int(oh * (max_long_side / ow))
            else:
                w = int(ow * (max_long_side / oh))
                h = max_long_side

        return h, w

    oh, ow = image.shape[1:]
    if isinstance(size, int):
        h, w = get_resize_hw((oh, ow), tgt_short_side=size, max_long_side=max_size)
        tgt_h, tgt_w = size, size
    else:
        h, w = get_resize_preserve_ratio(
            oh, ow, size[0], size[1], max_long_side=max_size
            )
        tgt_h, tgt_w = size

    scale_x, scale_y = ow / w, oh / h
    scaled_image = F.resize(torch.tensor(image), [h, w])

    # shift the image
    if max_shift is None:
        max_shift = 0
    extra_x, extra_y = max(0, w - tgt_w), max(0, h - tgt_h)
    offset_x = np.random.randint(
        max(-tgt_w // 2, -max(0, tgt_w - w) - max_shift),
        min(max_shift + extra_x, extra_x + (min(w, tgt_w) // 2)),
    )
    offset_y = np.random.randint(
        max(-tgt_h // 2, -max(0, tgt_h - h) - max_shift),
        min(max_shift + extra_y, extra_y + (min(h, tgt_h) // 2)),
    )

    # 0-pads, then crops if image size is smaller than output size along any edge
    scaled_cropped_image = F.crop(scaled_image, offset_y, offset_x, tgt_h, tgt_w)

    # update targets
    targets = copy.deepcopy(targets)

    # update scales and offsets
    sx, sy = targets["scales"]
    ox, oy = targets["offsets"]
    targets["offsets"] = ox + (offset_x * sx), oy + (offset_y * sy)
    targets["scales"] = sx * scale_x, sy * scale_y

    # update annotations
    anns = targets.get("annotations", {})

    kpt_scale = np.array([scale_x, scale_y])
    kpt_offset = np.array([offset_x, offset_y])
    for kpt_key in ["keypoints", "keypoints_unique"]:
        keypoints = anns.get(kpt_key)
        if keypoints is not None and len(keypoints) > 0:
            scaled_kpts = keypoints.copy()
            scaled_kpts[..., :2] = (scaled_kpts[..., :2] / kpt_scale) - kpt_offset
            scaled_kpts[(scaled_kpts[..., 0] >= tgt_w)] = -1
            scaled_kpts[(scaled_kpts[..., 1] >= tgt_h)] = -1
            scaled_kpts[(scaled_kpts[..., :2] < 0).any(axis=-1)] = -1
            anns[kpt_key] = scaled_kpts

    bbox_scale = np.array([scale_x, scale_y, scale_x, scale_y])
    bbox_offset = np.array([offset_x, offset_y, 0, 0])
    for bbox_key in ["boxes"]:
        boxes = anns.get(bbox_key)
        if boxes is not None and len(boxes) > 0:
            scaled_boxes = (boxes / bbox_scale) - bbox_offset
            scaled_boxes = _compute_crop_bounds(
                scaled_boxes, (tgt_h, tgt_w, 3), remove_empty=False,
            )
            anns[bbox_key] = scaled_boxes

    area = anns.get("area")
    if area is not None:
        if "boxes" in anns:  # recompute areas from the new bounding boxes
            widths = np.maximum(anns["boxes"][..., 2], 1)
            heights = np.maximum(anns["boxes"][..., 3], 1)
            anns["area"] = widths * heights
        else:  # just rescale
            scaled_area = area * (scale_x * scale_y)
            anns["area"] = scaled_area

    return scaled_cropped_image, targets
