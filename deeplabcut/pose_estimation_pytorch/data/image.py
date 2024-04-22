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
"""Classes and functions to manipulate images"""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.ops import box_convert

from deeplabcut.pose_estimation_pytorch.data.utils import _compute_crop_bounds


def load_image(filepath: str | Path, color_mode: str = "RGB") -> np.ndarray:
    """Loads an image from a file using cv2

    Args:
        filepath: the path of the file containing the image to load
        color_mode: {'RGB', 'BGR'} the color mode to load the image with

    Returns:
        the image as a numpy array
    """
    image = cv2.imread(str(filepath))
    if color_mode == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif not color_mode == "BGR":
        raise ValueError(f"Unsupported `color_mode`: {color_mode}")

    return image


def _crop_and_pad_image_torch(
    image: np.array, bbox: np.array, bbox_format: str, output_size: int
) -> tuple[np.array, tuple[int, int], tuple[int, int]]:
    """TODO: Reimplement this function with numpy and for non-square resize :)
    Only works for square cropped bounding boxes. Crops images around bounding boxes
    for top-down pose estimation in a MMpose style. Computes offsets so that
    coordinates in the original image can be mapped to the cropped one;

        x_cropped = (x - offset_x) / scale_x
        x_cropped = (y - offset_y) / scale_y

    Args:
        image: (h, w, c) the image to crop
        bbox: (4,) the bounding box to crop around
        bbox_format: {"xyxy", "xywh", "cxcywh"} the format of the bounding box
        output_size: the size to resize the image to

    Returns:
        cropped_image, (offset_x, offset_y), (scale_x, scale_y)
    """
    image = torch.tensor(image).permute(2, 0, 1)
    bbox = torch.tensor(bbox)
    if bbox_format != "cxcywh":
        bbox = box_convert(bbox.unsqueeze(0), bbox_format, "cxcywh").squeeze()

    c, h, w = image.shape
    crop_size = torch.max(bbox[2:])

    xmin = int(torch.clip(bbox[0] - (crop_size / 2), min=0, max=w - 1).cpu().item())
    xmax = int(torch.clip(bbox[0] + (crop_size / 2), min=1, max=w).cpu().item())
    ymin = int(torch.clip(bbox[1] - (crop_size / 2), min=0, max=h - 1).cpu().item())
    ymax = int(torch.clip(bbox[1] + (crop_size / 2), min=1, max=h).cpu().item())
    cropped_image = image[:, ymin:ymax, xmin:xmax]

    crop_h, crop_w = cropped_image.shape[1:3]
    pad_size = max(crop_h, crop_w)
    offset = (xmin, ymin)

    # Pad image if not square
    if not crop_h == crop_w:
        padded_cropped_image = torch.zeros((c, pad_size, pad_size), dtype=image.dtype)
        # Try to center bbox in padding
        w_start = 0
        if bbox[0] - (crop_size / 2) < 0:
            # padding on the left
            w_start = pad_size - crop_w
        elif bbox[0] + (crop_size / 2) >= w:
            # padding on the right
            w_start = 0

        h_start = 0
        if bbox[1] - (crop_size / 2) < 0:
            # padding at the top
            h_start = pad_size - crop_h
        elif bbox[1] + (crop_size / 2) >= h:
            # padding at the bottom
            h_start = 0

        h_end = h_start + crop_h
        w_end = w_start + crop_w
        offset = (offset[0] - w_start, offset[1] - h_start)
        padded_cropped_image[:, h_start:h_end, w_start:w_end] = cropped_image
        cropped_image = padded_cropped_image

    scale = pad_size / output_size
    output = F.resize(cropped_image, [output_size, output_size], antialias=True)
    return output.permute(1, 2, 0).numpy(), offset, (scale, scale)


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
