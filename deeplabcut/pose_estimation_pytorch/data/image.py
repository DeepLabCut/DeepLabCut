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
    
    def scale_kpts(
        keypoints: np.ndarray, kpt_scale: np.ndarray, kpt_offset: np.ndarray,
        tgt_h: int, tgt_w: int
    ) -> np.ndarray:
        scaled_kpts = keypoints.copy()
        scaled_kpts[..., :2] = (scaled_kpts[..., :2] / kpt_scale) - kpt_offset
        scaled_kpts[(scaled_kpts[..., 0] >= tgt_w)] = -1
        scaled_kpts[(scaled_kpts[..., 1] >= tgt_h)] = -1
        scaled_kpts[(scaled_kpts[..., :2] < 0).any(axis=-1)] = -1
        return scaled_kpts

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

    # update annotations and context
    anns = targets.get("annotations", {})
    context = targets.get("context", {})

    kpt_scale = np.array([scale_x, scale_y])
    kpt_offset = np.array([offset_x, offset_y])
    for kpt_key in ["keypoints", "keypoints_unique"]:
        keypoints = anns.get(kpt_key)
        if keypoints is not None and len(keypoints) > 0:
            anns[kpt_key] = scale_kpts(keypoints, kpt_scale, kpt_offset, tgt_h, tgt_w)
    cond_keypoints = context.get("cond_keypoints")
    if cond_keypoints is not None and len(cond_keypoints) > 0:
        context["cond_keypoints"] = scale_kpts(cond_keypoints, kpt_scale, kpt_offset, tgt_h, tgt_w)
        
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


def top_down_crop(
    image: np.ndarray,
    bbox: np.ndarray,
    output_size: tuple[int, int],
    margin: int = 0,
    center_padding: bool = False,
    crop_with_context: bool = True,
) -> tuple[np.array, tuple[int, int], tuple[float, float]]:
    """
    Crops images around bounding boxes for top-down pose estimation. Computes offsets so
    that coordinates in the original image can be mapped to the cropped one;

        x_cropped = (x - offset_x) / scale_x
        x_cropped = (y - offset_y) / scale_y

    Bounding boxes are expected to be in COCO-format (xywh).

    Args:
        image: (h, w, c) the image to crop
        bbox: (4,) the bounding box to crop around
        output_size: the (width, height) of the output cropped image
        margin: a margin to add around the bounding box before cropping
        center_padding: whether to center the image in the padding if any is needed
        crop_with_context: Whether to keep context around the bounding box when cropping

    Returns:
        cropped_image, (offset_x, offset_y), (scale_x, scale_y)
    """
    image_h, image_w, c = image.shape
    out_w, out_h = output_size
    x, y, w, h = bbox
    
    # Safety check for zero dimensions
    # if w <= 0:
    #     print(f"ERROR: Zero width in bbox {bbox}, using default width=1")
    #     w = 1
    # if h <= 0:
    #     print(f"ERROR: Zero height in bbox {bbox}, using default height=1")
    #     h = 1

    cx = x + w / 2
    cy = y + h / 2
    w += 2 * margin
    h += 2 * margin

    if crop_with_context:
        input_ratio = w / h
        output_ratio = out_w / out_h
        if input_ratio > output_ratio:  # h/w < h0/w0 => h' = w * h0/w0
            h = w / output_ratio
        elif input_ratio < output_ratio:  # w/h < w0/h0 => w' = h * w0/h0
            w = h * output_ratio

    # cx,cy,w,h will now give the right ratio -> check if padding is needed
    x1, y1 = int(round(cx - (w / 2))), int(round(cy - (h / 2)))
    x2, y2 = int(round(cx + (w / 2))), int(round(cy + (h / 2)))

    # pad symmetrically - compute total padding across axis
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if x1 < 0:
        pad_left = -x1
        x1 = 0
    if x2 > image_w:
        pad_right = x2 - image_w
        x2 = image_w
    if y1 < 0:
        pad_top = -y1
        y1 = 0
    if y2 > image_h:
        pad_bottom = y2 - image_h
        y2 = image_h

    w, h = x2 - x1, y2 - y1
    if not crop_with_context:
        input_ratio = w / h
        output_ratio = out_w / out_h
        if input_ratio > output_ratio:  # h/w < h0/w0 => h' = w * h0/w0
            w_pad = int(w - h * output_ratio) // 2
            pad_top += w_pad
            pad_bottom += w_pad

        elif input_ratio < output_ratio:  # w/h < w0/h0 => w' = h * w0/h0
            h_pad = int(h - (w / output_ratio)) // 2
            pad_left += h_pad
            pad_right += h_pad

    pad_x = pad_left + pad_right
    pad_y = pad_top + pad_bottom
    if center_padding:
        pad_left = pad_x // 2
        pad_top = pad_y // 2

    # crop the pixels we care about
    image_crop = np.zeros((h + pad_y, w + pad_x, c), dtype=image.dtype)
    image_crop[pad_top:pad_top + h, pad_left:pad_left + w] = image[y1:y2, x1:x2]

    # resize the cropped image
    image = cv2.resize(image_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # compute scale and offset
    offset = x1 - pad_left, y1 - pad_top
    scale = (w + pad_x) / out_w, (h + pad_y) / out_h
    return image, offset, scale
