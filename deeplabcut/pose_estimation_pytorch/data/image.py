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

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
from torchvision.transforms import functional as F


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
