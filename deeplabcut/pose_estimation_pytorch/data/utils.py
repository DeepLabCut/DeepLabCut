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
from __future__ import annotations

import warnings
from collections import defaultdict
from functools import reduce, lru_cache
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image


@lru_cache(maxsize=None)
def read_image_shape_fast(path: str | Path) -> tuple[int, int, int]:
    """Blazing fast and does not load the image into memory"""
    with Image.open(path) as img:
        width, height = img.size
        return len(img.getbands()), height, width


def bbox_from_keypoints(
    keypoints: np.ndarray,
    image_h: int,
    image_w: int,
    margin: int,
) -> np.ndarray:
    """
    Computes bounding boxes from keypoints.

    Args:
        keypoints: (..., num_keypoints, xy) the keypoints from which to get bboxes
        image_h: the height of the image
        image_w: the width of the image
        margin: the bounding box margin

    Returns:
        the bounding boxes for the keypoints, of shape (..., 4) in the xywh format
    """
    squeeze = False

    # we do not estimate bbox on keypoints that have 0 or -1 flag
    keypoints = np.copy(keypoints)
    keypoints[keypoints[..., -1] <= 0] = np.nan

    if len(keypoints.shape) == 2:
        squeeze = True
        keypoints = np.expand_dims(keypoints, axis=0)

    bboxes = np.full((keypoints.shape[0], 4), np.nan)
    with warnings.catch_warnings():  # silence warnings when all pose confidence levels are <= 0
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bboxes[:, :2] = np.nanmin(keypoints[..., :2], axis=1) - margin  # X1, Y1
        bboxes[:, 2:4] = np.nanmax(keypoints[..., :2], axis=1) + margin  # X2, Y2

    # can have NaNs if some individuals have no visible keypoints
    bboxes = np.nan_to_num(bboxes, nan=0)

    bboxes = np.clip(
        bboxes,
        a_min=[0, 0, 0, 0],
        a_max=[image_w, image_h, image_w, image_h],
    )
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]  # to width
    bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]  # to height
    if squeeze:
        return bboxes[0]

    return bboxes


def merge_list_of_dicts(
    list_of_dicts: list[dict], keys_to_include: list[str]
) -> dict[str, list]:
    """
    Flattens a list of dictionaries into a dictionary with the lists concatenated.

    Args:
        list_of_dicts: the dictionaries to merge
        keys_to_include: the keys to include in the new dictionary

    Returns:
        the merged dictionary

    Examples:
        input:
            list_of_dicts: [{"id": 0, "num": 1}, {"id": 1, "num": 10}]
            keys_to_include: ["id", "num"]
        output:
            {"id": [0, 1], "num": [1, 10]}
    """
    return reduce(
        lambda acc, d: {
            key: acc.get(key, []) + [value]
            for key, value in d.items()
            if key in keys_to_include
        },
        list_of_dicts,
        defaultdict(list),
    )


def map_image_path_to_id(images: list[dict]) -> dict[str, int]:
    """
    Binds the image paths to their respective IDs.

    Args:
        images: List of dictionaries containing image data in COCO-like format.
            Each dictionary should have 'file_name' and 'id' keys.

    Returns:
        A dictionary mapping image paths to their respective IDs.

    Examples:
        images = [{"file_name": "path/to/image1.jpg", "id": 1}, ...]
    """

    return {image["file_name"]: image["id"] for image in images}


def map_id_to_annotations(annotations: list[dict]) -> dict[int, list[int]]:
    """
    Maps image IDs to their corresponding annotation indices.

    Args:
        annotations: List of dictionaries containing annotation data. Each dictionary
            should have 'image_id' key.

    Returns:
        A dictionary mapping image IDs to lists of corresponding annotation indices.

    Examples:
        annotations = [{"image_id": 1, ...}, ...]
    """

    annotation_idx_map = defaultdict(list)
    for idx, annotation in enumerate(annotations):
        annotation_idx_map[annotation["image_id"]].append(idx)

    return annotation_idx_map


def _crop_and_pad_image(
    image: np.ndarray,
    coords: tuple[tuple[int, int], tuple[int, int]],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop the image using the given coordinates and pad the larger dimension to change
    the aspect ratio.

    Args:
        image: Image to crop, of shape (height, width, channels).
        coords: Coordinates for cropping as [(xmin, xmax), (ymin, ymax)].
        output_size: The (output_h, output_w) that this cropped image will be resized
            to. Used to compute padding to keep aspect ratios.

    Returns:
        Cropped (and possibly padded) image
        Padding (pad_h, pad_w)
    """
    cropped_image = image[coords[1][0] : coords[1][1], coords[0][0] : coords[0][1], :]

    crop_h, crop_w, c = cropped_image.shape
    pad_h, pad_w = 0, 0
    target_ratio_h = output_size[0] / crop_h
    target_ratio_w = output_size[1] / crop_w

    if target_ratio_h != target_ratio_w:
        if crop_h < crop_w:
            # Pad the height
            new_h = int(crop_w * output_size[0] / output_size[1])
            pad_h = new_h - crop_h
            pad_image = np.zeros((new_h, crop_w, c))
            y_offset = pad_h // 2
            pad_image[y_offset : y_offset + crop_h, :] = cropped_image
        else:
            # Pad the width
            new_w = int(crop_h * output_size[1] / output_size[0])
            pad_w = new_w - crop_w
            pad_image = np.zeros((crop_h, new_w, c))
            x_offset = pad_w // 2
            pad_image[:, x_offset : x_offset + crop_w] = cropped_image
    else:
        pad_image = cropped_image

    return pad_image, (pad_h, pad_w)


def _crop_and_pad_keypoints(
    keypoints: np.ndarray, coords: tuple[int, int], pad_size: tuple[int, int]
):
    """
    Adjust the keypoints after cropping and padding.

    Parameters:
        keypoints: The original keypoints, typically a 2D array of shape (..., 2).
        coords: The (xmin, ymin) crop coordinates used for cropping the image.
        pad_size: The padding sizes added to the cropped image, in the format (pad_h, pad_w).

    Returns:
        Adjusted keypoints.
    """
    keypoints[..., 0] -= coords[0]
    keypoints[..., 1] -= coords[1]
    keypoints[..., 0] += pad_size[1] // 2
    keypoints[..., 1] += pad_size[0] // 2
    return keypoints


def _crop_image_keypoints(
    image, keypoints, coords, output_size
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """TODO: Requires fixing
    Crop the image based on a given bounding box and resize it to the desired output
    size. Returns offsets and scales to map keypoints in the resized image to
    coordinates in the original image:

        x_original = (x_cropped * x_scale) + x_offset
        y_original = (y_cropped * y_scale) + y_offset

    Args:
        image: Image to crop, of shape (height, width, channels).
        coords: Coordinates for cropping as ((xmin, xmax), (ymin, ymax)).
        output_size: The (h, w) that the cropped image should be resized to.

    Returns:
        Cropped, possibly padded, and resized image.
        The position of the keypoints in the cropped, resized image
        Offsets used for cropping.
        The offsets to map predicted keypoints back to the original image
        The scale to map predicted keypoints back to the original image
    """

    cropped_image, pad_size = _crop_and_pad_image(image, coords, output_size)
    cropped_keypoints = _crop_and_pad_keypoints(
        keypoints, (coords[0][0], coords[1][0]), pad_size
    )

    offsets = (coords[0][0], coords[1][0])
    scales = [
        output_size[0] / cropped_image.shape[0],
        output_size[1] / cropped_image.shape[1],
    ]

    # TODO: Fix resizing, use OpenCV
    cropped_resized_image = np.resize(
        cropped_image, (*output_size, cropped_image.shape[2])
    )

    cropped_resized_keypoints = np.array(cropped_keypoints) * np.array(scales + [1])

    return cropped_resized_image, cropped_resized_keypoints, offsets, scales


def _compute_crop_bounds(
    bboxes: np.ndarray,
    image_shape: tuple[int, int, int],
    remove_empty: bool = True,
) -> np.ndarray:
    """
    Compute the boundaries for cropping an image based on a COCO-format bounding box
    and image shape by clipping values so the bounding boxes are entirely in the image.

    Args:
        bboxes: COCO-format bounding box of shape (b, xywh)
        image_shape: Shape of the image defined as (height, width, channels).

    Returns:
        The bounding boxes, clipped to be entirely inside the image
    """
    h, w = image_shape[:2]
    # to xyxy
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    # clip
    bboxes = np.clip(bboxes, 0, np.array([w, h, w, h]))
    # to xywh
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    # filter
    if remove_empty:
        squashed_bbox_mask = np.logical_or(bboxes[:, 2] <= 0, bboxes[:, 3] <= 0)
        bboxes = bboxes[~squashed_bbox_mask]
    return bboxes


def _extract_keypoints_and_bboxes(
    anns: list[dict],
    image_shape: tuple[int, int, int],
    num_joints: int,
    num_unique_bodyparts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Args:
        anns: COCO-style annotations
        image_shape: the (h, w, c) shape of the image for which to get annotations
        num_joints: the number of joints in the annotations

    Returns:
        keypoints, unique_keypoints, bboxes in xywh format, annotations_merged
    """
    keypoints = []
    original_bboxes = []
    anns_to_merge = []
    unique_keypoints = None
    h, w = image_shape[:2]
    for i, annotation in enumerate(anns):
        keypoints_individual = _annotation_to_keypoints(annotation, h, w)
        if annotation["individual"] != "single":
            bbox_individual = annotation["bbox"]
            original_bboxes.append(bbox_individual)
            keypoints.append(keypoints_individual)
            anns_to_merge.append(annotation)
        else:
            unique_keypoints = keypoints_individual

    if unique_keypoints is None:
        unique_keypoints = -1 * np.ones((num_unique_bodyparts, 3), dtype=float)

    keypoints = safe_stack(keypoints, (0, num_joints, 3))
    original_bboxes = safe_stack(original_bboxes, (0, 4))
    bboxes = _compute_crop_bounds(original_bboxes, image_shape, remove_empty=False)

    # at least 1 visible joint to keep individuals
    vis_mask = (keypoints[..., 2] > 0).any(axis=1)
    keypoints = keypoints[vis_mask]
    bboxes = bboxes[vis_mask]

    keys_to_merge = ["area", "category_id", "iscrowd", "individual_id"]
    anns_merged = {k: [] for k in keys_to_merge}
    if len(anns_to_merge) > 0:
        anns_merged = merge_list_of_dicts(anns_to_merge, keys_to_include=keys_to_merge)
    anns_merged = {k: np.array(v)[vis_mask] for k, v in anns_merged.items()}

    if len(anns_merged["area"]) != len(keypoints):
        raise ValueError(f"Missing area values! {anns_merged}, {keypoints.shape}")

    return keypoints, unique_keypoints, bboxes, anns_merged


def calc_area_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Calculate the area from keypoints

    TODO: in the pups benchmark, there are 5 keypoints perfectly aligned so
     the area is 0.
     How do we deal with that?
     Makes more sense to compute the area from the bboxes (they are padded)
     Below is a temporary fix, which sets a min height and width to 5
     Suggestion: compute min height/width using labeled data

    Args:
        keypoints (np.ndarray): array of keypoints

    Returns:
        np.ndarray: array containing the computed areas based on the keypoints
    """
    w = np.maximum(keypoints[:, :, 0].max(axis=1) - keypoints[:, :, 0].min(axis=1), 1)
    h = np.maximum(keypoints[:, :, 1].max(axis=1) - keypoints[:, :, 1].min(axis=1), 1)
    return w * h


def calc_bbox_overlap(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """
    Calculate the overlap between two bounding boxes

    Args:
        bbox1: the first bounding box in the format (x, y, w, h)
        bbox2: the second bounding box in the format (x, y, w, h)

    Returns:
        The overlap between
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))

    intersection = x_overlap * y_overlap
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union


def _annotation_to_keypoints(annotation: dict, h: int, w: int) -> np.array:
    """
    Convert the coco annotations into array of keypoints returns the array of the
    keypoints' visibility. If keypoint is not visible, the value for (x,y) coordinates
    is set to 0. If the keypoints are outside of the image, they are also set to 0.

    Args:
        annotation: dictionary containing coco-like annotations with essential
            `keypoints` field
        h: the image height
        w: the image width

    Returns:
        keypoints: np.array where the first two columns are x and y coordinates of the

    """
    # we don't mess up visibility flags here
    return annotation["keypoints"].reshape(-1, 3)


def apply_transform(
    transform: A.BaseCompose,
    image: np.ndarray,
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    class_labels: list[str],
) -> dict[str, np.ndarray]:
    """
    Applies a transformation to the provided image and keypoints.

    Args:
        transform: The transformation to apply.
        image: The input image to which the transformation will be applied.
        keypoints: List of keypoints to be transformed along with the image. Each keypoint
            is expected to be a tuple or list with at least three values,
            where the third value indicates the class label index.
        bboxes: List of bounding boxes to be transformed along with the image.
        class_labels: List of class labels corresponding to the keypoints.

    Returns:
        transformed: A dictionary containing the transformed image and keypoints.
    """

    if transform:
        oob_mask = out_of_bounds_keypoints(keypoints, image.shape)
        transformed = _apply_transform(
            transform, image, keypoints, bboxes, class_labels
        )

        transformed["keypoints"] = np.array(transformed["keypoints"])

        # out-of-bound keypoints have visibility flag 0. But we don't touch coordinates
        if np.sum(oob_mask) > 0:
            transformed["keypoints"][oob_mask, 2] = 0.0

        out_shape = transformed["image"].shape
        if len(transformed["keypoints"]) > 0:
            oob_mask = out_of_bounds_keypoints(transformed["keypoints"], out_shape)
            # out-of-bound keypoints have visibility flag 0. Don't touch coordinates
            if np.sum(oob_mask) > 0:
                transformed["keypoints"][oob_mask, 2] = 0.0

        # TODO: Check that the transformed bboxes are still within the image
        if len(transformed["bboxes"]) > 0:
            transformed["bboxes"] = np.array(transformed["bboxes"])
        else:
            transformed["bboxes"] = np.zeros(shape=(0, 4))

    else:
        transformed = {"keypoints": keypoints, "image": image}

    # do we ever need to do this if we had check_keypoints_within_bounds above?
    # np.nan_to_num(transformed["keypoints"], copy=False, nan=-1)
    return transformed


def _apply_transform(
    transform: A.BaseCompose,
    image: np.ndarray,
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    class_labels: list[str],
) -> dict[str, np.ndarray]:
    """
    Applies a transformation to the provided image and keypoints.

    Args:
        image : np.array or similar image data format
            The input image to which the transformation will be applied.

        keypoints : list or similar data format
            List of keypoints to be transformed along with the image. Each keypoint
            is expected to be a tuple or list with at least three values,
            where the third value indicates the class label index.

    Returns:
        dict
            A dictionary containing the transformed image and keypoints.
    """
    transformed = transform(
        image=image,
        keypoints=keypoints,
        class_labels=class_labels,
        bboxes=bboxes,
        bbox_labels=np.arange(len(bboxes)),
    )

    bboxes_out = np.zeros(bboxes.shape)
    for bbox, bbox_id in zip(transformed["bboxes"], transformed["bbox_labels"]):
        bboxes_out[bbox_id] = bbox

    transformed["bboxes"] = bboxes_out
    return transformed


def out_of_bounds_keypoints(keypoints: np.ndarray, shape: tuple) -> np.ndarray:
    """Computes which visible keypoints are outside an image

    Args:
        keypoints: A (N, 3) shaped array where N is the number of keypoints and each
            keypoint is represented as (x, y, visibility).
        shape: A tuple representing the shape or bounds as (height, width).

    Returns:
        A boolean array of shape (N,) where each element corresponds to whether
        the respective keypoint is visible (visibility > 0) and outside the image
        bounds. This mask can be used to set the visibility bit to 0 for keypoints that
        were kicked off an image due to augmentation.
    """
    return (keypoints[..., 2] > 0) & (
        np.isnan(keypoints[..., 0])
        | np.isnan(keypoints[..., 1])
        | (keypoints[..., 0] < 0)
        | (keypoints[..., 0] > shape[1])
        | (keypoints[..., 1] < 0)
        | (keypoints[..., 1] > shape[0])
    )


def pad_to_length(data: np.array, length: int, value: float) -> np.array:
    """
    Pads the first dimension of an array with a given value

    Args:
        data: the array to pad, of shape (l, ...), where l <= length
        length: the desired length of the tensor
        value: the value to pad with

    Returns:
        the padded array of shape (length, ...)
    """
    pad_length = length - len(data)
    if pad_length == 0:
        return data
    elif pad_length > 0:
        padding = value * np.ones((pad_length, *data.shape[1:]), dtype=data.dtype)
        return np.concatenate([data, padding])

    raise ValueError(f"Cannot pad! data.shape={data.shape} > length={length}")


def safe_stack(data: list[np.ndarray], default_shape: tuple[int, ...]) -> np.ndarray:
    """
    Stacks a list of arrays if there are any, otherwise returns an array of zeros
    of a desired shape.

    Args:
        data: the list of arrays to stack
        default_shape: the shape of the array to return if the list is empty

    Returns:
        the stacked data or empty array
    """
    if len(data) == 0:
        return np.zeros(default_shape, dtype=float)

    return np.stack(data, axis=0)
