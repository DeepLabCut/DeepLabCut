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

from typing import List, Tuple

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.models.target_generators.plateau_targets as deeplabcut_torch_plateau_targets


def get_target(
    batch_size: int,
    num_animals: int,
    num_joints: int,
    image_size: Tuple[int, int],
    locref_stdev: float,
    pos_dist_thresh: int,
):
    """Summary
    Getting the target generator for certain annotations, predictions and image size.

    Args:
        batch_size (int): number of images
        num_animals (int): number of animals
        num_joints (int): number of bodyparts
        image_size (tuple): image size in pixels
        locref_stdev (float): scaling factor
        pos_dist_thresh (int): radius plateau on the heatmap

    Returns:
        target_output (dict): containing the heatmaps, locref_maps and locref_masks.
        annotations (dict): containing input keypoint annotations.

    Examples:
        input:
            batch_size = 1
            num_animals = 1
            num_joints = 6
            image_size = (256,256)
            locref_stdev = 7.2801
            pos_dist_thresh = 17
        output:

    """
    annotations = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_joints, 2)
        )
    }  # 2 for x,y coords
    prediction = [torch.rand((batch_size, num_joints, image_size[0], image_size[1]))]
    generator = deeplabcut_torch_plateau_targets.PlateauLocrefGenerator(
        locref_stdev, num_joints, pos_dist_thresh
    )

    targets_output = generator(annotations, prediction, image_size)
    return targets_output, annotations


data = [(1, 1, 10, (256, 256), 7.2801, 17)]


@pytest.mark.parametrize(
    "batch_size, num_animals, num_joints, image_size, locref_stdev, pos_dist_thresh",
    data,
)
def test_expected_output(
    batch_size: int,
    num_animals: int,
    num_joints: int,
    image_size: Tuple[int, int],
    locref_stdev: float,
    pos_dist_thresh: int,
):
    """Summary:
    Testing if plateau_targets.py returns the expected output. We take a target generator from
    get_target function. Given a sequence of random numbers for batch_size, num_animals etc., we assert if
    it returns the expected heatmaps and locrefmaps, as well as checking if the output has the expected shape.

    Args:
        batch_size (int): number of images
        num_animals (int): number of animals
        num_joints (int): number of bodyparts
        image_size (tuple): image size in pixels
        locref_stdev (float): scaling factor
        pos_dist_thresh (int): radius plateau on heatmap

    Returns:
        None

    Examples:
        input:
            batch_size = 1
            num_animals = 1
            num_joints = 6
            image_size = (256,256)
            locref_stdev = 7.2801
            pos_dist_thresh = 17
    """
    targets_output, annotations = get_target(
        batch_size, num_animals, num_joints, image_size, locref_stdev, pos_dist_thresh
    )

    assert "heatmaps" in targets_output
    assert "locref_maps" in targets_output
    assert "locref_masks" in targets_output

    assert targets_output["heatmaps"].shape == (
        batch_size,
        num_joints,
        image_size[0],
        image_size[1],
    )  # heatmaps score output
    assert targets_output["locref_masks"].shape == (
        batch_size,
        num_joints * 2,
        image_size[0],
        image_size[1],
    )
    assert targets_output["locref_maps"].shape == (
        batch_size,
        num_joints * 2,
        image_size[0],
        image_size[1],
    )


data = [(1, 1, 10, (256, 256), 7.2801, 17)]


@pytest.mark.parametrize(
    "batch_size, num_animals, num_joints, image_size, locref_stdev, pos_dist_thresh",
    data,
)
def test_single_animal(
    batch_size: int,
    num_animals: int,
    num_joints: int,
    image_size: Tuple[int, int],
    locref_stdev: float,
    pos_dist_thresh: int,
):
    """Summary
    Testing, for single animals experiments (num_animals=1) if the distance between the expected keypoints
    and the annotations keypoints is smaller than the radius plateau.

    'argmax' function returns the indices of the max values of all elements in the input tensor.
    If there are multiple maximal values, such as in our case because it's a plateau, then the
    indices of the first maximal value are returned. From this tensor we exctact x,y coords
    and then concatenate these new tensors along a new dimension. Then, we assert if the distance between
    each x,y element in annotations and predicted keypoints is smaller or equal to the 'pos_dist_thresh',
    which represents the radius of the plateau heatmap.

    Args:
        batch_size (int): number of images
        num_animals (int): number of animals
        num_joints (int): number of bodyparts
        image_size (tuple): image size in pixels
        locref_stdev (float): scaling factor
        pos_dist_thresh (int): radius plateau on heatmap

    Returns:
        None

    Examples:
        input:
            batch_size = 1
            num_animals = 1
            num_joints = 6
            image_size = (256,256)
            locref_stdev = 7.2801
            pos_dist_thresh = 17
    """
    targets_output, annotations = get_target(
        batch_size, num_animals, num_joints, image_size, locref_stdev, pos_dist_thresh
    )

    targets_output = torch.tensor(
        targets_output["heatmaps"].reshape(1, 10, image_size[0] * image_size[1])
    )  # converting from dict to tensor. 'argmax' works on tensors.

    plt_max = torch.argmax(targets_output, dim=2)
    # get unraveled coords
    x = plt_max % image_size[1]
    y = plt_max // image_size[1]

    predict_kp = torch.stack((x, y), dim=-1)

    predict_kp = predict_kp.float()

    annotations["keypoints"] = torch.squeeze(annotations["keypoints"], dim=1)
    annotations["keypoints"] = annotations["keypoints"].float()

    dist = torch.norm(annotations["keypoints"] - predict_kp, p=2, dim=-1)
    assert (dist <= pos_dist_thresh).all()
