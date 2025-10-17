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

import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Generic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

import deeplabcut.core.metrics as metrics
import deeplabcut.pose_estimation_pytorch.runners.schedulers as schedulers
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import (
    attempt_snapshot_load,
    ModelType,
    Runner,
)
from deeplabcut.pose_estimation_pytorch.runners.logger import (
    BaseLogger,
    CSVLogger,
    ImageLoggerMixin,
)
from deeplabcut.pose_estimation_pytorch.runners.snapshots import TorchSnapshotManager
from deeplabcut.pose_estimation_pytorch.task import Task


def _get_heat( target: dict, head: str = "bodypart"):
    if head not in target or "heatmap" not in target[head] or "target" not in target[head]["heatmap"]:
        return None, None, None
    heat = target[head]["heatmap"]["target"]  # torch.Tensor
    if heat.ndim != 4:
        return None, None, None
    # layout detection
    # DLC uses channels-last (B, H, W, C) layout for heatmaps
    # Check if last dimension is smaller (likely to be channels)
    if heat.shape[-1] < heat.shape[1]:
        # Channels-last: (B, H, W, C)
        C_first = False
        H, W = heat.shape[1], heat.shape[2]
    else:
        # Channels-first: (B, C, H, W)
        C_first = True
        H, W = heat.shape[2], heat.shape[3]
    return heat, C_first, (H, W)

def _apply_mask_inplace(heat: torch.Tensor, b: int, c: int, mask: torch.Tensor, channels_first: bool):
    # mask shape: (H, W)
    if channels_first:
        heat[b, c, :, :] *= mask
    else:
        heat[b, :, :, c] *= mask

def _compute_scale_factor_svl(animal_keypoints, bodyparts, skeletal_links_lengths):
    """Return mm->pixel scale using SVL if available; else None."""
    try:
        snout_idx = bodyparts.index('snout')
        tail1_idx = bodyparts.index('tail1')
    except ValueError:
        return None

    # GT SVL in pixels (requires both visible)
    if (animal_keypoints[snout_idx, 2] > 0.5) and (animal_keypoints[tail1_idx, 2] > 0.5):
        gt_svl_pix = float(torch.norm(
            torch.tensor(animal_keypoints[snout_idx, :2]) -
            torch.tensor(animal_keypoints[tail1_idx, :2])
        ).item())
    else:
        return None

    # Expected SVL (mm) from reference
    expected_svl_mm = None
    if ('snout', 'tail1') in skeletal_links_lengths:
        expected_svl_mm = float(skeletal_links_lengths[('snout', 'tail1')])
    elif ('tail1', 'snout') in skeletal_links_lengths:
        expected_svl_mm = float(skeletal_links_lengths[('tail1', 'snout')])

    if expected_svl_mm is None or expected_svl_mm <= 0:
        return None

    return gt_svl_pix / expected_svl_mm  # pixels per mm


def apply_skeletal_target_masking(
    target: dict,
    batch_annotations: dict,
    skeletal_data: dict,
    bodyparts: list[str],
    device: torch.device,
    stride: float = 4.0,
    skeletal_radius_multiplier: float = 1.0,
    union_intersect_adjacent_skeletal_mask_alpha: float = 0.5,
) -> dict:
    """
    Truncate gaussian targets using skeletal limb lengths.
    Fixes:
      - respects channels-first (B,C,H,W) layout
      - converts mm->pixels via per-image SVL when available
    """
    if not skeletal_data or len(skeletal_data.get('links', [])) == 0:
        return target

    heat, channels_first, hw = _get_heat(target, head="bodypart")
    if heat is None:
        return target
    H, W = hw

    # limb neighborhood definition
    limb_adj = {
        'left_elbow':  ['left_shoulder', 'left_wrist'],
        'right_elbow': ['right_shoulder', 'right_wrist'],
        'left_wrist':  ['left_elbow'],
        'right_wrist': ['right_elbow'],
        'left_knee':   ['left_hip', 'left_ankle'],
        'right_knee':  ['right_hip', 'right_ankle'],
        'left_ankle':  ['left_knee'],
        'right_ankle': ['right_knee'],
    }

    B = heat.shape[0]
    for b in range(B):
        # per-sample skeletal refs (as lists)
        if isinstance(skeletal_data['links'], (list, tuple)) and b < len(skeletal_data['links']):
            sample_links = skeletal_data['links'][b]
            sample_lengths = skeletal_data['link_lengths'][b] if b < len(skeletal_data['link_lengths']) else []
        else:
            continue

        if not sample_links or b >= len(batch_annotations['keypoints']):
            continue

        # first animal only (single-animal assumption)
        animal_kpts = batch_annotations['keypoints'][b][0]  # (num_joints, 3)

        # build length lookup in mm for all directed pairs
        link_len = {}
        for li, link in enumerate(sample_links):
            if li >= len(sample_lengths):
                continue
            if isinstance(link, torch.Tensor):
                if link.numel() < 2: 
                    continue
                i1, i2 = int(link[0].item()), int(link[1].item())
            else:
                if len(link) < 2:
                    continue
                i1, i2 = int(link[0]), int(link[1])
            if i1 >= len(bodyparts) or i2 >= len(bodyparts):
                continue
            L = sample_lengths[li]
            if isinstance(L, torch.Tensor):
                if L.numel() != 1: 
                    continue
                L = float(L.item())
            else:
                L = float(L)
            if not (L > 0):
                continue
            bp1, bp2 = bodyparts[i1], bodyparts[i2]
            link_len[(bp1, bp2)] = L
            link_len[(bp2, bp1)] = L

        # mm->pixel scale via SVL (if present/visible)
        px_per_mm = _compute_scale_factor_svl(animal_kpts, bodyparts, link_len)

        # precompute grid once per sample
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        for limb_name, adj_list in limb_adj.items():
            if limb_name not in bodyparts:
                continue
            limb_idx = bodyparts.index(limb_name)

            circular_masks = []
            for adj_name in adj_list:
                if adj_name not in bodyparts:
                    continue
                adj_idx = bodyparts.index(adj_name)
                # require adjacent visible
                if float(animal_kpts[adj_idx, 2]) < 0.5:
                    continue

                adj_x_img = float(animal_kpts[adj_idx, 0])
                adj_y_img = float(animal_kpts[adj_idx, 1])

                # expected limb length (mm) -> pixels via SVL if possible
                L_mm = link_len.get((adj_name, limb_name), None)
                if L_mm is None:
                    continue

                if px_per_mm is not None:
                    L_px = L_mm * px_per_mm
                else:
                    # Fallback: use GT distance (simple mode behavior)
                    limb_idx2 = bodyparts.index(limb_name)
                    if float(animal_kpts[limb_idx2, 2]) < 0.5:
                        continue
                    L_px = ((adj_x_img - float(animal_kpts[limb_idx2, 0]))**2 +
                            (adj_y_img - float(animal_kpts[limb_idx2, 1]))**2) ** 0.5
                    if L_px < 1.0:
                        continue

                # map to heatmap coordinates
                cx = torch.tensor(adj_x_img / stride, device=device, dtype=torch.float32)
                cy = torch.tensor(adj_y_img / stride, device=device, dtype=torch.float32)
                r = torch.tensor(L_px / stride * skeletal_radius_multiplier,
                                 device=device, dtype=torch.float32)

                dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
                circular_masks.append((dist <= r).float())

            if not circular_masks:
                continue

            if len(circular_masks) == 1:
                mask = circular_masks[0]
            else:
                stacked = torch.stack(circular_masks, dim=0)  # (M,H,W)
                union_mask = torch.max(stacked, dim=0)[0]
                inter_mask = torch.min(stacked, dim=0)[0]
                alpha = float(union_intersect_adjacent_skeletal_mask_alpha)
                mask = (1.0 - alpha) * union_mask + alpha * inter_mask

            _apply_mask_inplace(heat, b, limb_idx, mask, channels_first=channels_first)

    return target


def apply_skeletal_target_masking_simple(
    target: dict,
    batch_annotations: dict,
    bodyparts: list[str],
    device: torch.device,
    stride: float = 4.0,  # Default stride for ResNet-based models
    skeletal_radius_multiplier: float = 1.0,
    union_intersect_adjacent_skeletal_mask_alpha: float = 0.5
) -> dict:
    """
    Apply skeletal-aware masking to target heatmaps using GT landmark distances.

    Similar to apply_skeletal_target_masking but estimates limb lengths dynamically
    from GT landmark coordinates instead of using reference skeletal data.

    For limb landmarks (elbows, wrists, knees, ankles), creates circular masks around
    adjacent landmarks with radius equal to the actual distance between GT landmarks.
    The masks from adjacent landmarks are combined and then multiplied with the target heatmap.

    Args:
        target: Target dictionary containing heatmap targets
        batch_annotations: Batch annotations containing keypoint positions
        bodyparts: List of bodypart names to get indices
        device: Device to put tensors on
        stride: Model stride for coordinate conversion
        skeletal_radius_multiplier: Multiplier for limb length radii
        union_intersect_adjacent_skeletal_mask_alpha: Interpolation factor between
                                                    union (0.0) and intersection (1.0).
                                                    0.0 = pure union (OR), 1.0 = pure intersection (AND)

    Returns:
        Modified target dictionary with masked heatmaps
    """
    if "bodypart" not in target or "heatmap" not in target["bodypart"] or "target" not in target["bodypart"]["heatmap"]:
        return target  # Return unmodified if target structure is unexpected

    heatmap_targets = target["bodypart"]["heatmap"]["target"]  # Shape: (batch, height, width, num_joints)
    batch_size, height, width, num_joints = heatmap_targets.shape

    # Define limb landmarks and their adjacent landmarks (same as original function)
    limb_landmark_mapping = {
        'left_elbow': ['left_shoulder', 'left_wrist'],
        'right_elbow': ['right_shoulder', 'right_wrist'],
        'left_wrist': ['left_elbow'],
        'right_wrist': ['right_elbow'],
        'left_knee': ['left_hip', 'left_ankle'],
        'right_knee': ['right_hip', 'right_ankle'],
        'left_ankle': ['left_knee'],
        'right_ankle': ['right_knee']
    }

    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Get keypoint annotations for this sample
        if batch_idx >= len(batch_annotations['keypoints']):
            continue  # Skip if no annotations for this sample

        keypoints = batch_annotations['keypoints'][batch_idx]  # Shape: (num_animals, num_joints, 3)
        if len(keypoints) == 0:
            continue  # Skip if no keypoints

        # Use first animal (assuming single animal)
        animal_keypoints = keypoints[0]  # Shape: (num_joints, 3)

        # Apply masking to each limb landmark
        for limb_name, adjacent_landmarks in limb_landmark_mapping.items():
            if limb_name not in bodyparts:
                continue

            limb_idx = bodyparts.index(limb_name)

            # Check if limb landmark is visible
            if animal_keypoints[limb_idx, 2] < 0.5:  # visibility threshold
                continue

            # Get limb landmark position
            limb_x = animal_keypoints[limb_idx, 0].item()
            limb_y = animal_keypoints[limb_idx, 1].item()

            # Collect all circular masks for interpolation
            circular_masks = []
            mask_applied = False

            # Add circular masks from each adjacent landmark
            for adj_name in adjacent_landmarks:
                if adj_name not in bodyparts:
                    continue

                adj_idx = bodyparts.index(adj_name)

                # Check if adjacent keypoint is visible
                if animal_keypoints[adj_idx, 2] < 0.5:  # visibility threshold
                    continue

                # Get adjacent landmark position (in image coordinates)
                adj_x = animal_keypoints[adj_idx, 0].item()
                adj_y = animal_keypoints[adj_idx, 1].item()

                # Calculate actual distance between limb and adjacent landmark (GT limb length)
                gt_limb_length = ((limb_x - adj_x) ** 2 + (limb_y - adj_y) ** 2) ** 0.5

                # Skip if landmarks are too close (likely annotation error)
                if gt_limb_length < 1.0:  # minimum 1 pixel distance
                    continue

                # Convert adjacent landmark to heatmap coordinates using stride
                heatmap_x = adj_x / stride
                heatmap_y = adj_y / stride

                # Calculate radius in heatmap coordinates using GT limb length
                radius = gt_limb_length / stride * skeletal_radius_multiplier

                # Create circular mask
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(height, device=device),
                    torch.arange(width, device=device),
                    indexing='ij'
                )

                # Calculate distance from adjacent landmark
                distances = torch.sqrt((x_coords - heatmap_x) ** 2 + (y_coords - heatmap_y) ** 2)

                # Create circular mask (1 inside circle, 0 outside)
                circle_mask = (distances <= radius).float()

                # Collect mask for interpolation
                circular_masks.append(circle_mask)
                mask_applied = True

            # Apply mask to target heatmap
            if mask_applied and len(circular_masks) > 0:
                # Interpolate between union and intersection
                if len(circular_masks) == 1:
                    # Single mask - no interpolation needed
                    mask = circular_masks[0]
                else:
                    # Multiple masks - interpolate between union and intersection
                    # Stack masks for efficient computation
                    stacked_masks = torch.stack(circular_masks, dim=0)  # Shape: (num_masks, height, width)

                    # Compute union (maximum across masks)
                    union_mask = torch.max(stacked_masks, dim=0)[0]

                    # Compute intersection (minimum across masks)
                    intersection_mask = torch.min(stacked_masks, dim=0)[0]

                    # Interpolate: mask = (1 - alpha) * union + alpha * intersection
                    alpha = union_intersect_adjacent_skeletal_mask_alpha
                    mask = (1.0 - alpha) * union_mask + alpha * intersection_mask

                # Multiply target heatmap by mask
                heatmap_targets[batch_idx, :, :, limb_idx] *= mask
            # If no mask was applied, leave target unchanged (equivalent to mask of all 1's)

    return target


def compute_skeletal_constraint_loss(
    predicted_keypoints: torch.Tensor,
    skeletal_data: dict,
    bodyparts: list[str],
    device: torch.device,
    loss_weight: float = 1.0,
    radius_multiplier: float = 1.0,
) -> torch.Tensor:
    """
    Compute skeletal constraint loss based on expected limb lengths.

    Args:
        predicted_keypoints: Tensor of shape (batch_size, num_animals, num_joints, 3)
                           where last dim is [x, y, visibility]
        skeletal_data: Dict containing 'links' and 'link_lengths' arrays
        bodyparts: List of bodypart names to get indices
        device: Device to put tensors on
        loss_weight: Weight for the skeletal loss
        radius_multiplier: Multiplier for the skeletal radius

    Returns:
        Skeletal constraint loss tensor
    """
    if not skeletal_data or len(skeletal_data.get('links', [])) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    batch_size = predicted_keypoints.shape[0]
    sample_losses = []

    # Get indices for snout and tail1 for normalization
    try:
        snout_idx = bodyparts.index('snout')
        tail1_idx = bodyparts.index('tail1')
    except ValueError:
        # If snout or tail1 not in bodyparts, return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    for batch_idx in range(batch_size):
        # Get skeletal data for this sample
        if isinstance(skeletal_data['links'], torch.Tensor):
            # If it's already a tensor, assume it's the same for all samples in batch
            links = skeletal_data['links']
            link_lengths = skeletal_data['link_lengths']
        elif isinstance(skeletal_data['links'], (list, np.ndarray)):
            # If it's a list/array, get the data for this batch index
            if len(skeletal_data['links']) > batch_idx:
                links = skeletal_data['links'][batch_idx]
                link_lengths = skeletal_data['link_lengths'][batch_idx]
            else:
                # Use the same data for all samples (broadcast)
                links = skeletal_data['links'][0] if len(skeletal_data['links']) > 0 else []
                link_lengths = skeletal_data['link_lengths'][0] if len(skeletal_data['link_lengths']) > 0 else []
        else:
            links = skeletal_data['links']
            link_lengths = skeletal_data['link_lengths']

        if len(links) == 0:
            continue

        # Convert to tensors if needed
        if not isinstance(links, torch.Tensor):
            links = torch.tensor(links, device=device, dtype=torch.long)
        if not isinstance(link_lengths, torch.Tensor):
            link_lengths = torch.tensor(link_lengths, device=device, dtype=torch.float32)

        # Get keypoints for this sample (assuming single animal for now)
        kpts = predicted_keypoints[batch_idx, 0]  # Shape: (num_joints, 3)

        # Check if snout and tail1 are visible and valid for normalization
        snout_vis = kpts[snout_idx, 2] > 0.5  # visibility threshold
        tail1_vis = kpts[tail1_idx, 2] > 0.5

        if not (snout_vis and tail1_vis):
            # Cannot normalize, skip this sample
            continue

        # Compute normalization factor (snout to tail1 distance)
        snout_pos = kpts[snout_idx, :2]  # [x, y]
        tail1_pos = kpts[tail1_idx, :2]  # [x, y]
        svl_distance = torch.norm(snout_pos - tail1_pos)

        if svl_distance < 1e-6:  # Avoid division by zero
            continue

        link_losses = []

        for link_idx in range(len(links)):
            # Check bounds for both links and link_lengths
            if link_idx >= len(link_lengths):
                continue  # Skip if no corresponding length data

            link = links[link_idx]
            expected_length = link_lengths[link_idx]

            # Handle different tensor/list formats
            if isinstance(link, torch.Tensor):
                if link.dim() == 0:  # 0-d tensor (scalar)
                    continue  # Skip invalid links
                elif link.dim() == 1 and len(link) >= 2:
                    bp1_idx, bp2_idx = link[0].item(), link[1].item()
                else:
                    continue  # Skip invalid links
            elif isinstance(link, (list, tuple)) and len(link) >= 2:
                bp1_idx, bp2_idx = link[0], link[1]
            else:
                continue  # Skip invalid links

            # Handle expected_length format
            if isinstance(expected_length, torch.Tensor):
                if expected_length.dim() == 0:
                    expected_length = expected_length.item()
                else:
                    continue  # Skip invalid expected lengths

            # Skip if expected length is NaN or invalid
            if torch.isnan(torch.tensor(expected_length)) or expected_length <= 0:
                continue

            # Check if both keypoints are visible
            bp1_vis = kpts[bp1_idx, 2] > 0.5
            bp2_vis = kpts[bp2_idx, 2] > 0.5

            if not (bp1_vis and bp2_vis):
                continue

            # Compute predicted distance
            bp1_pos = kpts[bp1_idx, :2]
            bp2_pos = kpts[bp2_idx, :2]
            predicted_distance = torch.norm(bp1_pos - bp2_pos)

            # Normalize both distances by predicted SVL to make them scale-invariant
            normalized_predicted = predicted_distance / svl_distance

            # For the expected length, we need to normalize it by the expected SVL
            # to get the relative proportion, then we can compare with the predicted proportion
            expected_svl_length = None

            # Try to find SVL link in the current links to get expected SVL
            for svl_idx in range(len(links)):
                svl_bp1, svl_bp2 = links[svl_idx]
                if (svl_bp1 == snout_idx and svl_bp2 == tail1_idx) or (svl_bp1 == tail1_idx and svl_bp2 == snout_idx):
                    expected_svl_length = link_lengths[svl_idx]
                    break

            if expected_svl_length is not None and expected_svl_length > 0:
                # Normalize expected length by expected SVL to get relative proportion
                normalized_expected = expected_length / expected_svl_length
            else:
                # If we can't find expected SVL, skip this constraint
                continue

            # Compute constraint loss: 0 if pred <= expected, (pred - expected)^2 if pred > expected
            diff = normalized_predicted - normalized_expected * radius_multiplier
            link_loss = torch.where(diff > 0, diff ** 2, torch.tensor(0.0, device=device))
            link_losses.append(link_loss)

        if len(link_losses) > 0:
            # Stack and average the losses to avoid in-place operations
            sample_loss = torch.stack(link_losses).mean()
            sample_losses.append(sample_loss)

    if len(sample_losses) > 0:
        # Stack and average all sample losses to avoid in-place operations
        total_loss = torch.stack(sample_losses).mean() * loss_weight
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss

def compute_skeletal_constraint_loss2(
    predicted_keypoints: torch.Tensor,
    skeletal_data: dict,
    bodyparts: list[str],
    device: torch.device,
    loss_weight: float = 1.0,
    radius_multiplier: float = 1.0,
) -> torch.Tensor:
    if not skeletal_data or len(skeletal_data.get('links', [])) == 0:
        return torch.zeros((), device=device)

    # required indices
    try:
        snout_idx = bodyparts.index('snout')
        tail1_idx = bodyparts.index('tail1')
    except ValueError:
        return torch.zeros((), device=device)

    B = predicted_keypoints.shape[0]
    sample_losses = []

    for b in range(B):
        # per-batch links / lengths (list/ndarray/tensor -> tensors on device)
        links_b = skeletal_data['links']
        lens_b  = skeletal_data['link_lengths']

        if isinstance(links_b, (list, np.ndarray)):
            links_b = links_b[b] if len(links_b) > b else links_b[0] if len(links_b) else []
        if isinstance(lens_b, (list, np.ndarray)):
            lens_b = lens_b[b] if len(lens_b) > b else lens_b[0] if len(lens_b) else []

        links_t = torch.as_tensor(links_b, device=device, dtype=torch.long)
        lens_t  = torch.as_tensor(lens_b, device=device, dtype=torch.float32)
        if links_t.numel() == 0 or lens_t.numel() == 0:
            continue
        if links_t.ndim == 1:
            links_t = links_t.view(-1, 2)

        # predicted kpts for (single) animal 0
        kpts = predicted_keypoints[b, 0]  # (J, 3)

        # need snout & tail1 visible to normalize
        if not (kpts[snout_idx, 2] > 0.5 and kpts[tail1_idx, 2] > 0.5):
            continue

        svl = torch.norm(kpts[snout_idx, :2] - kpts[tail1_idx, :2])
        if torch.isnan(svl) or svl < 1e-6:
            continue

        # find expected SVL in link set (handles both directions)
        a = links_t[:, 0]
        b2 = links_t[:, 1]
        svl_mask = ((a == snout_idx) & (b2 == tail1_idx)) | ((a == tail1_idx) & (b2 == snout_idx))
        if not torch.any(svl_mask):
            # no reference SVL => skip constraints (unchanged from your intent)
            continue
        expected_svl = lens_t[svl_mask][0]
        if not torch.isfinite(expected_svl) or expected_svl <= 0:
            continue

        # accumulate hinge^2 only when predicted proportion exceeds allowed proportion
        link_losses = []
        for i in range(links_t.shape[0]):
            bp1 = int(links_t[i, 0].item()); bp2 = int(links_t[i, 1].item())
            exp_len = lens_t[i]
            if not torch.isfinite(exp_len) or exp_len <= 0:
                continue

            # require both endpoints visible
            if not (kpts[bp1, 2] > 0.5 and kpts[bp2, 2] > 0.5):
                continue

            pred_d = torch.norm(kpts[bp1, :2] - kpts[bp2, :2])
            if torch.isnan(pred_d):
                continue

            # normalize by predicted SVL (pred proportion) vs expected proportion
            pred_prop = pred_d / svl
            exp_prop  = (exp_len / expected_svl) * radius_multiplier

            diff = pred_prop - exp_prop
            link_losses.append(torch.where(diff > 0, diff * diff, torch.zeros_like(diff)))

        if link_losses:
            sample_losses.append(torch.stack(link_losses).mean())

    if sample_losses:
        return torch.stack(sample_losses).mean() * float(loss_weight)
    return torch.zeros((), device=device)


class TrainingRunner(Runner, Generic[ModelType], metaclass=ABCMeta):
    """Base TrainingRunner class.

    A TrainingRunner is used to fit models to datasets. Subclasses must implement the
    ``step(self, batch, mode)`` method, which performs a single training or validation
    step on a batch of data. The step is different depending on the model type (e.g.
    a pose model step vs. an object detector step).

    Args:
        model: The model to fit.
        optimizer: The optimizer to use to fit the model.
        snapshot_manager: Manages how snapshots are saved to disk during training.
        device: The device on which to run training (e.g. 'cpu', 'cuda', 'cuda:0').
        gpus: Used to specify the GPU indices for multi-GPU training (e.g. [0, 1, 2, 3]
            to train on 4 GPUs). When a GPUs list is given, the device must be 'cuda'.
        eval_interval: The interval at which the model will be evaluated while training
            (e.g. `eval_interva=5` means the model will be evaluated every 5 epochs).
        snapshot_path: If continuing to train a model, the path to the snapshot to
            resume training from.
        scheduler: The learning rate scheduler (or it's configuration), if one should be
            used.
        load_scheduler_state_dict: When resuming training (snapshot_path is not None),
            attempts to load the scheduler state dict from the snapshot. If you've
            modified your scheduler, set this to False or the old scheduler parameters
            might be used.
        logger: Logger to monitor training (e.g. a WandBLogger).
        log_filename: Name of the file in which to store training stats.
        load_weights_only: Value for the torch.load() `weights_only` parameter if
            `snapshot_path` is not None.
            If False, the python pickle module is used implicitly, which is known to
            be insecure. Only set to False if you're loading data that you trust
            (e.g. snapshots that you created yourself). For more information, see:
                https://pytorch.org/docs/stable/generated/torch.load.html
            If None, the default value is used:
                `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`
    """

    def __init__(
        self,
        model: ModelType,
        optimizer: dict | torch.optim.Optimizer,
        snapshot_manager: TorchSnapshotManager,
        device: str = "cpu",
        gpus: list[int] | None = None,
        eval_interval: int = 1,
        snapshot_path: str | Path | None = None,
        scheduler: dict | torch.optim.lr_scheduler.LRScheduler | None = None,
        load_scheduler_state_dict: bool = True,
        logger: BaseLogger | None = None,
        log_filename: str = "learning_stats.csv",
        load_weights_only: bool | None = None,
        model_cfg: dict | None = None,
    ):
        super().__init__(
            model=model, device=device, gpus=gpus, snapshot_path=snapshot_path
        )
        if isinstance(optimizer, dict):
            optimizer = build_optimizer(model, optimizer)
        if isinstance(scheduler, dict):
            scheduler = schedulers.build_scheduler(scheduler, optimizer)

        self.eval_interval = eval_interval
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.snapshot_manager = snapshot_manager
        self.model_cfg = model_cfg or {}
        self.history: dict[str, list] = dict(train_loss=[], eval_loss=[])
        self.csv_logger = CSVLogger(
            train_folder=snapshot_manager.model_folder,
            log_filename=log_filename,
        )
        self.logger = logger
        self.starting_epoch = 0
        self.current_epoch = 0

        # some models cannot compute a validation loss (e.g. detectors)
        self._print_valid_loss = True

        if self.snapshot_path:
            snapshot = self.load_snapshot(
                self.snapshot_path,
                self.device,
                self.model,
                weights_only=load_weights_only,
            )
            self.starting_epoch = snapshot.get("metadata", {}).get("epoch", 0)

            if "optimizer" in snapshot:
                self.optimizer.load_state_dict(snapshot["optimizer"])

            self._load_scheduler_state_dict(load_scheduler_state_dict, snapshot)

        self._metadata = dict(epoch=self.starting_epoch, metrics=dict(), losses=dict())
        self._epoch_ground_truth = {}
        self._epoch_predictions = {}

    def state_dict(self) -> dict:
        """Returns: the state dict for the runner"""
        model = self.model
        if self._data_parallel:
            model = self.model.module

        state_dict_ = dict(
            metadata=self._metadata,
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        if self.scheduler is not None:
            state_dict_["scheduler"] = self.scheduler.state_dict()

        return state_dict_

    @abstractmethod
    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step

        Args:
            batch: the batch data on which to run a step
            mode: "train" or "eval". Defaults to "train".

        Raises:
            ValueError: if mode is not in {"train", "eval"}

        Returns:
            A dictionary containing the different losses for the step
        """

    @abstractmethod
    def _compute_epoch_metrics(self) -> dict[str, float]:
        """Computes the metrics using the data accumulated during an epoch

        Returns:
            A dictionary containing the different losses for the step
        """
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        display_iters: int,
    ) -> None:
        """Train model for the specified number of steps.

        Args:
            train_loader: Data loader, which is an iterator over train instances.
                Each batch contains image tensor and heat maps tensor input samples.
            valid_loader: Data loader used for validation of the model.
            epochs: The number of training epochs.
            display_iters: The number of iterations between each loss print

        Example:
           runner = Runner(model, optimizer, cfg, device='cuda')
           runner.fit(train_loader, valid_loader, "example/models" epochs=50)
        """
        if self._data_parallel:
            self.model = DataParallel(self.model, device_ids=self._gpus).cuda()
        else:
            self.model.to(self.device)

        if isinstance(self.logger, ImageLoggerMixin):
            self.logger.select_images_to_log(train_loader, valid_loader)

        # continuing to train a model: either total epochs or extra epochs
        if self.starting_epoch > 0:
            epochs = self.starting_epoch + epochs

        self.union_intersect_adjacent_skeletal_mask_alpha_start = 0.0
        if "union_intersect_adjacent_skeletal_mask_alpha_start" in self.model_cfg:
            self.union_intersect_adjacent_skeletal_mask_alpha_start = self.model_cfg[
                "union_intersect_adjacent_skeletal_mask_alpha_start"
            ]
        self.union_intersect_adjacent_skeletal_mask_alpha_end = 1.0
        if "union_intersect_adjacent_skeletal_mask_alpha_end" in self.model_cfg:
            self.union_intersect_adjacent_skeletal_mask_alpha_end = self.model_cfg[
                "union_intersect_adjacent_skeletal_mask_alpha_end"
            ]

        self.skeletal_radius_multiplier_start = 1.1
        if "skeletal_radius_multiplier_start" in self.model_cfg:
            self.skeletal_radius_multiplier_start = self.model_cfg[
                "skeletal_radius_multiplier_start"
            ]
        self.skeletal_radius_multiplier_end = 1.1
        if "skeletal_radius_multiplier_end" in self.model_cfg:
            self.skeletal_radius_multiplier_end = self.model_cfg[
                "skeletal_radius_multiplier_end"
            ]

        self.skeletal_loss_radius_multiplier = 1.0
        if "skeletal_loss_radius_multiplier" in self.model_cfg:
            self.skeletal_loss_radius_multiplier = self.model_cfg[
                "skeletal_loss_radius_multiplier"
            ]

        self.truncate_targets = False
        if "truncate_targets" in self.model_cfg:
            self.truncate_targets = self.model_cfg["truncate_targets"]

        self.union_intersect_adjacent_skeletal_mask_alpha = self.union_intersect_adjacent_skeletal_mask_alpha_start
        self.skeletal_radius_multiplier = self.skeletal_radius_multiplier_start

        union_intersect_adjacent_skeletal_mask_start_epoch = 50
        if "union_intersect_adjacent_skeletal_mask_start_epoch" in self.model_cfg:
            union_intersect_adjacent_skeletal_mask_start_epoch = self.model_cfg[
                "union_intersect_adjacent_skeletal_mask_start_epoch"
            ]
        union_intersect_adjacent_skeletal_mask_end_epoch = 150
        if "union_intersect_adjacent_skeletal_mask_end_epoch" in self.model_cfg:
            union_intersect_adjacent_skeletal_mask_end_epoch = self.model_cfg[
                "union_intersect_adjacent_skeletal_mask_end_epoch"
            ]

        n_union_intersection_steps = union_intersect_adjacent_skeletal_mask_end_epoch - union_intersect_adjacent_skeletal_mask_start_epoch

        self.use_skeletal_reference = False
        if "use_skeletal_reference" in self.model_cfg:
            self.use_skeletal_reference = self.model_cfg["use_skeletal_reference"]

        for e in range(self.starting_epoch + 1, epochs + 1):
            self.current_epoch = e
            self._metadata["epoch"] = e
            train_loss = self._epoch(
                train_loader, mode="train", display_iters=display_iters
            )
            if self.scheduler:
                self.scheduler.step()

            # Interpolate between union and intersection
            if e < epochs:
                t = (e - union_intersect_adjacent_skeletal_mask_start_epoch) / n_union_intersection_steps
                t = np.clip(t, 0, 1)
                self.union_intersect_adjacent_skeletal_mask_alpha = (
                    self.union_intersect_adjacent_skeletal_mask_alpha_start + (
                        self.union_intersect_adjacent_skeletal_mask_alpha_end - self.union_intersect_adjacent_skeletal_mask_alpha_start
                        ) * t)
                self.skeletal_radius_multiplier = (
                    self.skeletal_radius_multiplier_start + (
                        self.skeletal_radius_multiplier_end - self.skeletal_radius_multiplier_start
                        ) * t)

            lr = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {e}/{epochs} (lr={lr}), train loss {float(train_loss):.5f}"
            if e % self.eval_interval == 0:
                with torch.no_grad():
                    logging.info(f"Training for epoch {e} done, starting evaluation")
                    valid_loss = self._epoch(
                        valid_loader, mode="eval", display_iters=display_iters
                    )
                    if self._print_valid_loss:
                        msg += f", valid loss {float(valid_loss):.5f}"

            self.snapshot_manager.update(e, self.state_dict(), last=(e == epochs))
            logging.info(msg)

            epoch_metrics = self._metadata.get("metrics")
            if (
                e % self.eval_interval == 0
                and epoch_metrics is not None
                and len(epoch_metrics) > 0
            ):
                logging.info(f"Model performance:")
                line_length = max([len(name) for name in epoch_metrics.keys()]) + 2
                for name, score in epoch_metrics.items():
                    logging.info(f"  {(name + ':').ljust(line_length)}{score:6.2f}")

    def _epoch(
        self,
        loader: torch.utils.data.DataLoader,
        mode: str = "train",
        display_iters: int = 500,
    ) -> float:
        """Facilitates training over an epoch. Returns the loss over the batches.

        Args:
            loader: Data loader, which is an iterator over instances.
                Each batch contains image tensor and heat maps tensor input samples.
            mode: str identifier to instruct the Runner whether to train or evaluate.
                Possible values are: "train" or "eval".
            display_iters: the number of iterations between each loss print

        Raises:
            ValueError: When the given mode is invalid

        Returns:
            epoch_loss: Average of the loss over the batches.
        """
        if mode == "train":
            self.model.train()
        elif mode == "eval" or mode == "inference":
            self.model.eval()
        else:
            raise ValueError(f"Runner mode must be train or eval, found mode={mode}.")

        epoch_loss = []
        loss_metrics = defaultdict(list)
        for i, batch in enumerate(loader):
            losses_dict = self.step(batch, mode)
            if "total_loss" in losses_dict:
                epoch_loss.append(losses_dict["total_loss"])
                if (i + 1) % display_iters == 0 and mode != "eval":
                    logging.info(
                        f"Number of iterations: {i + 1}, "
                        f"loss: {losses_dict['total_loss']:.5f}, "
                        f"lr: {self.optimizer.param_groups[0]['lr']}"
                    )

            for key in losses_dict.keys():
                loss_metrics[key].append(losses_dict[key])

        perf_metrics = None
        if mode == "eval":
            perf_metrics = self._compute_epoch_metrics()
            self._metadata["metrics"] = perf_metrics
            self._epoch_predictions = {}
            self._epoch_ground_truth = {}

        if len(epoch_loss) > 0:
            epoch_loss = np.mean(epoch_loss).item()
        else:
            epoch_loss = 0
        self.history[f"{mode}_loss"].append(epoch_loss)

        metrics_to_log = {}
        if perf_metrics:
            for name, score in perf_metrics.items():
                if not isinstance(score, (int, float)):
                    score = 0.0
                metrics_to_log[name] = score

        for key in loss_metrics:
            name = f"{mode}.{key}"
            val = float("nan")
            if np.sum(~np.isnan(loss_metrics[key])) > 0:
                val = np.nanmean(loss_metrics[key]).item()
            self._metadata["losses"][name] = val
            metrics_to_log[f"losses/{name}"] = val

        self.csv_logger.log(metrics_to_log, step=self.current_epoch)
        if self.logger:
            self.logger.log(metrics_to_log, step=self.current_epoch)

        return epoch_loss

    def _load_scheduler_state_dict(self, load_state_dict: bool, snapshot: dict) -> None:
        if self.scheduler is None:
            return

        loaded_state_dict = False
        if load_state_dict and "scheduler" in snapshot:
            try:
                schedulers.load_scheduler_state(self.scheduler, snapshot["scheduler"])
                loaded_state_dict = True
            except ValueError as err:
                logging.warning(
                    "Failed to load the scheduler state_dict. The scheduler will "
                    "restart at epoch 0. This is expected if the scheduler "
                    "configuration was edited since the original snapshot was "
                    f"trained. Error: {err}"
                )

        if not loaded_state_dict and self.starting_epoch > 0:
            logging.info(
                f"Setting the scheduler starting epoch to {self.starting_epoch}"
            )
            self.scheduler.last_epoch = self.starting_epoch


class PoseTrainingRunner(TrainingRunner[PoseModel]):
    """Runner to train pose estimation models"""

    def __init__(
        self,
        model: PoseModel,
        optimizer: torch.optim.Optimizer,
        load_head_weights: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: The neural network for solving pose estimation task.
            optimizer: A PyTorch optimizer for updating model parameters.
            load_head_weights: When `snapshot_path` is not None, whether to load the
                head weights from the saved snapshot or just the backbone weights.
            **kwargs: TrainingRunner kwargs
        """
        self._load_head_weights = load_head_weights
        super().__init__(model, optimizer, **kwargs)

    def load_snapshot(
        self,
        snapshot_path: str | Path,
        device: str,
        model: PoseModel,
        weights_only: bool | None = None,
    ) -> dict:
        """Loads the state dict for a model from a file

        This method loads a file containing a DeepLabCut PyTorch model snapshot onto
        a given device, and sets the model weights using the state_dict.

        Args:
            snapshot_path: the path containing the model weights to load
            device: the device on which the model should be loaded
            model: the model for which the weights are loaded
            weights_only: Value for torch.load() `weights_only` parameter.
                If False, the python pickle module is used implicitly, which is known to
                be insecure. Only set to False if you're loading data that you trust
                (e.g. snapshots that you created yourself). For more information, see:
                    https://pytorch.org/docs/stable/generated/torch.load.html
                If None, the default value is used:
                    `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`

        Returns:
            The content of the snapshot file.
        """
        snapshot = attempt_snapshot_load(snapshot_path, device, weights_only)
        if self._load_head_weights:
            model.load_state_dict(snapshot["model"])
        else:
            backbone_prefix = "backbone."
            backbone_weights = {
                k[len(backbone_prefix) :]: v
                for k, v in snapshot["model"].items()
                if k.startswith(backbone_prefix)
            }
            model.backbone.load_state_dict(backbone_weights)

        return snapshot

    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Runner must be in train or eval mode, but {mode} was found."

        Returns:
            dict: {
                "total_loss": aggregate_loss,
                "aux_loss_1": loss_value,
                ...,
            }
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"BottomUpSolver must be in train or eval mode, but {mode} was found."
            )

        if mode == "train":
            self.optimizer.zero_grad()

        inputs = batch["image"]
        inputs = inputs.to(self.device).float()
        if 'cond_keypoints' in batch['context']:
            cond_kpts = batch['context']['cond_keypoints']
            outputs = self.model(inputs, cond_kpts=cond_kpts)
        else:
            outputs = self.model(inputs)

        if self._data_parallel:
            underlying_model = self.model.module
        else:
            underlying_model = self.model

        target = underlying_model.get_target(outputs, batch["annotations"])

        # Apply skeletal-aware masking to target heatmaps if skeletal data is available
        if self.truncate_targets and "skeletal_data" in batch:
            # Get bodyparts from model config metadata
            if hasattr(self, 'model_cfg') and 'metadata' in self.model_cfg and 'bodyparts' in self.model_cfg['metadata']:
                bodyparts = self.model_cfg['metadata']['bodyparts']

                skeleletal_radius_multiplier = self.skeletal_radius_multiplier
                # if hasattr(self, 'model_cfg') and 'skeletal_radius_multiplier' in self.model_cfg:
                #     skeleletal_radius_multiplier = self.model_cfg['skeletal_radius_multiplier']

                union_intersect_adjacent_skeletal_mask_alpha = self.union_intersect_adjacent_skeletal_mask_alpha
                # if hasattr(self, 'model_cfg') and 'union_intersect_adjacent_skeletal_mask_alpha' in self.model_cfg:
                #     union_intersect_adjacent_skeletal_mask_alpha = self.model_cfg['union_intersect_adjacent_skeletal_mask_alpha']

                # Get the stride for the bodypart head
                try:
                    stride = underlying_model.get_stride("bodypart")
                except (AttributeError, KeyError):
                    # Fallback to default stride if not available
                    stride = 4.0

                if self.use_skeletal_reference:
                    target = apply_skeletal_target_masking(
                        target=target,
                        batch_annotations=batch["annotations"],
                        skeletal_data=batch["skeletal_data"],
                        bodyparts=bodyparts,
                        device=self.device,
                        stride=stride,
                        skeletal_radius_multiplier=skeleletal_radius_multiplier,
                        union_intersect_adjacent_skeletal_mask_alpha=union_intersect_adjacent_skeletal_mask_alpha
                    )
                else:
                    target = apply_skeletal_target_masking_simple(
                        target=target,
                        batch_annotations=batch["annotations"],
                        bodyparts=bodyparts,
                        device=self.device,
                        stride=stride,
                        skeletal_radius_multiplier=skeleletal_radius_multiplier,
                        union_intersect_adjacent_skeletal_mask_alpha=union_intersect_adjacent_skeletal_mask_alpha
                    )

        losses_dict = underlying_model.get_loss(outputs, target)

        # Add skeletal constraint loss if skeletal data is available
        if "skeletal_data" in batch:
            # Get bodyparts from model config metadata
            bodyparts = None
            if hasattr(self, 'model_cfg') and 'metadata' in self.model_cfg:
                bodyparts = self.model_cfg['metadata'].get('bodyparts')

            if bodyparts is not None:
                # Get predicted keypoints from the model outputs
                predictions = underlying_model.get_predictions(outputs)
                if "bodypart" in predictions and "poses" in predictions["bodypart"]:
                    predicted_keypoints = predictions["bodypart"]["poses"]  # Shape: (batch, num_animals, num_joints, 3)

                    # Get skeletal loss weight from model config or use default
                    skeletal_loss_weight = 0.10  # Default weight
                    if hasattr(self, 'model_cfg') and 'skeletal_loss_weight' in self.model_cfg:
                        skeletal_loss_weight = self.model_cfg['skeletal_loss_weight']
                    elif hasattr(self, 'model_cfg') and 'train_settings' in self.model_cfg:
                        skeletal_loss_weight = self.model_cfg['train_settings'].get('skeletal_loss_weight', 0.10)

                    # Convert skeletal data from batch format to loss function format
                    skeletal_data_for_loss = {
                        "links": batch["skeletal_data"]["links"],
                        "link_lengths": batch["skeletal_data"]["link_lengths"]
                    }

                    # Get skeletal constraint loss
                    skeletal_loss = compute_skeletal_constraint_loss(
                        predicted_keypoints=predicted_keypoints,
                        skeletal_data=skeletal_data_for_loss,
                        bodyparts=bodyparts,
                        device=self.device,
                        loss_weight=skeletal_loss_weight,
                        radius_multiplier=self.skeletal_loss_radius_multiplier
                    )

                    losses_dict["skeletal_loss"] = skeletal_loss
                    # Create a new tensor to avoid in-place operations
                    losses_dict["total_loss"] = losses_dict["total_loss"] + skeletal_loss

        if mode == "train":
            losses_dict["total_loss"].backward()
            self.optimizer.step()

        if isinstance(self.logger, ImageLoggerMixin):
            self.logger.log_images(batch, outputs, target, step=self.current_epoch)

        if mode == "eval":
            predictions = {
                name: {k: v.detach().cpu().numpy() for k, v in pred.items()}
                for name, pred in underlying_model.get_predictions(outputs).items()
            }

            ground_truth = batch["annotations"]["keypoints"]
            if batch["annotations"]["with_center_keypoints"][0]:
                ground_truth = ground_truth[..., :-1, :]

            self._update_epoch_predictions(
                name="bodyparts",
                gt_keypoints=ground_truth,
                pred_keypoints=predictions["bodypart"]["poses"],
                offsets=batch["offsets"],
                scales=batch["scales"],
            )
            if "unique_bodypart" in predictions:
                self._update_epoch_predictions(
                    name="unique_bodyparts",
                    gt_keypoints=batch["annotations"]["keypoints_unique"],
                    pred_keypoints=predictions["unique_bodypart"]["poses"],
                    offsets=batch["offsets"],
                    scales=batch["scales"],
                )

        return {k: v.detach().cpu().numpy() for k, v in losses_dict.items()}

    def _compute_epoch_metrics(self) -> dict[str, float]:
        """Computes the metrics using the data accumulated during an epoch
        Returns:
            A dictionary containing the different losses for the step
        """
        scores = metrics.compute_metrics(
            ground_truth=self._epoch_ground_truth["bodyparts"],
            predictions=self._epoch_predictions["bodyparts"],
            single_animal=False,
            unique_bodypart_gt=self._epoch_ground_truth.get("unique_bodyparts"),
            unique_bodypart_poses=self._epoch_predictions.get("unique_bodyparts"),
            pcutoff=0.6,
            compute_detection_rmse=False,
        )
        return {f"metrics/test.{metric}": value for metric, value in scores.items()}

    def _update_epoch_predictions(
        self,
        name: str,
        gt_keypoints: torch.Tensor,
        pred_keypoints: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Updates the stored predictions with a new batch"""
        epoch_gt_metric = self._epoch_ground_truth.get(name, {})
        epoch_metric = self._epoch_predictions.get(name, {})
        assert len(gt_keypoints) == len(pred_keypoints)
        assert len(offsets) == len(scales)
        scales = scales.detach().cpu().numpy()
        offsets = offsets.detach().cpu().numpy()

        for gt, pred, scale, offset in zip(
            gt_keypoints,
            pred_keypoints,
            scales,
            offsets,
        ):
            ground_truth = gt.detach().cpu().numpy()
            pred = pred.copy()

            # rescale to the full image for TD or CTD
            ground_truth[..., :2] = (ground_truth[..., :2] * scale) + offset
            pred[..., :2] = (pred[..., :2] * scale) + offset

            # we don't care about image paths here - use a default index
            index = len(epoch_metric) + 1
            epoch_gt_metric[f"sample{index:09}"] = ground_truth
            epoch_metric[f"sample{index:09}"] = pred

        self._epoch_ground_truth[name] = epoch_gt_metric
        self._epoch_predictions[name] = epoch_metric


class DetectorTrainingRunner(TrainingRunner[BaseDetector]):
    """Runner to train object detection models"""

    def __init__(self, model: BaseDetector, optimizer: torch.optim.Optimizer, **kwargs):
        """
        Args:
            model: The detector model to train.
            optimizer: The optimizer to use to train the model.
            **kwargs: TrainingRunner kwargs
        """
        log_filename = "learning_stats_detector.csv"
        if "log_filename" in kwargs:
            log_filename = kwargs.pop("log_filename")

        super().__init__(model, optimizer, log_filename=log_filename, **kwargs)
        self._pycoco_warning_displayed = False
        self._print_valid_loss = False

    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Runner must be in train or eval mode, but {mode} was found."

        Returns:
            dict: {
                'total_loss': torch.Tensor,
                'aux_loss_1': torch.Tensor,
                ...,
            }
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"DetectorSolver must be in train or eval mode, but {mode} was found."
            )

        if mode == "train":
            self.optimizer.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        images = batch["image"]
        images = images.to(self.device)

        if self._data_parallel:
            underlying_model = self.model.module
        else:
            underlying_model = self.model

        target = underlying_model.get_target(batch["annotations"])
        for item in target:  # target is a list here
            for key in item:
                if item[key] is not None:
                    item[key] = item[key].to(self.device)

        losses, predictions = self.model(images, target)

        # losses only returned during training, not evaluation
        if mode == "train":
            losses["total_loss"] = sum(loss_part for loss_part in losses.values())
            losses["total_loss"].backward()
            self.optimizer.step()
            losses = {k: v.detach().cpu().numpy() for k, v in losses.items()}

        elif mode == "eval":
            losses["total_loss"] = float("nan")
            self._update_epoch_predictions(
                paths=batch["path"],
                sizes=batch["original_size"],
                bboxes=batch["annotations"]["boxes"],
                predictions=predictions,
                offsets=batch["offsets"],
                scales=batch["scales"],
            )

        return losses

    def _compute_epoch_metrics(self) -> dict[str, float]:
        """Returns: bounding box metrics, if"""
        try:
            return {
                f"metrics/test.{k}": v
                for k, v in metrics.compute_bbox_metrics(
                    self._epoch_ground_truth, self._epoch_predictions
                ).items()
            }
        except ModuleNotFoundError:
            if not self._pycoco_warning_displayed:
                logging.info(
                    "\nNote:\n"
                    "Cannot compute bounding box metrics as ``pycocotools`` is not "
                    "installed. If you want bounding box mAP metrics when training "
                    "detectors for top-down models, please run ``pip install "
                    "pycocotools``.\n"
                )
                self._pycoco_warning_displayed = True

        return {}

    def _update_epoch_predictions(
        self,
        paths: torch.Tensor,
        sizes: torch.Tensor,
        bboxes: torch.Tensor,
        predictions: list[dict[str, torch.Tensor]],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Updates the stored predictions with a new batch"""
        for img_path, img_size, img_bboxes, img_pred, scale, offset in zip(
            paths, sizes, bboxes, predictions, scales, offsets
        ):
            scale_x, scale_y = scale
            scale_factors = np.array([scale_x, scale_y, scale_x, scale_y])
            offset = np.array(offset)

            # remove bboxes that are not visible
            img_bbox_mask = (img_bboxes[:, 2] > 0.0) & (img_bboxes[:, 3] > 0.0)
            img_bboxes = img_bboxes[img_bbox_mask]

            # rescale ground truth bounding boxes
            gt_rescaled = img_bboxes.cpu().numpy() * scale_factors
            gt_rescaled[..., :2] = gt_rescaled[..., :2] + offset

            # convert to COCO format (xywh) before rescaling
            pred_rescaled = img_pred["boxes"].detach().cpu().numpy()
            pred_rescaled[:, 2] -= pred_rescaled[:, 0]
            pred_rescaled[:, 3] -= pred_rescaled[:, 1]
            pred_rescaled[..., :4] = pred_rescaled[..., :4] * scale_factors
            pred_rescaled[..., :2] = pred_rescaled[..., :2] + offset

            self._epoch_ground_truth[img_path] = {
                "bboxes": gt_rescaled,
                "width": img_size[1],
                "height": img_size[0],
            }
            self._epoch_predictions[img_path] = {
                "bboxes": pred_rescaled,
                "scores": img_pred["scores"].detach().cpu().numpy(),
            }


def build_training_runner(
    runner_config: dict,
    model_folder: Path,
    task: Task,
    model: nn.Module,
    device: str,
    gpus: list[int] | None = None,
    snapshot_path: str | Path | None = None,
    load_head_weights: bool = True,
    logger: BaseLogger | None = None,
    model_cfg: dict | None = None,
) -> TrainingRunner:
    """
    Build a runner object according to a pytorch configuration file

    Args:
        runner_config: the configuration for the runner
        model_folder: the folder where models should be saved
        task: the task the runner will perform
        model: the model to run
        device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
        gpus: the list of GPU indices to use for multi-GPU training
        snapshot_path: the snapshot from which to load the weights
        load_head_weights: When `snapshot_path` is not None and a pose model is being
            trained, whether to load the head weights from the saved snapshot.
        logger: the logger to use, if any

    Returns:
        the runner that was built
    """
    optimizer = build_optimizer(model, runner_config["optimizer"])
    scheduler = schedulers.build_scheduler(runner_config.get("scheduler"), optimizer)

    # if no custom snapshot prefix is defined, use the default one
    snapshot_prefix = runner_config.get("snapshot_prefix")
    if snapshot_prefix is None or len(snapshot_prefix) == 0:
        snapshot_prefix = task.snapshot_prefix

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        snapshot_manager=TorchSnapshotManager(
            snapshot_prefix=snapshot_prefix,
            model_folder=model_folder,
            key_metric=runner_config.get("key_metric"),
            key_metric_asc=runner_config.get("key_metric_asc"),
            max_snapshots=runner_config["snapshots"]["max_snapshots"],
            save_epochs=runner_config["snapshots"]["save_epochs"],
            save_optimizer_state=runner_config["snapshots"]["save_optimizer_state"],
        ),
        device=device,
        gpus=gpus,
        eval_interval=runner_config.get("eval_interval"),
        snapshot_path=snapshot_path,
        scheduler=scheduler,
        load_scheduler_state_dict=runner_config.get("load_scheduler_state_dict", True),
        logger=logger,
        load_weights_only=runner_config.get("load_weights_only", None),
        model_cfg=model_cfg,
    )
    if task == Task.DETECT:
        return DetectorTrainingRunner(**kwargs)

    kwargs["load_head_weights"] = load_head_weights
    return PoseTrainingRunner(**kwargs)


def build_optimizer(
    model: nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """Builds an optimizer from a configuration.

    Args:
        model: The model to optimize.
        optimizer_config: The configuration for the optimizer.

    Returns:
        The optimizer for the model built according to the given configuration.
    """
    optim_cls = getattr(torch.optim, optimizer_config["type"])
    optimizer = optim_cls(params=model.parameters(), **optimizer_config["params"])
    return optimizer
