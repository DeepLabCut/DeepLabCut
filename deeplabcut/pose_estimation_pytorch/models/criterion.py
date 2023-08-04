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

from typing import Tuple

import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

LOSSES = Registry("losses", build_func=build_from_cfg)


class WeightedMSELoss(nn.MSELoss):
    """
    Weighted Mean Squared Error (MSE) Loss.

    This loss computes the Mean Squared Error between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 (masked items)
    are excluded from the loss calculation.
    """

    def __init__(self) -> None:
        """
        Constructor of the class WeightedMSELoss
        """
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = 1
    ) -> torch.Tensor:
        """Summary:
        Compute the weighted Mean Squared Error loss.

        Args:
            prediction: predicted tensor
            target: target tensor
            weights: weights for each element in the loss calculation. Defaults to 1.

        Returns:
            Weighted Mean Squared Error Loss.
        """
        loss_item = self.mse_loss(prediction, target)
        loss_item_weighted = loss_item * weights

        loss_without_zeros = loss_item_weighted[loss_item_weighted != 0]
        if loss_without_zeros.nelement() == 0:
            return torch.tensor(0.0)
        return torch.mean(loss_without_zeros)


class WeightedHuberLoss(nn.HuberLoss):
    """
    Weighted Huber Loss.

    This loss computes the Huber loss between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 are
    excluded from the loss calculation.
    """

    def __init__(self) -> None:
        """Summary:
        Constructor of the WeightedHuberLoss class.
        """
        super(WeightedHuberLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(reduction="none")

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, weights: int = 1
    ) -> torch.Tensor:
        """Summary:
        Compute the weighted Huber loss.

        Args:
            prediction: predicted tensor
            target: target tensor
            weights: Weights for each element in the loss calculation. Defaults to 1.

        Returns:
            Weighted Huber loss.
        """
        loss_item = self.huber_loss(prediction, target)
        loss_item_weighted = loss_item * weights

        loss_without_zeros = loss_item_weighted[loss_item_weighted != 0]
        if loss_without_zeros.nelement() == 0:
            return torch.tensor(0.0)
        return torch.mean(loss_without_zeros)


class WeightedBCELoss(nn.BCEWithLogitsLoss):
    """
    Weighted Binary Cross Entropy (BCE) Loss.

    This loss computes the Binary Cross Entropy loss between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 are
    excluded from the loss calculation.
    """

    def __init__(self) -> None:
        """Summary:
        Constructor of the WeightedBCELoss.
        """
        super(WeightedBCELoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, weights: int = 1
    ) -> torch.Tensor:
        """Summary:
        Compute the weighted Binary Cross Entropy loss.

        Args:
            prediction: _description_
            target: _description_
            weights: _description_. Defaults to 1.

        Returns:
            Weighted Binary Cross Entropy loss.
        """
        loss_item = self.BCELoss(prediction, target)
        loss_item_weighted = loss_item * weights

        loss_without_zeros = loss_item_weighted[loss_item_weighted != 0]
        if loss_without_zeros.nelement() == 0:
            return torch.tensor(0.0)
        return torch.mean(loss_without_zeros)


@LOSSES.register_module
class PoseLoss(nn.Module):
    """
    Pose Lose Function.

    This loss function computes the weighted sum of heatmap and locref loss for keypoint detection and
    localization, respectively. The locref loss can be either Mean Squared Error (MSE) or Huber Loss,
    depending on the locref_huber_loss flag.
    """

    def __init__(
        self,
        loss_weight_locref: float = 0.1,
        locref_huber_loss: bool = False,
        apply_sigmoid: bool = False,
    ) -> None:
        """Summary:
        Constructter of the PoseLoss class.

        Args:
            loss_weight_locref: weight for loss_locref part (parsed from the pose_cfg.yaml from the dlc_models folder)
            locref_huber_loss: if True uses torch.nn.HuberLoss for locref (default is False).
            apply_sigmoid: whether to apply sigmoid to the heatmap predictions should be true
                                    for MSE, false for BCE (since it already applies it by itself)

        Returns:
            None.
        """
        super(PoseLoss, self).__init__()
        if locref_huber_loss:
            self.locref_criterion = WeightedHuberLoss()
        else:
            self.locref_criterion = WeightedMSELoss()
        self.loss_weight_locref = loss_weight_locref
        self.heatmap_criterion = WeightedBCELoss()
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, prediction: tuple, target: dict) -> tuple:
        """Summary:
        Forward pass of the Pose Loss function.

        Args:
            prediction: a tuple containing the predicted heatmap and locref of size
                    '(heatmaps, locref)' of size '(batch_size, h, w, number_keypoints), (batch_size, h, w, 2*number_keypoints)'
            target: dictionary containing the target tensors, including 'heatmaps',
                'locref_maps', 'locref_masks', and 'weights' (optional, default is None).
            {
            'heatmaps': (batch_size x number_body_parts x heatmap_size[0] x heatmap_size[1]),
            'heatmaps_ignored': (batch_size x number_body_parts x heatmap_size[0] x heatmap_size[1])
                                weights for the heatmaps
            'locref_maps': (batch_size x 2 * number_body_parts x heatmap_size[0] x heatmap_size[1]),
            'locref_masks': (batch_size x 2 * number_body_parts x heatmap_size[0] x heatmap_size[1]),
            }
        Returns:
            A tuple containing the total_loss, heatmap_loss and locref_loss.

        Examples:
            prediction = (predicted_heatmaps, predicted_locref)
            target = {
                'heatmaps': torch.tensor([batch_size, num_keypoints, h, w]),
                'locref_maps': torch.tensor([batch_size, 2 * num_keypoints, h, w]),
                'locref_masks': torch.tensor([batch_size, 2 * num_keypoints, h, w]),
                'weights': torch.tensor([batch_size, num_keypoints])
            }
            total_loss, heatmap_loss, locref_loss = criterion(prediction, target)
        """
        heatmaps, locref = prediction
        if self.apply_sigmoid:
            heatmap_loss = self.heatmap_criterion(
                self.sigmoid(heatmaps),
                target["heatmaps"],
                target.get("heatmaps_ignored", 1),
            )
        else:
            heatmap_loss = self.heatmap_criterion(
                heatmaps, target["heatmaps"], target.get("heatmaps_ignored", 1)
            )

        locref_loss = self.locref_criterion(
            locref, target["locref_maps"], target["locref_masks"]
        )
        total_loss = locref_loss * self.loss_weight_locref + heatmap_loss
        return {
            "total_loss": total_loss,
            "heatmap_loss": heatmap_loss,
            "locref_loss": locref_loss,
        }


@LOSSES.register_module
class HeatmapOnlyLoss(nn.Module):
    """
    Heatmap-Only Loss Function.

    This loss function computes the weighted Binary Cross Entropy (BCE) loss for heatmap predictions.
    """

    def __init__(self, apply_sigmoid: bool = False) -> None:
        """Summary:
        Constructor for the HeatmapOnlyLoss class.

        Args:
            apply_sigmoid: whether to apply sigmoid to the heatmap predictions should be true for MSE, false for BCE (since it already applies it by itself)

        Return:
            None
        """
        super(HeatmapOnlyLoss, self).__init__()
        self.heatmap_criterion = WeightedBCELoss()
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, prediction: tuple, target) -> torch.Tensor:
        """Summary:
        Forward pass of the Heatmap_Only Loss function.

        Args:
            prediction: tuple containing the predicted heatmap and locref of size
                            (batch_size, h, w, number_keypoints), (batch_size, h, w, 2*number_keypoints)`
            target: dictionary containing the target tensors: {
                'heatmaps': (batch_size x number_body_parts x heatmap_size[0] x heatmap_size[1]),
                'locref_maps': (batch_size x 2 * number_body_parts x heatmap_size[0] x heatmap_size[1]),
                'locref_masks': (batch_size x 2 * number_body_parts x heatmap_size[0] x heatmap_size[1]),
                'weights': (optional, default is None)
                }

        Returns:
            heatmap_loss: the computed heatmap loss.
        """
        heatmaps = prediction[0]
        if self.apply_sigmoid:
            heatmap_loss = self.heatmap_criterion(
                self.sigmoid(heatmaps),
                target["heatmaps"],
                target.get("heatmaps_ignored", 1),
            )
        else:
            heatmap_loss = self.heatmap_criterion(
                heatmaps, target["heatmaps"], target.get("heatmaps_ignored", 1)
            )

        return {
            "total_loss": heatmap_loss,
        }
