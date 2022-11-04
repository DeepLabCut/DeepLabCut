import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

LOSSES = Registry('losses', build_func=build_from_cfg)


class WeightedMSELoss(nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def __call__(self, prediction, target, weights):
        loss_item = self.mse_loss(prediction, target)
        loss_item_weighted = loss_item * weights
        return torch.mean(loss_item_weighted)

@LOSSES.register_module
class PoseLoss(nn.Module):
    def __init__(self,
                 loss_weight_locref: float = 0.1,
                 locref_huber_loss: bool = False):
        """

        Parameters
        ----------
        loss_weight_locref: float
            Weight for loss_locref part
            (parsed from the pose_cfg.yaml from the dlc_models folder)
        locref_huber_loss: bool
            If `True` uses torch.nn.HuberLoss for locref
            (default is False)

        """
        super(PoseLoss, self).__init__()
        if locref_huber_loss:
            self.locref_criterion = nn.HuberLoss()
        else:
            self.locref_criterion = WeightedMSELoss()
        self.loss_weight_locref = loss_weight_locref
        self.heatmap_criterion = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        """

        Parameters
        ----------
        prediction: tuple of Tensors `(heatmaps, locref)` of size `(batch_size, h, w, number_keypoints), (batch_size, h, w, 2*number_keypoints)`
            Predicted heatmap and locref
        target: dict = {
            'heatmaps': torch.Tensor (batch_size x number_body_parts x heatmap_size[0] x heatmap_size[1]),
            'locref_maps': torch.Tensor (batch_size x 2 * number_body_parts x heatmap_size[0] x heatmap_size[1]),
            'locref_masks': torch.Tensor (batch_size x 2 * number_body_parts x heatmap_size[0] x heatmap_size[1]),
            'weights': torch.Tensor (optional, default is None)
        }
        Returns
        -------
        loss: sum
        """
        heatmaps, locref = prediction
        heatmap_loss = self.heatmap_criterion(heatmaps,
                                              target['heatmaps'])
        locref_loss = self.loss_weight_locref * self.locref_criterion(locref,
                                                                      target['locref_maps'],
                                                                      target['locref_masks'])
        total_loss = locref_loss + heatmap_loss
        return total_loss, heatmap_loss, locref_loss
