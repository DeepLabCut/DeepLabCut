import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

LOSSES = Registry('losses', build_func=build_from_cfg)


class WeightedMSELoss(nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        loss_item = self.mse_loss(prediction, target)
        loss_item_weighted = loss_item * weights

        loss_without_zeros = loss_item_weighted[loss_item_weighted != 0]
        if loss_without_zeros.nelement() == 0:
            return torch.tensor(0.)
        return torch.mean(loss_without_zeros)
    
class WeightedHuberLoss(nn.HuberLoss):

    def __init__(self):
        super(WeightedHuberLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        loss_item = self.huber_loss(prediction, target)
        loss_item_weighted = loss_item*weights

        loss_without_zeros = loss_item_weighted[loss_item_weighted != 0]
        if loss_without_zeros.nelement() == 0:
            return torch.tensor(0.)
        return torch.mean(loss_without_zeros)

class WeightedBCELoss(nn.BCEWithLogitsLoss):
    
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, prediction, target, weights=1):
        loss_item = self.BCELoss(prediction, target)
        loss_item_weighted = loss_item*weights

        loss_without_zeros = loss_item_weighted[loss_item_weighted != 0]
        if loss_without_zeros.nelement() == 0:
            return torch.tensor(0.)
        return torch.mean(loss_without_zeros)

@LOSSES.register_module
class PoseLoss(nn.Module):
    def __init__(self,
                 loss_weight_locref: float = 0.1,
                 locref_huber_loss: bool = False,
                 apply_sigmoid: bool= False):
        """

        Parameters
        ----------
        loss_weight_locref: float
            Weight for loss_locref part
            (parsed from the pose_cfg.yaml from the dlc_models folder)
        locref_huber_loss: bool
            If `True` uses torch.nn.HuberLoss for locref
            (default is False)
        apply_sigmoid : wether to apply sigmoid to the heatmap predictions should be true for MSE, false for BCE (since it already applies it by itself)

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
        if self.apply_sigmoid:
            heatmap_loss = self.heatmap_criterion(self.sigmoid(heatmaps),
                                              target['heatmaps'],
                                              target.get('heatmaps_ignored', 1))
        else:
            heatmap_loss = self.heatmap_criterion(heatmaps,
                                              target['heatmaps'],
                                              target.get('heatmaps_ignored', 1))
        
        locref_loss = self.locref_criterion(locref,
                                            target['locref_maps'],
                                            target['locref_masks'])
        total_loss = locref_loss*self.loss_weight_locref + heatmap_loss
        return total_loss, heatmap_loss, locref_loss
