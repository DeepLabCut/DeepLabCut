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
import torch


def easy_triplet_loss():
    def loss_func(anchor, positive, neg):
        triplet_loss = torch.nn.TripletMarginLoss()
        loss = triplet_loss(anchor, positive, neg)
        return loss

    return loss_func
