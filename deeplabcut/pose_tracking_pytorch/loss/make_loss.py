import torch


def easy_triplet_loss():
    def loss_func(anchor, positive, neg):
        triplet_loss = torch.nn.TripletMarginLoss()
        loss = triplet_loss(anchor, positive, neg)
        return loss

    return loss_func
