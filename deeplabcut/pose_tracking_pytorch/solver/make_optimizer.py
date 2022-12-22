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


def make_easy_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg["base_lr"]
        weight_decay = cfg["weight_decay"]
        if "bias" in key:
            lr = cfg["base_lr"] * cfg["bias_lr_factor"]
            weight_decay = cfg["weight_decay_bias"]
        if cfg["large_fc_lr"]:
            if "classifier" in key or "arcface" in key:
                lr = cfg["base_lr"] * 2
                print("Using two times learning rate for fc ")

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer_name = cfg["optimizer_name"]
    if optimizer_name == "SGD":
        optimizer = getattr(torch.optim, optimizer_name)(
            params, momentum=cfg["momentum"]
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params, lr=cfg["base_lr"], weight_decay=cfg["weight_decay"]
        )
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)

    return optimizer
