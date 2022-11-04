from typing import Dict

import torch

from deeplabcut.pose_estimation_pytorch.models import PoseModel, BACKBONES, HEADS, LOSSES
from deeplabcut.pose_estimation_pytorch.solvers import LOGGER, SINGLE_ANIMAL_SOLVER
from deeplabcut.utils import auxiliaryfunctions


def build_pose_model(cfg: Dict,
                     pose_cfg: Dict):
    backbone = BACKBONES.build(dict(cfg['backbone']))
    head_heatmaps = HEADS.build(dict(cfg['heatmap_head']))
    head_locref = HEADS.build(dict(cfg['locref_head']))
    if cfg.get('neck'):
        neck = None
    else:
        neck = None
    pose_model = PoseModel(cfg=pose_cfg,
                           backbone=backbone,
                           head_heatmaps=head_heatmaps,
                           head_locref=head_locref,
                           neck=neck,
                           **cfg['pose_model'])

    return pose_model


def build_solver(cfg: Dict):
    pose_cfg = auxiliaryfunctions.read_config(cfg['pose_cfg_path'])
    pose_model = build_pose_model(cfg['model'], pose_cfg)

    get_optimizer = getattr(torch.optim, cfg['optimizer']['type'])
    optimizer = get_optimizer(params=pose_model.parameters(), **cfg['optimizer']['params'])

    criterion = LOSSES.build(cfg['criterion'])

    if cfg.get('scheduler'):
        _scheduler = getattr(torch.optim.lr_scheduler,
                             cfg['scheduler']['type'])
        scheduler = _scheduler(optimizer=optimizer,
                               **cfg['scheduler']['params'])
    else:
        scheduler = None

    if cfg.get('logger'):
        logger = LOGGER.build(dict(**cfg['logger'],
                                   model=pose_model))
    else:
        logger = None

    solver = SINGLE_ANIMAL_SOLVER.build(dict(**cfg['solver'],
                                             model=pose_model,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             cfg=pose_cfg,
                                             device=cfg['device'],
                                             scheduler=scheduler,
                                             logger=logger))
    return solver
