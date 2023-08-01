# Copyright (c) OpenMMLab. All rights reserved.
from .classfication_loss import BCELoss
from .heatmap_loss import AdaptiveWingLoss

from .mse_loss import JointsMSELoss, JointsOHKMMSELoss
from .multi_loss_factory import  HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss)

__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 
    'MultiLossFactory',   'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss'
]
