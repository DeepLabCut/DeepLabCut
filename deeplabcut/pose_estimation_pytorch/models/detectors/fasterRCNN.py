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

import torchvision.models.detection as detection

from deeplabcut.pose_estimation_pytorch.models.detectors.base import DETECTORS
from deeplabcut.pose_estimation_pytorch.models.detectors.torchvision import (
    TorchvisionDetectorAdaptor,
)


@DETECTORS.register_module
class FasterRCNN(TorchvisionDetectorAdaptor):
    """A FasterRCNN detector

    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
        Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. "Faster r-cnn: Towards
        real-time object detection with region proposal networks." Advances in neural
        information processing systems 28 (2015).

    This class is a wrapper of the torchvision implementation of a FasterRCNN (source:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py).

    Some of the available FasterRCNN variants (from fastest to most powerful):
      - fasterrcnn_mobilenet_v3_large_fpn
      - fasterrcnn_resnet50_fpn
      - fasterrcnn_resnet50_fpn_v2

    Args:
        variant: The FasterRCNN variant to use (see all options at
            https://pytorch.org/vision/stable/models.html#object-detection).
        pretrained: Whether to load model weights pretrained on COCO
        box_score_thresh: during inference, only return proposals with a classification
            score greater than box_score_thresh
    """

    def __init__(
        self,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        variant: str = "fasterrcnn_mobilenet_v3_large_fpn",
        pretrained: bool = False,
        box_score_thresh: float = 0.01,
    ) -> None:
        if not variant.lower().startswith("fasterrcnn"):
            raise ValueError(
                "The version must start with `fasterrcnn`. See available models at "
                "https://pytorch.org/vision/stable/models.html#object-detection"
            )

        super().__init__(
            model=variant,
            weights=None,  # Always pass None to ensure num_classes=2 is used
            num_classes=2,  # Always use 2 classes for superanimal models
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            box_score_thresh=box_score_thresh,
        )

        if not pretrained:
            num_classes = 2
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
            
