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
"""Module to adapt torchvision detectors for DeepLabCut"""
from __future__ import annotations

import torch
import torchvision.models.detection as detection

from deeplabcut.pose_estimation_pytorch.models.detectors.base import (
    BaseDetector,
)


class TorchvisionDetectorAdaptor(BaseDetector):
    """An adaptor for torchvision detectors

    This class is an adaptor for torchvision detectors to DeepLabCut detectors. Some of
    the models (from fastest to most powerful) available are:
      - ssdlite320_mobilenet_v3_large
      - fasterrcnn_mobilenet_v3_large_fpn
      - fasterrcnn_resnet50_fpn_v2

    This class should not be used out-of-the-box. Subclasses (such as FasterRCNN or
    SSDLite) should be used instead.

    The torchvision implementation does not allow to get both predictions and losses
    with a single forward pass. Therefore, during evaluation only bounding box metrics
    (mAP, mAR) are available for the test set. See validation loss issue:
    - https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333/12
    - https://stackoverflow.com/a/65347721

    Args:
        model: The torchvision model to use (see all options at
            https://pytorch.org/vision/stable/models.html#object-detection).
        weights: The weights to load for the model. If None, no pre-trained weights are
            loaded.
        num_classes: Number of classes that the model should output. If None, the number
            of classes the model is pre-trained on is used.
        freeze_bn_stats: Whether to freeze stats for BatchNorm layers.
        freeze_bn_weights: Whether to freeze weights for BatchNorm layers.
        box_score_thresh: during inference, only return proposals with a classification
            score greater than box_score_thresh
    """

    def __init__(
        self,
        model: str,
        weights: str | None = None,
        num_classes: int | None = 2,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        box_score_thresh: float = 0.01,
        model_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            pretrained=weights is not None,
        )

        # Load the model
        model_fn = getattr(detection, model)
        if model_kwargs is None:
            model_kwargs = {}

        self.model = model_fn(
            weights=weights,
            box_score_thresh=box_score_thresh,
            num_classes=num_classes,
            **model_kwargs,
        )

        # See source:  https://stackoverflow.com/a/65347721
        self.model.eager_outputs = lambda losses, detections: (losses, detections)

    def forward(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """
        Forward pass of the torchvision detector

        Args:
            x: images to be processed, of shape (b, c, h, w)
            targets: ground-truth boxes present in the images

        Returns:
            losses: {'loss_name': loss_value}
            detections: for each of the b images, {"boxes": bounding_boxes}
        """
        return self.model(x, targets)

    def get_target(self, labels: dict) -> list[dict[str, torch.Tensor]]:
        """
        Returns target in a format a torchvision detector can handle

        Args:
            labels: dict of annotations, must contain the keys:
                area: tensor containing area information for each annotation
                labels: tensor containing class labels for each annotation
                is_crowd: tensor indicating if each annotation is a crowd (1) or not (0)
                image_id: tensor containing image ids for each annotation
                boxes: tensor containing bounding box information for each annotation

        Returns:
            res: list of dictionaries, each representing target information for a single
                annotation. Each dictionary contains the following keys:
                    'area'
                    'labels'
                    'is_crowd'
                    'boxes'

        Examples:
            input:
                annotations = {
                    "area": torch.Tensor([100, 200]),
                    "labels": torch.Tensor([1, 2]),
                    "is_crowd": torch.Tensor([0, 1]),
                    "boxes": torch.Tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
                }
            output:
                res =  [
                    {
                        'area': tensor([100.]),
                        'labels': tensor([1]),
                        'image_id': tensor([1]),
                        'is_crowd': tensor([0]),
                        'boxes': tensor([[10., 20., 40., 60.]])
                    },
                    {
                        'area': tensor([200.]),
                        'labels': tensor([2]),
                        'image_id': tensor([1]),
                        'is_crowd': tensor([1]),
                        'boxes': tensor([[50., 60., 70., 80.]])
                    }
                ]
        """
        res = []
        for i, box_ann in enumerate(labels["boxes"]):
            mask = (box_ann[:, 2] > 0.0) & (box_ann[:, 3] > 0.0)
            box_ann = box_ann[mask]
            # bbox format conversion (x, y, w, h) -> (x1, y1, x2, y2)
            box_ann[:, 2] += box_ann[:, 0]
            box_ann[:, 3] += box_ann[:, 1]
            res.append(
                {
                    "area": labels["area"][i][mask],
                    "labels": labels["labels"][i][mask].long(),
                    "is_crowd": labels["is_crowd"][i][mask].long(),
                    "boxes": box_ann,
                }
            )

        return res
