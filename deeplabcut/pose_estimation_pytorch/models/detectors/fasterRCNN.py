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

import torch
import torchvision.models.detection as detection

from deeplabcut.pose_estimation_pytorch.models.detectors.base import (
    DETECTORS,
    BaseDetector,
)


@DETECTORS.register_module
class FasterRCNN(BaseDetector):
    """A FasterRCNN detector

    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
        Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. "Faster r-cnn: Towards
        real-time object detection with region proposal networks." Advances in neural
        information processing systems 28 (2015).

    This class is a wrapper of the torchvision implementation of a FasterRCNN (source:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py).
    Any variant implemented in torchvision can be used through this wrapper (see
    available models at https://pytorch.org/vision/stable/models.html#object-detection).

    Some of the variants (from fastest to most powerful) available:
      - fasterrcnn_mobilenet_v3_large_fpn
      - fasterrcnn_resnet50_fpn
      - fasterrcnn_resnet50_fpn_v2

    The torchvision implementation does not allow to get both predictions and losses
    with a single forward pass. Therefore, during evaluation only bounding box metrics
    (mAP, mAR) are available for the test set. See validation loss issue:
    - https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333/12
    - https://stackoverflow.com/a/65347721

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
        pretrained: bool = True,
        box_score_thresh: float = 0.01,
    ) -> None:
        if not variant.lower().startswith("fasterrcnn"):
            raise ValueError(
                "The version must start with `fasterrcnn`. See available models at "
                "https://pytorch.org/vision/stable/models.html#object-detection"
            )

        super().__init__(
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            pretrained=pretrained,
        )
        model_fn = getattr(detection, variant)
        weights = None
        if self._pretrained:
            weights = "COCO_V1"

        # Load the model
        self.model = model_fn(weights=weights, box_score_thresh=box_score_thresh)

        # Modify the base predictor to output the correct number of classes
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

        # See source:  https://stackoverflow.com/a/65347721
        self.model.eager_outputs = lambda losses, detections: (losses, detections)

    def forward(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """
        Forward pass of the Faster R-CNN

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
        Returns target in a format FasterRCNN can handle

        Args:
            labels: dict of annotations, must contain the keys:
                area: tensor containing area information for each annotation
                labels: tensor containing class labels for each annotation
                is_crowd: tensor indicating if each annotation is a crowd (1) or not (0)
                image_id: tensor containing image ids for each annotation
                boxes: tensor containing bounding box information for each annotation

        Returns:
            res: list of dictionaries, each representing target information for a single annotation.
                 Each dictionary contains the following keys:
                 'area'
                 'labels'
                 'is_crowd'
                 'boxes'

        Examples:
            input:
                annotations = {"area": torch.Tensor([100, 200]),
                    "labels": torch.Tensor([1, 2]),
                    "is_crowd": torch.Tensor([0, 1]),
                    "boxes": torch.Tensor([[10, 20, 30, 40], [50, 60, 70, 80]])}
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
                    "labels": labels["labels"][i][mask],
                    "is_crowd": labels["is_crowd"][i][mask],
                    "boxes": box_ann,
                }
            )

        return res
