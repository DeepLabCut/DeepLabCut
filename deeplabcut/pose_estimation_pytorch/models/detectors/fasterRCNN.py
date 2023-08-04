from typing import List

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base import DETECTORS, BaseDetector


@DETECTORS.register_module
class FasterRCNN(BaseDetector):
    """
    Definition of the class object FasterRCNN.
    Faster Region-based Convolutional Neural Network (R-CNN) is a popular object detection model
    that builds upn the R-CNN framework.

    I. A. Siradjuddin, Reynaldi and A. Muntasa, "Faster Region-based Convolutional Neural Network
    for Mask Face Detection," 2021 5th International Conference on Informatics and Computational
    Sciences (ICICoS), Semarang, Indonesia, 2021, pp. 282-286, doi: 10.1109/ICICoS53627.2021.9651744.
    """

    def __init__(
        self,
    ):
        """Summary:
        Constructor of the FasterRCNN object.
        Loads the data.

        Args:
            None

        Return:
            None
        """
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        )
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x: torch.Tensor, targets: dict = None) -> torch.Tensor:
        """Summary:
        Forward pass of the Faster R-CNN

        Args:
            x: input tensor to the detector
            targets: dictionary containing target information for training.
                                      Defaults to None.

        Returns:
            Output tensor from the detector. If targets are provided, returns
            a tuple of losses (classification and regression).
            If targets are not provided, returns a tensor with predicted bounding
            boxes and associated scores.
        """
        return self.model(x, targets)

    def get_target(self, annotations: dict) -> List[dict]:
        """Summary:
        Returns target in a format FasterRCNN can handle

        Args:
            annotations: dict of annotations, must contain the keys:
                         area: tensor containing area information for each annotation.
                         labels: tensor containing class labels for each annotation.
                         is_crowd: tensor indicating if each annotation is a crowd (1) or not (0).
                         image_id: tensor containing image ids for each annotation
                         boxes: tensor containing bounding box information for each annotation

        Returns:
            res: list of dictionaries, each representing target information for a single annotation.
                 Each dictionary contains the following keys:
                 'area'
                 'labels'
                 'image_id'
                 'is_crowd'
                 'boxes'

        Examples:
            input:
                annotations = {"area": torch.Tensor([100, 200]),
                    "labels": torch.Tensor([1, 2]),
                    "is_crowd": torch.Tensor([0, 1]),
                    "image_id": torch.Tensor([1, 1]),
                    "boxes": torch.Tensor([[10, 20, 30, 40], [50, 60, 70, 80]])}
            output:
                res =  [{'area': tensor([100.]), 'labels': tensor([1]), 'image_id': tensor([1]), 'is_crowd': tensor([0]),
                    'boxes': tensor([[10., 20., 40., 60.]])},
                    {'area': tensor([200.]), 'labels': tensor([2]), 'image_id': tensor([1]), 'is_crowd': tensor([1]), 'boxes':
                    tensor([[50., 60., 70., 80.]])}]
        """
        res = []
        for i, _ in enumerate(annotations["image_id"]):
            box_ann = annotations["boxes"][i].clone()

            mask = (box_ann[:, 2] > 0.0) & (box_ann[:, 3] > 0.0)
            box_ann = box_ann[mask]
            # bbox format conversion (x, y, w, h) -> (x1, y1, x2, y2)
            box_ann[:, 2] += box_ann[:, 0]
            box_ann[:, 3] += box_ann[:, 1]
            res.append(
                {
                    "area": annotations["area"][i],
                    "labels": annotations["labels"][i],
                    "image_id": annotations["image_id"][i],
                    "is_crowd": annotations["is_crowd"][i],
                    "boxes": box_ann,
                }
            )

        return res
