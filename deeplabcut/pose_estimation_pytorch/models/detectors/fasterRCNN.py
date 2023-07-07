import torch
from typing import List
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from .base import DETECTORS, BaseDetector


@DETECTORS.register_module
class FasterRCNN(BaseDetector):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        )
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def get_target(self, annotations):
        """
        Returns target in a format FasterRCNN can handle
        Args:
            annotations : dict of annotations, must contain the keys 'area', 'labels',
                         'is_crowd', 'image_id', 'boxes'

        Output:
            list of the target dictionaries (not the same serialisation for batches as default pytorch does)
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
