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
class SSDLite(TorchvisionDetectorAdaptor):
    """An SSD object detection model"""

    def __init__(
        self,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        pretrained: bool = False,
        pretrained_from_imagenet: bool = False,
        box_score_thresh: float = 0.01,
    ) -> None:
        model_kwargs = dict(weights_backbone=None)
        if pretrained_from_imagenet:
            model_kwargs["weights_backbone"] = "IMAGENET1K_V2"

        super().__init__(
            model="ssdlite320_mobilenet_v3_large",
            weights=None,
            num_classes=2,
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            box_score_thresh=box_score_thresh,
            model_kwargs=model_kwargs,
        )

        if pretrained and not pretrained_from_imagenet:
            weights = detection.SSDLite320_MobileNet_V3_Large_Weights.verify("COCO_V1")
            state_dict = weights.get_state_dict(progress=False, check_hash=True)
            for k, v in state_dict.items():
                key_parts = k.split(".")
                if (
                    len(key_parts) == 6
                    and key_parts[0] == "head"
                    and key_parts[1] == "classification_head"
                    and key_parts[2] == "module_list"
                    and key_parts[4] == "1"
                    and key_parts[5] in ("weight", "bias")
                ):
                    # number of COCO classes: 90 + background (91)
                    # number of DLC classes: 1 + background (2)
                    # -> only keep weights for the background + first class

                    # future improvement: find best-suited class for the project
                    #   and use those weights, instead of naively taking the first
                    all_classes_size = v.shape[0]
                    two_classes_size = 2 * (all_classes_size // 91)
                    state_dict[k] = v[:two_classes_size]

            self.model.load_state_dict(state_dict)
