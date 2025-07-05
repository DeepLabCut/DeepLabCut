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
import numpy as np
from PIL import Image
import torchvision

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
        model_name: Optional name of the model
        superanimal_name: Optional name of the superanimal model
    """

    def __init__(
        self,
        model: str = "fasterrcnn_resnet50_fpn_v2",
        weights: str | None = None,
        num_classes: int | None = 2,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        box_score_thresh: float = 0.01,
        model_kwargs: dict | None = None,
        model_name: str | None = None,
        superanimal_name: str | None = None,
    ) -> None:
        super().__init__(
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            pretrained=weights is not None,
        )
        self.model_name = model_name
        self.superanimal_name = superanimal_name

        model_fn = getattr(detection, model)
        if model_kwargs is None:
            model_kwargs = {}

        # Get the proper weights class
        if weights == "COCO_V1" or weights is None:
            if model == "fasterrcnn_mobilenet_v3_large_fpn":
                weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            elif model == "fasterrcnn_resnet50_fpn_v2":
                weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            else:
                weights = None

        if weights is not None:
            self.model = model_fn(
                weights=weights,
                box_score_thresh=box_score_thresh,
                **model_kwargs,
            )
        else:
            self.model = model_fn(
                weights=weights,
                box_score_thresh=box_score_thresh,
                num_classes=num_classes,
                **model_kwargs,
            )

        self.transforms = weights.transforms() if weights is not None else None

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
        print(f"DEBUG: TorchvisionDetectorAdaptor.forward called")
        print(f"DEBUG: Model training mode: {self.model.training}")
        print(f"DEBUG: Targets provided: {targets is not None}")
        if targets is not None:
            print(f"DEBUG: Number of targets: {len(targets)}")
            print(f"DEBUG: First target keys: {targets[0].keys() if targets else 'No targets'}")
            print(f"DEBUG: First target boxes shape: {targets[0]['boxes'].shape if targets and 'boxes' in targets[0] else 'No boxes'}")
        
        result = self.model(x, targets)
        
        print(f"DEBUG: Model result type: {type(result)}")
        if isinstance(result, tuple):
            print(f"DEBUG: Model result length: {len(result)}")
            print(f"DEBUG: First element type: {type(result[0])}")
            if len(result) > 0 and isinstance(result[0], dict):
                print(f"DEBUG: First element keys: {result[0].keys()}")
                if 'loss_classifier' in result[0]:
                    print(f"DEBUG: Loss classifier type: {type(result[0]['loss_classifier'])}")
                    print(f"DEBUG: Loss classifier requires_grad: {result[0]['loss_classifier'].requires_grad if hasattr(result[0]['loss_classifier'], 'requires_grad') else 'N/A'}")
        else:
            print(f"DEBUG: Model result is not a tuple, it's: {result}")
        
        # Handle different return formats from torchvision models
        if isinstance(result, tuple):
            if len(result) == 2:
                # Standard format: (losses, predictions)
                print(f"DEBUG: Returning standard format (losses, predictions)")
                return result
            elif len(result) > 2:
                # Some models return additional values, take first two
                print(f"DEBUG: Model returned {len(result)} values, taking first two")
                return result[0], result[1]
            else:
                # Single value, assume it's predictions
                print(f"DEBUG: Model returned single value, assuming predictions")
                # Return zero loss tensor for training compatibility
                device = x.device
                return {"total_loss": torch.tensor(0.0, device=device)}, result[0]
        else:
            # Single value, assume it's predictions
            print(f"DEBUG: Model returned non-tuple, assuming predictions")
            # Return zero loss tensor for training compatibility
            device = x.device
            return {"total_loss": torch.tensor(0.0, device=device)}, result

    def inference(self, images) -> list[dict[str, np.ndarray]]:
        """
        Run inference on images using the torchvision detector
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            List of detection results, each containing "bboxes" in xywh format
        """


        self.model.eval()
        device = next(self.model.parameters()).device
        
        print(f"DEBUG: Device: {device}")
        # Check if model has roi_heads (FasterRCNN) or not (SSD)
        if hasattr(self.model, 'roi_heads'):
            print(f"DEBUG: Model threshold: {self.model.roi_heads.score_thresh}")
        else:
            print(f"DEBUG: SSD model detected")
        
        results = []
        
        with torch.no_grad():
            for i, image in enumerate(images):
                print(f"DEBUG: Processing image {i}")
                
                # Convert to PIL Image if needed
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image).convert("RGB")
                elif not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                
                print(f"DEBUG: Image size: {image.size}")
                
                # Apply proper preprocessing
                if self.transforms is not None:
                    batch = [self.transforms(image).to(device)]
                    print(f"DEBUG: Using transforms preprocessing")
                else:
                    # For SSD models, we need specific preprocessing
                    if hasattr(self.model, 'roi_heads'):
                        # FasterRCNN preprocessing
                        import torchvision.transforms as transforms
                        preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                    else:
                        # SSD preprocessing - resize to 320x320
                        import torchvision.transforms as transforms
                        preprocess = transforms.Compose([
                            transforms.Resize((320, 320)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                    batch = [preprocess(image).to(device)]
                    print(f"DEBUG: Using {'SSD' if not hasattr(self.model, 'roi_heads') else 'FasterRCNN'} preprocessing")
                
                print(f"DEBUG: Batch tensor shape: {batch[0].shape}")
                
                # Run detection - call model directly without going through forward method
                print(f"DEBUG: Model type: {type(self.model)}")
                print(f"DEBUG: Model training mode: {self.model.training}")
                print(f"DEBUG: Model device: {next(self.model.parameters()).device}")
                print(f"DEBUG: Model box_score_thresh: {self.model.roi_heads.score_thresh}")
                print(f"DEBUG: Model nms_thresh: {self.model.roi_heads.nms_thresh}")
                
                # Call the underlying torchvision model directly for inference
                predictions = self.model(batch)
                print(f"DEBUG: Raw predictions type: {type(predictions)}")
                print(f"DEBUG: Raw predictions length: {len(predictions) if isinstance(predictions, (list, tuple)) else 'Not list/tuple'}")
                
                # Handle the output format - during inference, should be list of dicts
                if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
                    prediction = predictions[0]  # First image
                    print(f"DEBUG: Prediction type: {type(prediction)}")
                    print(f"DEBUG: Prediction keys: {prediction.keys() if isinstance(prediction, dict) else 'Not a dict'}")
                    print(f"DEBUG: Prediction content: {prediction}")
                else:
                    prediction = predictions
                    print(f"DEBUG: Single prediction type: {type(prediction)}")
                    print(f"DEBUG: Single prediction keys: {prediction.keys() if isinstance(prediction, dict) else 'Not a dict'}")
                    print(f"DEBUG: Single prediction content: {prediction}")
                
                # Check if predictions are empty due to threshold
                if isinstance(prediction, dict) and len(prediction) > 0:
                    if 'scores' in prediction:
                        print(f"DEBUG: Max score: {prediction['scores'].max() if len(prediction['scores']) > 0 else 'No scores'}")
                        print(f"DEBUG: Scores above threshold: {(prediction['scores'] > self.model.roi_heads.score_thresh).sum()}")
                
                if not isinstance(prediction, dict) or "boxes" not in prediction:
                    # Unexpected output, return empty
                    print(f"DEBUG: Unexpected output format")
                    results.append({
                        "bboxes": np.zeros((0, 4)),
                        "bbox_scores": np.zeros(0)
                    })
                    continue

                bboxes = prediction["boxes"].cpu().numpy()
                labels = prediction["labels"].cpu().numpy()
                scores = prediction["scores"].cpu().numpy()
                
                print(f"DEBUG: All detections - bboxes: {bboxes.shape}, labels: {labels}, scores: {scores}")

                # Handle empty detections
                if len(bboxes) == 0:
                    detected_bboxes = np.zeros((0, 4))
                    detected_scores = np.zeros(0)
                    print(f"DEBUG: No detections at all")
                else:
                    # For humanbody models, filter for humans (COCO class 1)
                    # For quadruped and other models, return all detections
                    if self.superanimal_name == 'superanimal_humanbody':
                        detection_mask = labels == 1
                        print(f"DEBUG: Filtering for humans (COCO class 1) - humanbody model")
                    else:
                        detection_mask = np.ones(len(bboxes), dtype=bool)
                        print(f"DEBUG: Returning all detections for non-humanbody model - {self.superanimal_name}")
                    detected_bboxes = bboxes[detection_mask]
                    detected_scores = scores[detection_mask]
                    detected_labels = labels[detection_mask]
                    print(f"DEBUG: Detections - bboxes: {detected_bboxes.shape}, labels: {detected_labels}, scores: {detected_scores}")

                # Convert to xywh format
                if len(detected_bboxes) > 0:
                    # Convert from (x1, y1, x2, y2) to (x, y, w, h)
                    detected_bboxes[:, 2] -= detected_bboxes[:, 0]  # width = x2 - x1
                    detected_bboxes[:, 3] -= detected_bboxes[:, 1]  # height = y2 - y1
                    
                    # Sort by confidence and keep top detections
                    sorted_indices = np.argsort(detected_scores)[::-1]
                    detected_bboxes = detected_bboxes[sorted_indices]
                    detected_scores = detected_scores[sorted_indices]
                    
                    # Limit to reasonable number of detections
                    max_detections = 10
                    if len(detected_bboxes) > max_detections:
                        detected_bboxes = detected_bboxes[:max_detections]
                        detected_scores = detected_scores[:max_detections]
                else:
                    detected_bboxes = np.zeros((0, 4))
                    detected_scores = np.zeros(0)

                results.append({
                    "bboxes": detected_bboxes,
                    "bbox_scores": detected_scores
                })
        
        return results

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

def torchvision_detector_inference(images, threshold=0.1, device="cpu"): 
    """
    Run the Colab-style torchvision detector on a list of images.
    Args:
        images: list of np.ndarray or PIL.Image
        threshold: float, detection threshold
        device: str, device to run on
    Returns:
        list of dicts with 'bboxes' and 'scores'
    """
    import torchvision.models.detection as detection
    from PIL import Image
    import numpy as np
    import torch

    # Use the exact working logic from colab_style_detector_test.py
    weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    detector = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, box_score_thresh=threshold
    )
    detector.eval()
    detector.to(device)
    preprocess = weights.transforms()

    results = []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        batch = [preprocess(image).to(device)]
        with torch.no_grad():
            predictions = detector(batch)[0]
        
        bboxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        
        # Filter for humans (COCO class 1)
        human_mask = labels == 1
        human_bboxes = bboxes[human_mask]
        human_scores = scores[human_mask]
        
        # Convert to xywh format
        if len(human_bboxes) > 0:
            human_bboxes[:, 2] -= human_bboxes[:, 0]
            human_bboxes[:, 3] -= human_bboxes[:, 1]
        
        results.append({
            "bboxes": human_bboxes,
            "scores": human_scores
        })
    return results
