# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/rf_detr/model.py
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from deeplabcut.pose_estimation_pytorch.models.detectors.external import (
    EXTERNAL_DETECTORS,
    BaseExternalDetector,
    DetectionResult,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.rf_detr.config import (
    RFDETRDetectorConfig,
)

logger = logging.getLogger(__name__)


class RFDETRDetectorModel(BaseExternalDetector):
    """
    Hugging Face RF-DETR adapter for DLC external detector workflows.

    This adapter returns raw XYXY absolute pixel boxes. Generic filtering/ranking,
    max-detection selection, and XYXY -> XYWH conversion are handled by
    BaseExternalDetector.
    """

    backend_name = "hf_rf_detr"

    def __init__(self, config: RFDETRDetectorConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = RFDETRDetectorConfig(**config)

        self.config = config
        self.device = config.resolved_device()
        self.use_fp16 = config.resolved_use_fp16()

        self.processor = None
        self.model = None
        self._loaded = False
        self._target_label_ids_cache: set[int] | None = None

        if not self.config.lazy_load:
            self.load_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._loaded and self.model is not None and self.processor is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "HF RF-DETR requires Transformers with AutoImageProcessor and "
                "AutoModelForObjectDetection support. Install or upgrade with:\n\n"
                "  pip install --upgrade transformers"
            ) from exc

        logger.info("Loading HF RF-DETR detector")
        logger.info("Model ID: %s", self.config.model_id)
        logger.info("Device: %s", self.device)
        logger.info("Target classes: %s", self.config.target_classes)
        logger.info("Target label IDs: %s", self.config.target_label_ids)

        hf_kwargs = self.config.hf_kwargs()

        self.processor = AutoImageProcessor.from_pretrained(
            self.config.model_id,
            **hf_kwargs,
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.config.model_id,
            **hf_kwargs,
        )

        self.model.to(self.device)
        self.model.eval()

        self._target_label_ids_cache = None
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded or self.model is None or self.processor is None:
            self.load_model()

    def unload_model(self) -> None:
        self.processor = None
        self.model = None
        self._loaded = False
        self._target_label_ids_cache = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    @staticmethod
    def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu()

        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dimensions, got shape={tuple(tensor.shape)}.")

        # CHW -> HWC
        if tensor.shape[0] in {1, 3, 4}:
            arr = tensor.permute(1, 2, 0).numpy()
        else:
            arr = tensor.numpy()

        if arr.ndim != 3:
            raise ValueError(f"Expected image array HxWxC, got shape={arr.shape}.")

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        if arr.shape[2] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got shape={arr.shape}.")

        if arr.dtype != np.uint8:
            if np.nanmax(arr) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    def _target_label_ids(self) -> set[int] | None:
        """
        Return the set of label IDs to keep, or None if all labels should be kept.
        """
        if self._target_label_ids_cache is not None:
            return self._target_label_ids_cache

        assert self.model is not None

        keep: set[int] = set()

        if self.config.target_label_ids is not None:
            keep.update(int(i) for i in self.config.target_label_ids)

        if self.config.target_classes is not None:
            id2label = getattr(self.model.config, "id2label", {})
            normalized = {str(label).strip().lower(): int(label_id) for label_id, label in id2label.items()}

            missing = []
            for class_name in self.config.target_classes:
                key = class_name.strip().lower()
                if key in normalized:
                    keep.add(normalized[key])
                else:
                    missing.append(class_name)

            if missing and not self.config.allow_missing_target_classes:
                known = sorted(normalized.keys())
                raise ValueError(
                    "Could not find target class(es) in RF-DETR id2label: "
                    f"{missing}. Known labels include: {known[:20]}"
                )

        self._target_label_ids_cache = keep if keep else None
        return self._target_label_ids_cache

    def _filter_labels(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep_ids = self._target_label_ids()
        if keep_ids is None:
            return boxes_xyxy, scores, labels

        keep = np.asarray([int(label) in keep_ids for label in labels], dtype=bool)
        return boxes_xyxy[keep], scores[keep], labels[keep]

    # ------------------------------------------------------------------
    # BaseExternalDetector API
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[torch.Tensor],
    ) -> list[DetectionResult]:
        """
        Run HF RF-DETR on a batch of images.

        Returns:
            One detection dict per image:
                {
                    "boxes": FloatTensor[N, 4],   # XYXY absolute pixel coords
                    "scores": FloatTensor[N],
                    "labels": LongTensor[N],
                }
        """
        self.ensure_loaded()
        assert self.model is not None
        assert self.processor is not None

        pil_images = [self._tensor_to_pil_rgb(tensor) for tensor in images]

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)

        device_type = torch.device(self.device).type
        use_autocast = self.use_fp16 and device_type == "cuda"

        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=use_autocast,
            ):
                outputs = self.model(**inputs)

        # HF object detection post-processing expects [batch_size, 2] as [height, width].
        target_sizes = torch.tensor(
            [(image.height, image.width) for image in pil_images],
            dtype=torch.long,
            device=self.device,
        )

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.score_threshold,
        )

        detections: list[DetectionResult] = []

        for result in results:
            boxes = result.get("boxes", torch.empty((0, 4), device=self.device))
            scores = result.get("scores", torch.empty((0,), device=self.device))
            labels = result.get(
                "labels",
                torch.empty((0,), dtype=torch.long, device=self.device),
            )

            if len(boxes) == 0:
                boxes_np, scores_np, labels_np = self._empty_arrays()
            else:
                boxes_np = boxes.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 4)
                scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)

                if len(boxes_np) != len(scores_np):
                    raise ValueError(
                        f"RF-DETR returned mismatched boxes/scores: boxes={boxes_np.shape}, scores={scores_np.shape}."
                    )

                if len(boxes_np) != len(labels_np):
                    raise ValueError(
                        f"RF-DETR returned mismatched boxes/labels: boxes={boxes_np.shape}, labels={labels_np.shape}."
                    )

                boxes_np, scores_np, labels_np = self._filter_labels(
                    boxes_np,
                    scores_np,
                    labels_np,
                )

            boxes_t = torch.as_tensor(
                boxes_np,
                dtype=torch.float32,
                device=self.device,
            )
            scores_t = torch.as_tensor(
                scores_np,
                dtype=torch.float32,
                device=self.device,
            )
            labels_t = torch.as_tensor(
                labels_np,
                dtype=torch.long,
                device=self.device,
            )

            detections.append(
                {
                    "boxes": boxes_t,
                    "scores": scores_t,
                    "labels": labels_t,
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "backend": self.backend_name,
            "model_id": self.config.model_id,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "score_threshold": self.config.score_threshold,
            "max_detections": self.config.max_detections,
            "bbox_selection_strategy": self.config.bbox_selection_strategy,
            "min_box_area": self.config.min_box_area,
            "clip_boxes": self.config.clip_boxes,
            "filter_invalid_boxes": self.config.filter_invalid_boxes,
            "target_classes": self.config.target_classes,
            "target_label_ids": self.config.target_label_ids,
            "allow_missing_target_classes": self.config.allow_missing_target_classes,
            "image_color_order": self.config.image_color_order,
            "raw_box_format": "xyxy",
            "output_box_format": "xywh",
            "coordinate_system": "absolute_pixels",
            "config": self.config.model_dump(mode="json"),
        }


EXTERNAL_DETECTORS.register_module(module=RFDETRDetectorModel)
