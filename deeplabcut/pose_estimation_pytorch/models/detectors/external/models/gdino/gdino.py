from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from deeplabcut.pose_estimation_pytorch.models.detectors.external import (
    BaseExternalDetector,
    DetectionResult,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.gdino.config import (
    GroundingDINODetectorConfig,
)

logger = logging.getLogger(__name__)


class GroundingDINODetectorModel(BaseExternalDetector):
    """
    Hugging Face Transformers GroundingDINO adapter for DLC external detectors.

    This version uses:
        transformers.AutoProcessor
        transformers.AutoModelForZeroShotObjectDetection

    It does not import the original groundingdino package and does not require
    compiling groundingdino._C custom ops.
    """

    backend_name = "grounding_dino_hf"

    def __init__(self, config: GroundingDINODetectorConfig) -> None:
        super().__init__()

        self.config = config
        self.device = config.resolved_device()
        self.use_fp16 = config.resolved_use_fp16()

        self.processor = None
        self.model = None
        self._loaded = False

        if not self.config.lazy_load:
            self.load_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._loaded and self.model is not None and self.processor is not None:
            return

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers is required for the HF GroundingDINO detector. "
                "Install it with `pip install transformers`."
            ) from exc

        logger.info("Loading HF GroundingDINO detector")
        logger.info("Model ID: %s", self.config.model_id)
        logger.info("Device: %s", self.device)
        logger.info("Classes/prompts: %s", self.config.text_labels())

        processor_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        model_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.config.cache_dir is not None:
            processor_kwargs["cache_dir"] = self.config.cache_dir
            model_kwargs["cache_dir"] = self.config.cache_dir

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            **processor_kwargs,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

        self.model.to(self.device)
        self.model.eval()

        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded or self.model is None or self.processor is None:
            self.load_model()

    def unload_model(self) -> None:
        self.processor = None
        self.model = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_arrays() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    @staticmethod
    def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
        """
        Convert a CHW or HWC torch image tensor to PIL RGB.

        Supports:
          - uint8 [0, 255]
          - float [0, 1]
          - float [0, 255]
        """
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

    def _predict_one_xyxy(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Run HF GroundingDINO for one PIL image.

        Returns:
            boxes_xyxy:
                np.ndarray[N, 4], absolute pixel XYXY boxes.
            scores:
                np.ndarray[N].
        """
        self.ensure_loaded()
        assert self.model is not None
        assert self.processor is not None

        # Use a single period-separated prompt string for compatibility with
        # transformers versions whose tokenizer does not accept nested label lists.
        text_prompt = self.config.formatted_prompt()

        inputs = self.processor(
            images=image,
            text=text_prompt,
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

        # HF expects target_sizes as [height, width].
        target_sizes = [image.size[::-1]]

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            # Some transformers versions use box_threshold instead of threshold.
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=target_sizes,
            )

        result = results[0]
        boxes = result.get("boxes", torch.empty((0, 4)))
        scores = result.get("scores", torch.empty((0,)))

        if len(boxes) == 0:
            return self._empty_arrays()

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        else:
            boxes = np.asarray(boxes)

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.asarray(scores)

        return (
            boxes.astype(np.float32, copy=False).reshape(-1, 4),
            scores.astype(np.float32, copy=False).reshape(-1),
        )

    def _postprocess_xyxy(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Clean raw XYXY detector outputs while keeping XYXY format.

        BaseExternalDetector.inference(...) converts DetectionResult XYXY boxes into
        DLC detector context XYWH boxes.
        """
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if len(boxes_xyxy) == 0:
            return self._empty_arrays()

        if len(boxes_xyxy) != len(scores):
            raise ValueError(f"Expected one score per box, got boxes={boxes_xyxy.shape}, scores={scores.shape}.")

        if self.config.clip_boxes:
            width, height = image_size
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, width)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, height)

        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        areas = widths * heights

        valid = np.ones(len(boxes_xyxy), dtype=bool)

        if self.config.filter_invalid_boxes:
            valid &= np.isfinite(boxes_xyxy).all(axis=1)
            valid &= np.isfinite(scores)
            valid &= widths > 0
            valid &= heights > 0

        if self.config.min_box_area > 0:
            valid &= areas >= self.config.min_box_area

        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        areas = areas[valid]

        if len(boxes_xyxy) == 0:
            return self._empty_arrays()

        if self.config.largest_only:
            keep = np.array([int(np.argmax(areas))], dtype=np.int64)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]

        if self.config.max_detections is not None and len(scores) > self.config.max_detections:
            order = np.argsort(-scores)[: self.config.max_detections]
            boxes_xyxy = boxes_xyxy[order]
            scores = scores[order]

        return (
            boxes_xyxy.astype(np.float32, copy=False),
            scores.astype(np.float32, copy=False),
        )

    # ------------------------------------------------------------------
    # BaseExternalDetector API
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[torch.Tensor],
    ) -> list[DetectionResult]:
        """
        BaseExternalDetector API.

        Args:
            images:
                List of image tensors, typically CHW.

        Returns:
            One detection dict per image:
                {
                    "boxes": FloatTensor[N, 4],   # XYXY absolute pixel coords
                    "scores": FloatTensor[N],
                    "labels": LongTensor[N],
                }
        """
        self.ensure_loaded()

        detections: list[DetectionResult] = []

        with torch.inference_mode():
            for tensor in images:
                image = self._tensor_to_pil_rgb(tensor)

                boxes_xyxy, scores = self._predict_one_xyxy(image)
                boxes_xyxy, scores = self._postprocess_xyxy(
                    boxes_xyxy=boxes_xyxy,
                    scores=scores,
                    image_size=image.size,
                )

                boxes_t = torch.as_tensor(
                    boxes_xyxy,
                    dtype=torch.float32,
                    device=self.device,
                )
                scores_t = torch.as_tensor(
                    scores,
                    dtype=torch.float32,
                    device=self.device,
                )

                # HF post-processing returns text labels, but DLC top-down only
                # needs boxes/scores here. Use placeholder class IDs.
                labels_t = torch.zeros(
                    (len(scores_t),),
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
            "classes": list(self.config.classes),
            "text_labels": self.config.text_labels(),
            "formatted_prompt": self.config.formatted_prompt(),
            "device": self.device,
            "use_fp16": self.use_fp16,
            "box_threshold": self.config.box_threshold,
            "text_threshold": self.config.text_threshold,
            "max_detections": self.config.max_detections,
            "largest_only": self.config.largest_only,
            "min_box_area": self.config.min_box_area,
            "clip_boxes": self.config.clip_boxes,
            "filter_invalid_boxes": self.config.filter_invalid_boxes,
            "image_color_order": self.config.image_color_order,
            "box_format": "xyxy",
            "coordinate_system": "absolute_pixels",
            "config": self.config.model_dump(mode="json"),
        }
