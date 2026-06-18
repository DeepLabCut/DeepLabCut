# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/owl_v2/model.py
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from deeplabcut.pose_estimation_pytorch.models.detectors.external.base import (
    EXTERNAL_DETECTORS,
    BaseExternalDetector,
    DetectionResult,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.owl_v2.config import (
    OWLv2DetectorConfig,
)

logger = logging.getLogger(__name__)


class OWLv2DetectorModel(BaseExternalDetector):
    """
    Hugging Face OWLv2 adapter for DLC external detector workflows.

    This adapter returns raw XYXY absolute pixel boxes. Generic filtering/ranking,
    max-detection selection, and conversion to DLC XYWH detector context are handled
    by BaseExternalDetector.
    """

    backend_name = "hf_owlv2"

    def __init__(self, config: OWLv2DetectorConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = OWLv2DetectorConfig(**config)

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
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise ImportError(
                "Transformers with OWLv2 support is required for OWLv2DetectorModel. "
                "Install or upgrade Transformers with:\n\n"
                "  pip install --upgrade transformers"
            ) from exc

        logger.info("Loading HF OWLv2 detector")
        logger.info("Model ID: %s", self.config.model_id)
        logger.info("Device: %s", self.device)
        logger.info("Classes/prompts: %s", self.config.text_labels())

        hf_kwargs = self.config.hf_kwargs()

        self.processor = Owlv2Processor.from_pretrained(
            self.config.model_id,
            **hf_kwargs,
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.config.model_id,
            **hf_kwargs,
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

    def _label_index_for_text(self, text_label: str, prompts: list[str]) -> int:
        try:
            return prompts.index(text_label)
        except ValueError:
            return -1

    # ------------------------------------------------------------------
    # BaseExternalDetector API
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[torch.Tensor],
    ) -> list[DetectionResult]:
        """
        Run OWLv2 on a batch of images.

        Returns:
            One detection dict per image:
                {
                    "boxes": FloatTensor[N, 4],   # XYXY absolute pixel coords
                    "scores": FloatTensor[N],
                    "labels": LongTensor[N],      # prompt index, or -1 if unknown
                }
        """
        self.ensure_loaded()
        assert self.model is not None
        assert self.processor is not None

        pil_images = [self._tensor_to_pil_rgb(tensor) for tensor in images]
        prompts = self.config.text_labels()

        # OWLv2 expects text as list[list[str]], one prompt list per image.
        text_inputs = [prompts for _ in pil_images]

        inputs = self.processor(
            text=text_inputs,
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

        target_sizes = torch.tensor(
            [(image.height, image.width) for image in pil_images],
            dtype=torch.long,
            device=self.device,
        )

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.config.score_threshold,
            text_labels=text_inputs,
        )

        detections: list[DetectionResult] = []

        for result in results:
            boxes = result.get("boxes", torch.empty((0, 4), device=self.device))
            scores = result.get("scores", torch.empty((0,), device=self.device))
            text_labels = result.get("text_labels", [])

            if len(boxes) == 0:
                boxes_np, scores_np, labels_np = self._empty_arrays()

            else:
                boxes_np = boxes.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 4)
                scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

                if len(boxes_np) != len(scores_np):
                    raise ValueError(
                        f"OWLv2 returned mismatched boxes/scores: boxes={boxes_np.shape}, scores={scores_np.shape}."
                    )

                labels_np = np.asarray(
                    [self._label_index_for_text(str(label), prompts) for label in text_labels],
                    dtype=np.int64,
                )

                if len(labels_np) != len(boxes_np):
                    # Some Transformers versions may not return text_labels as expected.
                    labels_np = np.full((len(boxes_np),), -1, dtype=np.int64)

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
            "classes": list(self.config.classes),
            "text_labels": self.config.text_labels(),
            "prompt_mode": self.config.prompt_mode,
            "prompt_template": self.config.prompt_template,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "score_threshold": self.config.score_threshold,
            "max_detections": self.config.max_detections,
            "bbox_selection_strategy": self.config.bbox_selection_strategy,
            "min_box_area": self.config.min_box_area,
            "clip_boxes": self.config.clip_boxes,
            "filter_invalid_boxes": self.config.filter_invalid_boxes,
            "image_color_order": self.config.image_color_order,
            "raw_box_format": "xyxy",
            "output_box_format": "xywh",
            "coordinate_system": "absolute_pixels",
            "config": self.config.model_dump(mode="json"),
        }


EXTERNAL_DETECTORS.register_module(module=OWLv2DetectorModel)
