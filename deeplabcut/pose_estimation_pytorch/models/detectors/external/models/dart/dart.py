# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/dart/dart.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from sam3.model_builder import (
    build_pruned_sam3_image_model,
    build_sam3_image_model,
    load_pruned_config,
)
from tqdm import tqdm

from deeplabcut.pose_estimation_pytorch.data.bboxes import _xyxy_to_xywh
from deeplabcut.pose_estimation_pytorch.models.detectors.external import BaseExternalDetector

from .config import SAM3DARTDetectorConfig

logger = logging.getLogger(__name__)


def _coerce_to_pil_rgb(item: Any) -> Image.Image:
    """
    Accept common DLC/external-detector inputs:
      - str / Path
      - PIL.Image
      - np.ndarray
      - (image, context) tuples
    and return a PIL RGB image.
    """
    # Handle DLC-style tuples such as (image, context)
    if isinstance(item, tuple) and len(item) > 0:
        item = item[0]

    if isinstance(item, Image.Image):
        return item if item.mode == "RGB" else item.convert("RGB")

    if isinstance(item, (str, Path)):
        return Image.open(item).convert("RGB")

    if isinstance(item, np.ndarray):
        arr = item
        if arr.ndim == 2:
            # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 ndarray for image input, got shape={arr.shape}")

        # If coming from OpenCV, the caller should convert BGR->RGB before passing.
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    raise TypeError(
        f"Unsupported image input type: {type(item)!r}. Supported: Path/str, PIL.Image, np.ndarray, (image, context)."
    )


class DARTDetectorModel(BaseExternalDetector):
    """
    PyTorch-only SAM3/DART detector adapter for DeepLabCut external-detector workflows.

    Contract:
        inference(images, shelf_writer=None) -> list[dict]
    Each dict contains:
        {
            "bboxes": np.ndarray[N, 4],      # XYWH in pixels
            "bbox_scores": np.ndarray[N],    # detection confidences
        }
    """

    def __init__(
        self,
        config: SAM3DARTDetectorConfig,
    ) -> None:
        super().__init__()

        self.config = config

        if config.imgsz % 14 != 0:
            raise ValueError(f"imgsz must be divisible by 14, got {config.imgsz}")
        if not Path(config.checkpoint).is_file():
            logger.warning(f"Checkpoint file not found: {config.checkpoint}. The model will fail to load.")
            from sam3.model_builder import download_ckpt_from_hf

            config.checkpoint = download_ckpt_from_hf()

        self.classes = list(config.classes)
        self.checkpoint = config.checkpoint
        self.device = config.device
        self.imgsz = config.imgsz
        self.confidence = config.confidence
        self.nms = config.nms
        self.compile_mode = config.compile_mode
        self.max_detections = config.max_detections
        self.largest_only = config.largest_only
        self.skip_blocks = config.skip_blocks
        self.mask_blocks = config.mask_blocks

        self.predictor = None
        self._loaded = False
        # Optional warmup
        self._warmed_up = False

    def load_model(self) -> None:
        pruned_config = load_pruned_config(self.checkpoint) if self.checkpoint else None
        if pruned_config is not None:
            model = build_pruned_sam3_image_model(
                checkpoint_path=self.checkpoint,
                pruning_config=pruned_config,
                device=self.device,
                eval_mode=True,
                skip_blocks=self.skip_blocks,
            )
            # Matches DART demo behavior for distilled checkpoints
            if model.transformer.decoder.presence_token is not None:
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=self.checkpoint,
                eval_mode=True,
                skip_blocks=self.skip_blocks,
                mask_blocks=self.mask_blocks,
            )

        # Precompute position encodings if using a non-default resolution
        if self.imgsz != 1008:
            pos_enc = model.backbone.vision_backbone.position_encoding
            pos_enc.precompute_for_resolution(self.imgsz)

        # Fast predictor, PyTorch only: no TRT args passed here
        self.predictor = Sam3MultiClassPredictorFast(
            model,
            device=self.device,
            resolution=self.imgsz,
            compile_mode=self.compile_mode,
            use_fp16=(self.device.startswith("cuda")),
            presence_threshold=0.05,
            detection_only=True,
            trt_engine_path=None,
            trt_enc_dec_engine_path=None,
        )

        # Precompute text embeddings once
        self.predictor.set_classes(self.classes)
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded or self.predictor is None:
            self.load_model()

    def warmup(self) -> None:
        """
        Optional one-time warmup to pay compile/CUDA startup cost upfront.
        """
        if self._warmed_up:
            return

        dummy = Image.new("RGB", (self.imgsz, self.imgsz))
        with torch.inference_mode():
            state = self.predictor.set_image(dummy)
            _ = self.predictor.predict(
                state,
                confidence_threshold=self.confidence,
                nms_threshold=self.nms,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._warmed_up = True

    def _postprocess(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_w, max_h = image_size if image_size is not None else (None, None)
        if boxes_xyxy.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        # Keep only largest detection if requested
        if self.largest_only:
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            keep = np.array([int(np.argmax(areas))], dtype=np.int64)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]

        # Keep top-k by confidence if requested
        if self.max_detections is not None and len(scores) > self.max_detections:
            order = np.argsort(-scores)[: self.max_detections]
            boxes_xyxy = boxes_xyxy[order]
            scores = scores[order]

        valid = (
            np.isfinite(boxes_xyxy).all(axis=1)
            & np.isfinite(scores)
            & ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) > 0)
            & ((boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) > 0)
        )
        if not valid.all():
            logger.warning(
                f"Filtering out {len(valid) - valid.sum()} "
                f"invalid detections with boxes={boxes_xyxy[~valid]} and scores={scores[~valid]}"
            )
        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)

        # Clamp widths/heights to non-negative
        boxes_xywh[:, 2:] = np.maximum(boxes_xywh[:, 2:], 0.0)
        # Clamp to image size if given (handles partial detections at borders)
        if max_w is not None and max_h is not None:
            boxes_xywh[:, 0] = np.clip(boxes_xywh[:, 0], 0, max_w)
            boxes_xywh[:, 1] = np.clip(boxes_xywh[:, 1], 0, max_h)
            boxes_xywh[:, 2] = np.clip(boxes_xywh[:, 2], 0, max_w - boxes_xywh[:, 0])
            boxes_xywh[:, 3] = np.clip(boxes_xywh[:, 3], 0, max_h - boxes_xywh[:, 1])

        return boxes_xywh.astype(np.float32), scores.astype(np.float32)

    def inference(self, images, shelf_writer=None, profile: bool = False):
        self.ensure_loaded()

        outputs = []

        if self.config.warmup_on_first_inference and not self._warmed_up:
            self.warmup()

        with torch.inference_mode():
            for item in tqdm(images):
                image = _coerce_to_pil_rgb(item)

                state = self.predictor.set_image(image)
                results = self.predictor.predict(
                    state,
                    confidence_threshold=self.confidence,
                    nms_threshold=self.nms,
                )

                boxes_xyxy = results["boxes"].detach().cpu().numpy()
                scores = results["scores"].detach().cpu().numpy()

                boxes_xywh, scores = self._postprocess(boxes_xyxy, scores)

                outputs.append(
                    {
                        "bboxes": boxes_xywh,
                        "bbox_scores": scores,
                    }
                )

        return outputs
