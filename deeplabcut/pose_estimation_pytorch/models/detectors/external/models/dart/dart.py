# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/dart/dart.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from deeplabcut.pose_estimation_pytorch.models.detectors.external import (
    EXTERNAL_DETECTORS,
    BaseExternalDetector,
    DetectionResult,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.dart.config import (
    SAM3DARTDetectorConfig,
)

logger = logging.getLogger(__name__)
try:
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
    from sam3.model_builder import (
        build_pruned_sam3_image_model,
        build_sam3_image_model,
        load_pruned_config,
    )
except ImportError as e:
    logger.warning("Failed to import SAM3/DART modules: %s", e)


# @EXTERNAL_DETECTORS.register_module()
class DARTDetectorModel(BaseExternalDetector):
    """
    PyTorch-only SAM3/DART detector adapter for DeepLabCut external detector workflows.

    The DART/SAM3 predictor is expected to return XYXY boxes internally.
    """

    backend_name = "sam3_dart"

    def __init__(self, config: SAM3DARTDetectorConfig) -> None:
        super().__init__()

        self.config = config
        self.device = config.resolved_device()
        self.use_fp16 = config.resolved_use_fp16()

        self.predictor: Sam3MultiClassPredictorFast | None = None
        self._loaded = False
        self._warmed_up = False
        self._resolved_checkpoint: str | None = None

        if not self.config.lazy_load:
            self.load_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self) -> str | None:
        """
        Resolve the configured SAM3 checkpoint path.

        Downloading is intentionally opt-in through ``allow_checkpoint_download`` so
        object construction and model loading remain reproducible and explicit.
        """
        checkpoint = self.config.checkpoint

        if checkpoint is None:
            return None

        path = Path(checkpoint).expanduser()
        if path.is_file():
            return str(path)

        if self.config.allow_checkpoint_download:
            from sam3.model_builder import download_ckpt_from_hf

            downloaded = download_ckpt_from_hf()
            logger.info("Downloaded SAM3 checkpoint to %s", downloaded)
            return downloaded

        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {path}. Pass a valid checkpoint path or set allow_checkpoint_download=True."
        )

    def load_model(self) -> None:
        """
        Load the SAM3/DART model and initialize the fast multi-class predictor.
        """
        if self._loaded and self.predictor is not None:
            return

        checkpoint = self._resolve_checkpoint()
        self._resolved_checkpoint = checkpoint

        logger.info("Loading SAM3/DART detector")
        logger.info("Device: %s", self.device)
        logger.info("Use FP16: %s", self.use_fp16)
        logger.info("Checkpoint: %s", checkpoint)
        logger.info("Classes/prompts: %s", self.config.classes)
        logger.info("Image size: %s", self.config.imgsz)

        pruned_config = load_pruned_config(checkpoint) if checkpoint else None

        if pruned_config is not None:
            model = build_pruned_sam3_image_model(
                checkpoint_path=checkpoint,
                pruning_config=pruned_config,
                device=self.device,
                eval_mode=True,
                skip_blocks=self.config.skip_blocks,
            )

            # Matches DART demo behavior for distilled checkpoints.
            if model.transformer.decoder.presence_token is not None:
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=checkpoint,
                eval_mode=True,
                skip_blocks=self.config.skip_blocks,
                mask_blocks=self.config.mask_blocks,
            )
        model = model.to(self.device)

        # Precompute position encodings if using a non-default resolution.
        if self.config.imgsz != 1008:
            pos_enc = model.backbone.vision_backbone.position_encoding
            pos_enc.precompute_for_resolution(self.config.imgsz)

        self.predictor = Sam3MultiClassPredictorFast(
            model,
            device=self.device,
            resolution=self.config.imgsz,
            compile_mode=self.config.compile_mode,
            use_fp16=self.use_fp16,
            presence_threshold=self.config.presence_threshold,
            detection_only=self.config.detection_only,
            trt_engine_path=None,
            trt_enc_dec_engine_path=None,
        )

        # Precompute text embeddings once.
        self.predictor.set_classes(self.config.classes)
        self._loaded = True

    def ensure_loaded(self) -> None:
        """
        Lazily load the detector model if needed.
        """
        if not self._loaded or self.predictor is None:
            self.load_model()

    def unload_model(self) -> None:
        """
        Release predictor references and clear CUDA cache when available.
        Useful for notebooks and long-running workflows.
        """
        self.predictor = None
        self._loaded = False
        self._warmed_up = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        """
        Optional one-time warmup to pay compile/CUDA startup cost upfront.
        """
        self.ensure_loaded()

        if self._warmed_up:
            return

        assert self.predictor is not None

        dummy = Image.new("RGB", (self.config.imgsz, self.config.imgsz))
        with torch.inference_mode():
            state = self.predictor.set_image(dummy)
            _ = self.predictor.predict(
                state,
                confidence_threshold=self.config.score_threshold,
                nms_threshold=self.config.nms_threshold,
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        self._warmed_up = True

    # ------------------------------------------------------------------
    # Inference helpers
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

    def _postprocess_xyxy(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize raw DART XYXY detector outputs while keeping XYXY format.

        This method intentionally only performs shape validation and optional clipping.
        Generic filtering/ranking logic is handled by BaseExternalDetector when converting
        DetectionResult -> DLC DetectorContext.
        """
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if len(boxes_xyxy) == 0:
            return self._empty_arrays()

        if len(boxes_xyxy) != len(scores):
            raise ValueError(f"Expected one score per box, got boxes={boxes_xyxy.shape}, scores={scores.shape}.")

        if self.config.clip_boxes and image_size is not None:
            width, height = image_size
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, width)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, height)

        return (
            boxes_xyxy.astype(np.float32, copy=False),
            scores.astype(np.float32, copy=False),
        )

    def _predict_pil_xyxy(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Run DART/SAM3 on one PIL image.

        Returns:
            boxes_xyxy:
                np.ndarray[N, 4], absolute pixel XYXY boxes.
            scores:
                np.ndarray[N].
        """
        self.ensure_loaded()
        assert self.predictor is not None

        state = self.predictor.set_image(image)
        results = self.predictor.predict(
            state,
            confidence_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
        )

        return self._extract_boxes_scores(results)

    @staticmethod
    def _extract_boxes_scores(results: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract XYXY boxes and scores from the SAM3/DART predictor output.
        """
        if "boxes" not in results or "scores" not in results:
            raise KeyError(f"DART predictor result must contain 'boxes' and 'scores'. Got keys={list(results.keys())}.")

        boxes = results["boxes"]
        scores = results["scores"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        else:
            boxes = np.asarray(boxes)

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.asarray(scores)

        boxes = boxes.astype(np.float32, copy=False).reshape(-1, 4)
        scores = scores.astype(np.float32, copy=False).reshape(-1)

        if len(boxes) != len(scores):
            raise ValueError(f"DART returned mismatched boxes/scores: boxes={boxes.shape}, scores={scores.shape}.")

        return boxes, scores

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

        if self.config.warmup_on_first_inference and not self._warmed_up:
            self.warmup()

        detections: list[DetectionResult] = []

        with torch.inference_mode():
            for tensor in images:
                image = self._tensor_to_pil_rgb(tensor)

                boxes_xyxy, scores = self._predict_pil_xyxy(image)
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
        """
        Return reproducibility metadata for bbox artifacts and DLC config.
        """
        return {
            "name": self.__class__.__name__,
            "backend": self.backend_name,
            "classes": list(self.config.classes),
            "checkpoint": str(self.config.checkpoint) if self.config.checkpoint is not None else None,
            "resolved_checkpoint": self._resolved_checkpoint,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "imgsz": self.config.imgsz,
            "score_threshold": self.config.score_threshold,
            "nms_threshold": self.config.nms_threshold,
            "presence_threshold": self.config.presence_threshold,
            "max_detections": self.config.max_detections,
            "bbox_selection_strategy": self.config.bbox_selection_strategy,
            "min_box_area": self.config.min_box_area,
            "clip_boxes": self.config.clip_boxes,
            "filter_invalid_boxes": self.config.filter_invalid_boxes,
            "image_color_order": self.config.image_color_order,
            "compile_mode": self.config.compile_mode,
            "skip_blocks": sorted(self.config.skip_blocks) if self.config.skip_blocks else None,
            "mask_blocks": self.config.mask_blocks,
            "detection_only": self.config.detection_only,
            "box_format": "xywh",
            "coordinate_system": "absolute_pixels",
            "config": self.config.model_dump(mode="json"),
        }


EXTERNAL_DETECTORS.register_module(module=DARTDetectorModel)
