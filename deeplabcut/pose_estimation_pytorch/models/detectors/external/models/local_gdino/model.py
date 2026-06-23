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
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.local_gdino.config import (
    LocalGroundingDINODetectorConfig,
)

logger = logging.getLogger(__name__)


class LocalGroundingDINODetectorModel(BaseExternalDetector):
    """
    Local/offline GroundingDINO adapter for DLC external detector workflows.

    This adapter uses the original GroundingDINO package with local config and
    checkpoint files. It returns raw XYXY absolute pixel boxes. Generic filtering,
    max-detection selection, and XYXY -> XYWH conversion are handled by
    BaseExternalDetector.
    """

    backend_name = "local_groundingdino"

    def __init__(self, config: LocalGroundingDINODetectorConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = LocalGroundingDINODetectorConfig(**config)

        self.config = config
        self.device = config.resolved_device()
        self.use_fp16 = config.resolved_use_fp16()

        self.model = None
        self.transform = None
        self._loaded = False

        if not self.config.lazy_load:
            self.load_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._loaded and self.model is not None and self.transform is not None:
            return

        try:
            import groundingdino.datasets.transforms as T
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
        except ImportError as exc:
            raise ImportError(
                "LocalGroundingDINODetectorModel requires the original `groundingdino` package. "
                "Install/build GroundingDINO locally, then retry."
            ) from exc

        config_file, checkpoint = self.config.require_existing_files()

        logger.info("Loading local GroundingDINO detector")
        logger.info("Config file: %s", config_file)
        logger.info("Checkpoint: %s", checkpoint)
        logger.info("Device: %s", self.device)
        logger.info("Classes/prompts: %s", self.config.classes)

        args = SLConfig.fromfile(config_file)
        args.device = self.device

        model = build_model(args)
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint_data["model"]), strict=False)
        model.to(self.device)
        model.eval()

        transform = T.Compose(
            [
                T.RandomResize(
                    [self.config.input_short_side],
                    max_size=self.config.input_max_size,
                ),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        self.model = model
        self.transform = transform
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded or self.model is None or self.transform is None:
            self.load_model()

    def unload_model(self) -> None:
        self.model = None
        self.transform = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_detection(device: str) -> DetectionResult:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
            "scores": torch.empty((0,), dtype=torch.float32, device=device),
            "labels": torch.empty((0,), dtype=torch.long, device=device),
        }

    @staticmethod
    def _tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu()

        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dimensions, got shape={tuple(tensor.shape)}.")

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

    @staticmethod
    def _cxcywh_normalized_to_xyxy_absolute(
        boxes: torch.Tensor,
        *,
        width: int,
        height: int,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0, 4), dtype=torch.float32, device=boxes.device)

        scale = torch.tensor(
            [width, height, width, height],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes = boxes * scale

        cx = boxes[:, 0]
        cy = boxes[:, 1]
        bw = boxes[:, 2]
        bh = boxes[:, 3]

        x0 = cx - bw / 2
        y0 = cy - bh / 2
        x1 = cx + bw / 2
        y1 = cy + bh / 2

        return torch.stack([x0, y0, x1, y1], dim=1).to(torch.float32)

    # ------------------------------------------------------------------
    # BaseExternalDetector API
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[torch.Tensor],
    ) -> list[DetectionResult]:
        """
        Run local GroundingDINO on a batch of images.

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
        assert self.transform is not None

        caption = self.config.caption()
        detections: list[DetectionResult] = []

        device_type = torch.device(self.device).type
        use_autocast = self.use_fp16 and device_type == "cuda"

        for tensor in images:
            image_pil = self._tensor_to_pil_rgb(tensor)
            width, height = image_pil.size

            image_tensor, _ = self.transform(image_pil, None)
            image_tensor = image_tensor.to(self.device)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.float16,
                    enabled=use_autocast,
                ):
                    outputs = self.model(image_tensor[None], captions=[caption])

            logits = outputs["pred_logits"].sigmoid()[0]
            boxes_cxcywh = outputs["pred_boxes"][0]

            scores = logits.max(dim=1)[0]
            keep = scores > self.config.score_threshold

            scores = scores[keep]
            boxes_cxcywh = boxes_cxcywh[keep]

            if len(scores) == 0:
                detections.append(self._empty_detection(self.device))
                continue

            boxes_xyxy = self._cxcywh_normalized_to_xyxy_absolute(
                boxes_cxcywh,
                width=width,
                height=height,
            )

            labels = torch.zeros(
                (len(scores),),
                dtype=torch.long,
                device=self.device,
            )

            detections.append(
                {
                    "boxes": boxes_xyxy.to(self.device),
                    "scores": scores.to(torch.float32).to(self.device),
                    "labels": labels,
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
            "classes": list(self.config.classes),
            "caption": self.config.caption(),
            "config_file": self.config.config_file,
            "checkpoint": self.config.checkpoint,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "score_threshold": self.config.score_threshold,
            "text_threshold": self.config.text_threshold,
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


EXTERNAL_DETECTORS.register_module(module=LocalGroundingDINODetectorModel)
