# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/grounding_dino/grounding_dino.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm

from deeplabcut.pose_estimation_pytorch.data.bboxes import _xyxy_to_xywh
from deeplabcut.pose_estimation_pytorch.models.detectors.external import BaseExternalDetector
from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.gdino.config import (
    GroundingDINODetectorConfig,
)

logger = logging.getLogger(__name__)


def _coerce_to_pil_rgb(item: Any, *, color_order: str = "RGB") -> Image.Image:
    if isinstance(item, tuple) and len(item) > 0:
        item = item[0]

    if isinstance(item, Image.Image):
        return item if item.mode == "RGB" else item.convert("RGB")

    if isinstance(item, (str, Path)):
        with Image.open(item) as img:
            return img.convert("RGB")

    if isinstance(item, np.ndarray):
        arr = item

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3:
            raise ValueError(f"Expected image ndarray with shape HxWxC, got {arr.shape}.")

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        if arr.shape[2] != 3:
            raise ValueError(f"Expected 3 or 4 image channels, got {arr.shape}.")

        if color_order == "BGR":
            arr = arr[:, :, ::-1]
        elif color_order != "RGB":
            raise ValueError(f"Unsupported color_order={color_order!r}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported image input type: {type(item)!r}")


class GroundingDINODetectorModel(BaseExternalDetector):
    """
    DLC external-detector adapter for installed GroundingDINO.

    Uses GroundingDINO as an imported dependency rather than copying/modifying
    GroundingDINO model internals.
    """

    backend_name = "grounding_dino"

    def __init__(self, config: GroundingDINODetectorConfig) -> None:
        super().__init__()

        self.config = config
        self.device = config.resolved_device()
        self.use_fp16 = config.resolved_use_fp16()

        self.model = None
        self._transform = None
        self._loaded = False

        if not self.config.lazy_load:
            self.load_model()

    def load_model(self) -> None:
        if self._loaded and self.model is not None:
            return

        try:
            import groundingdino.datasets.transforms as T
            from groundingdino.util.inference import load_model
        except ImportError as exc:
            raise ImportError(
                "GroundingDINO is not installed. Install it for example with:\n"
                "  pip install git+https://github.com/IDEA-Research/GroundingDINO.git\n"
                "or clone the repository and run `pip install -e .`."
            ) from exc

        logger.info("Loading GroundingDINO detector")
        logger.info("Device: %s", self.device)
        logger.info("Config: %s", self.config.model_config_path)
        logger.info("Checkpoint: %s", self.config.model_checkpoint_path)
        logger.info("Classes/prompts: %s", self.config.classes)

        self.model = load_model(
            model_config_path=str(self.config.model_config_path),
            model_checkpoint_path=str(self.config.model_checkpoint_path),
            device=self.device,
        )
        self.model.to(self.device)
        self.model.eval()

        # Same transform pattern used by GroundingDINO's inference helpers.
        self._transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded or self.model is None or self._transform is None:
            self.load_model()

    @staticmethod
    def _empty_arrays() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    def _prepare_image_tensor(self, image: Image.Image) -> torch.Tensor:
        assert self._transform is not None
        image_tensor, _ = self._transform(image, None)
        return image_tensor

    def _predict_one_xyxy(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Run GroundingDINO for one PIL image.

        Returns:
            boxes_xyxy:
                Absolute pixel XYXY boxes.
            scores:
                Detection scores.
        """
        self.ensure_loaded()
        assert self.model is not None

        from groundingdino.util.inference import predict

        width, height = image.size
        image_tensor = self._prepare_image_tensor(image)

        boxes_cxcywh, logits, _phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.config.formatted_prompt(),
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.device,
        )

        if boxes_cxcywh.numel() == 0:
            return self._empty_arrays()

        # GroundingDINO returns normalized cxcywh boxes.
        boxes_xyxy = box_convert(
            boxes=boxes_cxcywh,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        scale = torch.tensor(
            [width, height, width, height],
            dtype=boxes_xyxy.dtype,
            device=boxes_xyxy.device,
        )
        boxes_xyxy = boxes_xyxy * scale

        if logits.ndim == 2:
            scores = logits.max(dim=1).values
        else:
            scores = logits

        return (
            boxes_xyxy.detach().cpu().numpy().astype(np.float32, copy=False),
            scores.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def _postprocess(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
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

        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)

        return (
            boxes_xywh.astype(np.float32, copy=False),
            scores.astype(np.float32, copy=False),
        )

    @staticmethod
    def _format_dlc_output(
        boxes_xywh: np.ndarray,
        scores: np.ndarray,
    ) -> dict[str, np.ndarray]:
        boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
            raise ValueError(f"Expected bboxes shape [N, 4], got {boxes_xywh.shape}.")

        if scores.ndim != 1:
            raise ValueError(f"Expected bbox_scores shape [N], got {scores.shape}.")

        if len(boxes_xywh) != len(scores):
            raise ValueError(f"Expected one score per bbox, got {len(scores)} scores for {len(boxes_xywh)} boxes.")

        return {
            "bboxes": boxes_xywh,
            "bbox_scores": scores,
        }

    def inference(self, images, shelf_writer=None):
        _ = shelf_writer

        self.ensure_loaded()

        outputs: list[dict[str, np.ndarray]] = []
        iterator = tqdm(images) if self.config.show_progress else images

        with torch.inference_mode():
            for item in iterator:
                image = _coerce_to_pil_rgb(
                    item,
                    color_order=self.config.image_color_order,
                )

                boxes_xyxy, scores = self._predict_one_xyxy(image)

                boxes_xywh, scores = self._postprocess(
                    boxes_xyxy=boxes_xyxy,
                    scores=scores,
                    image_size=image.size,
                )

                outputs.append(self._format_dlc_output(boxes_xywh, scores))

        return outputs

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "backend": self.backend_name,
            "classes": list(self.config.classes),
            "formatted_prompt": self.config.formatted_prompt(),
            "model_config_path": str(self.config.model_config_path),
            "model_checkpoint_path": str(self.config.model_checkpoint_path),
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
            "box_format": "xywh",
            "coordinate_system": "absolute_pixels",
            "config": self.config.model_dump(mode="json"),
        }

    def unload_model(self) -> None:
        self.model = None
        self._transform = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
