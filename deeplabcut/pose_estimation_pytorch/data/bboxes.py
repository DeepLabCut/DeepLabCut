from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

BBoxFormat = Literal["xywh", "xyxy"]
EvalMode: TypeAlias = Literal["train", "test"]


class DetectorContext(TypedDict):
    bboxes: np.ndarray
    bbox_scores: np.ndarray


ImageWithContext: TypeAlias = tuple[Path, DetectorContext]
ImagesWithContext: TypeAlias = list[ImageWithContext]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _numpy_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_jsonable(x) for x in obj]
    return obj


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


# -----------------------------------------------------------------------------
# Base model
# -----------------------------------------------------------------------------


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# -----------------------------------------------------------------------------
# BBox schemas
# -----------------------------------------------------------------------------


class BBoxEntry(StrictBaseModel):
    """
    Bounding box output for one image.

    `bboxes` are stored in pixel coordinates, with format declared by `bbox_format`.
    `bbox_scores` is aligned one-to-one with `bboxes`.
    """

    bboxes: list[tuple[float, float, float, float]]
    bbox_scores: list[float]
    bbox_format: BBoxFormat = "xyxy"
    image_path: Path | None = None

    @classmethod
    def from_detector_context(
        cls,
        context: DetectorContext,
        *,
        image_path: Path | None = None,
        bbox_format: BBoxFormat = "xywh",
    ) -> BBoxEntry:
        """
        Build a schema entry from DLC-style detector context.

        Args:
            context:
                Expected format:
                    {
                        "bboxes": np.ndarray[N, 4],
                        "bbox_scores": np.ndarray[N]
                    }
            image_path:
                Optional path of the corresponding image.
            bbox_format:
                Format of `context["bboxes"]`.
                Use:
                  - "xywh" for DLC postprocessed detector outputs / top-down context
                  - "xyxy" if adapting raw detector outputs before DLC postprocessing

        Returns:
            BBoxEntry
        """
        if "bboxes" not in context:
            raise ValueError("Detector context must contain 'bboxes'.")

        bboxes = np.asarray(context["bboxes"], dtype=np.float32).reshape(-1, 4)

        if "bbox_scores" in context:
            scores = np.asarray(context["bbox_scores"], dtype=np.float32).reshape(-1)
        else:
            # Allow score-less contexts, but fill with 1.0
            scores = np.ones((len(bboxes),), dtype=np.float32)

        if len(scores) != len(bboxes):
            raise ValueError(f"Expected one bbox score per bbox, but got {len(scores)} scores for {len(bboxes)} boxes.")

        return cls(
            bboxes=[tuple(map(float, box)) for box in bboxes],
            bbox_scores=[float(s) for s in scores],
            bbox_format=bbox_format,
            image_path=image_path,
        )

    def to_array(self, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
        """Return bboxes as a NumPy array of shape [N, 4]."""
        return np.asarray(self.bboxes, dtype=dtype).reshape(-1, 4)

    def to_xywh(self, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
        """Return bboxes in xywh format."""
        boxes = self.to_array(dtype=dtype)
        if self.bbox_format == "xyxy":
            boxes = _xyxy_to_xywh(boxes)
        return boxes

    def to_xyxy(self, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
        """Return bboxes in xyxy format."""
        boxes = self.to_array(dtype=dtype)
        if self.bbox_format == "xywh":
            boxes = _xywh_to_xyxy(boxes)
        return boxes

    def to_detector_context(
        self,
        *,
        dtype: np.dtype[Any] = np.float32,
        target_format: BBoxFormat = "xywh",
    ) -> DetectorContext:
        """
        Convert this entry to DLC detector context format.

        Args:
            dtype:
                NumPy dtype for emitted arrays.
            target_format:
                Desired bbox format in the returned context.
                For most DLC top-down dataset / pose use, this should be "xywh".

        Returns:
            {
                "bboxes": np.ndarray[N, 4],
                "bbox_scores": np.ndarray[N],
            }
        """
        if target_format == "xywh":
            bboxes = self.to_xywh(dtype=dtype)
        else:
            bboxes = self.to_xyxy(dtype=dtype)

        return {
            "bboxes": bboxes,
            "bbox_scores": np.asarray(self.bbox_scores, dtype=dtype),
        }


class BBoxes(StrictBaseModel):
    train: list[BBoxEntry] = Field(default_factory=list)
    test: list[BBoxEntry] = Field(default_factory=list)

    @classmethod
    def from_file(cls, json_file: Path, missing_ok: bool = False) -> BBoxes:
        if not json_file.exists():
            if missing_ok:
                return cls()
            raise FileNotFoundError(f"BBoxes file not found: {json_file}")
        return cls.from_json(json_file.read_text(encoding="utf-8"))

    @classmethod
    def from_json(cls, json_str: str) -> BBoxes:
        return cls.model_validate_json(json_str)

    def dump_json(self, json_file: Path) -> None:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        json_file.write_text(self.model_dump_json(indent=4), encoding="utf-8")

    def to_images_with_context(
        self,
        image_paths: list[Path],
        mode: EvalMode,
        *,
        target_format: BBoxFormat = "xywh",
    ) -> ImagesWithContext:
        """
        Zip image paths with detector context in DLC expected format.
        """
        mode_bboxes = getattr(self, mode)
        if len(image_paths) != len(mode_bboxes):
            raise ValueError(f"Got {len(image_paths)} {mode} images but {len(mode_bboxes)} bbox entries.")

        return [
            (
                image_path,
                bbox_entry.to_detector_context(target_format=target_format),
            )
            for image_path, bbox_entry in zip(image_paths, mode_bboxes, strict=False)
        ]
