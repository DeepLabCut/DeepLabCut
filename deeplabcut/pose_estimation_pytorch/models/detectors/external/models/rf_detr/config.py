# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/rf_detr/config.py
from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from deeplabcut.pose_estimation_pytorch.models.detectors.external.config.base import (
    ExternalDetectorConfig,
)


class RFDETRDetectorConfig(ExternalDetectorConfig):
    """
    Hugging Face RF-DETR detector config.

    Uses Roboflow RF-DETR checkpoints through:
        AutoImageProcessor
        AutoModelForObjectDetection
    """

    model_id: str = Field(
        default="Roboflow/rf-detr-medium",
        description="Hugging Face RF-DETR model ID or local checkpoint directory.",
    )

    local_files_only: bool = False
    trust_remote_code: bool = False
    cache_dir: str | None = None

    target_classes: list[str] | None = Field(
        default=None,
        description=(
            "Optional class names to keep after inference. If None, all classes are kept. "
            "Names are matched against model.config.id2label case-insensitively."
        ),
    )

    target_label_ids: list[int] | None = Field(
        default=None,
        description=(
            "Optional numeric label IDs to keep after inference. If both target_classes "
            "and target_label_ids are provided, the union is used."
        ),
    )

    allow_missing_target_classes: bool = False

    @field_validator("target_classes")
    @classmethod
    def normalize_target_classes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Invalid target class: {item!r}")
            cleaned.append(item.strip())

        return cleaned

    @field_validator("target_label_ids")
    @classmethod
    def normalize_target_label_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None

        return [int(v) for v in value]

    def hf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs
