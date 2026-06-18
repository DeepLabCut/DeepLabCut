from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from deeplabcut.pose_estimation_pytorch.models.detectors.external.config.base import (
    ExternalDetectorConfig,
)


class RTDETRV2DetectorConfig(ExternalDetectorConfig):
    """
    Hugging Face RT-DETRv2 detector config.

    Notes:
        This is a closed-vocabulary detector. For public checkpoints such as
        ``PekingU/rtdetr_v2_r18vd``, labels are usually COCO labels. This is not an
        open-vocabulary text-prompt detector like GroundingDINO.
    """

    model_id: str = Field(
        default="PekingU/rtdetr_v2_r18vd",
        description="Hugging Face model ID or local checkpoint directory.",
    )

    local_files_only: bool = Field(
        default=False,
        description="Whether to only use local Hugging Face files.",
    )

    trust_remote_code: bool = Field(
        default=False,
        description="Passed to Hugging Face from_pretrained(...).",
    )

    cache_dir: str | None = Field(
        default=None,
        description="Optional Hugging Face cache directory.",
    )

    target_classes: list[str] | None = Field(
        default=None,
        description=(
            "Optional class names to keep after model inference. If None, all classes "
            "are kept. Names are matched against model.config.id2label case-insensitively."
        ),
    )

    target_label_ids: list[int] | None = Field(
        default=None,
        description=(
            "Optional numeric label IDs to keep after model inference. If both "
            "target_classes and target_label_ids are provided, the union is used."
        ),
    )

    allow_missing_target_classes: bool = Field(
        default=False,
        description=(
            "If False, raise when a target class is not found in model.config.id2label. "
            "If True, missing target classes are ignored."
        ),
    )

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
