# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/config/base.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExternalDetectorConfig(BaseModel):
    """
    Shared config for all DLC-compatible external object detectors.

    Concrete detector configs should subclass this.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Runtime
    device: str | None = Field(
        default=None,
        description="Torch device. If None, auto-select cuda, then mps, then cpu.",
    )
    use_fp16: bool | None = Field(
        default=None,
        description="If None, use fp16 automatically on CUDA only.",
    )

    # Inference behavior
    batch_size: int = Field(default=1, ge=1)
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_detections: int | None = Field(default=None, ge=1)
    largest_only: bool = False

    # Output normalization
    output_box_format: Literal["xywh"] = "xywh"
    coordinate_system: Literal["absolute_pixels"] = "absolute_pixels"
    min_box_area: float = Field(default=0.0, ge=0.0)
    clip_boxes: bool = True
    filter_invalid_boxes: bool = True

    # Image input handling
    image_color_order: Literal["RGB", "BGR"] = "RGB"

    # UX / lifecycle
    lazy_load: bool = True
    warmup_on_first_inference: bool = True
    show_progress: bool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str | None) -> str | None:
        if value is None:
            return value

        valid_prefixes = ("cpu", "cuda", "mps")
        if not value.startswith(valid_prefixes):
            raise ValueError(f"Unsupported device '{value}'. Expected one of: cpu, cuda, cuda:N, mps.")

        return value

    @staticmethod
    def resolve_device(device: str | None) -> str:
        if device is not None:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
            if device == "mps":
                mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                if not mps_available:
                    raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
            return device

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def resolved_device(self) -> str:
        return self.resolve_device(self.device)

    def resolved_use_fp16(self) -> bool:
        if self.use_fp16 is not None:
            return self.use_fp16
        return self.resolved_device().startswith("cuda")


class PromptedDetectorConfig(ExternalDetectorConfig):
    """
    Shared config for detectors that use text prompts/classes.
    """

    classes: list[str] = Field(
        default_factory=list,
        description="List of object classes or text prompts to detect.",
    )

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one class/prompt is required.")

        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Invalid class/prompt: {item!r}")
            cleaned.append(item.strip())

        return cleaned


class CheckpointDetectorConfig(ExternalDetectorConfig):
    checkpoint: str | Path | None = None
    allow_checkpoint_download: bool = False

    @field_validator("checkpoint")
    @classmethod
    def normalize_checkpoint(cls, value: str | Path | None) -> str | None:
        if value is None:
            return None
        return str(Path(value).expanduser())

    def require_existing_checkpoint(self) -> str:
        if self.checkpoint is None:
            raise ValueError("A checkpoint path is required.")

        path = Path(self.checkpoint)
        if path.is_file():
            return str(path)

        if self.allow_checkpoint_download:
            # Let the concrete detector decide how to download.
            return str(path)

        raise FileNotFoundError(
            f"Checkpoint file not found: {path}. Pass a valid checkpoint or set allow_checkpoint_download=True."
        )
