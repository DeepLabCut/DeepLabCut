# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/dart/config.py

from pydantic import Field, field_validator

from deeplabcut.pose_estimation_pytorch.models.detectors.external.config.base import (
    CheckpointDetectorConfig,
    PromptedDetectorConfig,
)


class SAM3DARTDetectorConfig(PromptedDetectorConfig, CheckpointDetectorConfig):
    imgsz: int = Field(default=1008, ge=14)
    nms_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    presence_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    compile_mode: str | None = None
    skip_blocks: set[int] | None = None
    mask_blocks: list[str] | None = None

    show_progress: bool = True

    detection_only: bool = True

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, value: int) -> int:
        if value % 14 != 0:
            raise ValueError(f"SAM3/DART imgsz must be divisible by 14, got {value}.")
        return value
