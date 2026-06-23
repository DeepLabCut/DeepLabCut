from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from deeplabcut.pose_estimation_pytorch.models.detectors.external.config.base import (
    PromptedDetectorConfig,
)


class LocalGroundingDINODetectorConfig(PromptedDetectorConfig):
    """
    Local/offline GroundingDINO detector config.

    Uses the original GroundingDINO package with a local Python config file and
    local checkpoint path.
    """

    config_file: str | Path = Field(
        ...,
        description="Path to GroundingDINO config file, e.g. GroundingDINO_SwinT_OGC.py.",
    )
    checkpoint: str | Path = Field(
        ...,
        description="Path to local GroundingDINO checkpoint, e.g. groundingdino_swint_ogc.pth.",
    )

    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    input_short_side: int = Field(default=800, ge=1)
    input_max_size: int = Field(default=1333, ge=1)

    @field_validator("config_file", "checkpoint")
    @classmethod
    def normalize_path(cls, value: str | Path) -> str:
        return str(Path(value).expanduser())

    def require_existing_files(self) -> tuple[str, str]:
        config_file = Path(self.config_file)
        checkpoint = Path(self.checkpoint)

        if not config_file.is_file():
            raise FileNotFoundError(f"GroundingDINO config file not found: {config_file}")

        if not checkpoint.is_file():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint}")

        return str(config_file), str(checkpoint)

    def caption(self) -> str:
        caption = " . ".join(self.classes).lower().strip()
        if not caption.endswith("."):
            caption += "."
        return caption
