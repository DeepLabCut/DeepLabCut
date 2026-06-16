# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/grounding_dino/config.py
from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from deeplabcut.pose_estimation_pytorch.models.detectors.external.models.config.base import (
    PromptedDetectorConfig,
)


class GroundingDINODetectorConfig(PromptedDetectorConfig):
    """
    Config for the installed GroundingDINO external detector adapter.
    """

    model_config_path: str | Path
    model_checkpoint_path: str | Path

    box_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    prompt_separator: str = " . "
    append_period: bool = True

    @field_validator("model_config_path", "model_checkpoint_path")
    @classmethod
    def normalize_path(cls, value: str | Path) -> str:
        return str(Path(value).expanduser())

    def formatted_prompt(self) -> str:
        prompt = self.prompt_separator.join(self.classes).strip()
        if self.append_period and not prompt.endswith("."):
            prompt += " ."
        return prompt
