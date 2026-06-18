from __future__ import annotations

from pydantic import Field

from deeplabcut.pose_estimation_pytorch.models.detectors.external.config.base import (
    PromptedDetectorConfig,
)


class GroundingDINODetectorConfig(PromptedDetectorConfig):
    """
    Config for the Transformers GroundingDINO external detector adapter.
    """

    model_id: str = "IDEA-Research/grounding-dino-tiny"

    box_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    prompt_prefix: str = "a "
    lowercase_prompt: bool = True

    local_files_only: bool = False
    trust_remote_code: bool = False
    cache_dir: str | None = None

    def text_labels(self) -> list[str]:
        """
        Return prompt labels in the format expected by the processor.
        """
        labels = []
        for cls in self.classes:
            label = cls.strip()
            if self.prompt_prefix and not label.startswith(self.prompt_prefix):
                label = f"{self.prompt_prefix}{label}"
            if self.lowercase_prompt:
                label = label.lower()
            labels.append(label)
        return labels

    def formatted_prompt(self) -> str:
        prompt = ". ".join(self.text_labels()).strip()
        if prompt and not prompt.endswith("."):
            prompt += "."
        return prompt
