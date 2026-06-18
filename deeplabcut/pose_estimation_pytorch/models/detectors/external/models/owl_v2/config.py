# deeplabcut/pose_estimation_pytorch/models/detectors/external/models/owl_v2/config.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from deeplabcut.pose_estimation_pytorch.models.detectors.external.config.base import (
    PromptedDetectorConfig,
)


class OWLv2DetectorConfig(PromptedDetectorConfig):
    """
    Hugging Face OWLv2 detector config.

    OWLv2 is an open-vocabulary detector. The ``classes`` field contains text prompts
    or class names. If ``prompt_mode='template'``, each class is formatted with
    ``prompt_template`` before being sent to the model.
    """

    model_id: str = Field(
        default="google/owlv2-base-patch16-ensemble",
        description="Hugging Face model ID or local checkpoint directory.",
    )

    prompt_mode: Literal["raw", "template"] = Field(
        default="template",
        description=(
            "If 'raw', use classes exactly as provided. If 'template', format each class with prompt_template."
        ),
    )

    prompt_template: str = Field(
        default="a photo of a {}",
        description="Template used when prompt_mode='template'.",
    )

    local_files_only: bool = False
    trust_remote_code: bool = False
    cache_dir: str | None = None

    def text_labels(self) -> list[str]:
        if self.prompt_mode == "raw":
            return list(self.classes)

        return [self.prompt_template.format(label) for label in self.classes]

    def hf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs
