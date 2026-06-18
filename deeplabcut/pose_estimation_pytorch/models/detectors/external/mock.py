from __future__ import annotations

import torch

from .base import EXTERNAL_DETECTORS, BaseExternalDetector


@EXTERNAL_DETECTORS.register_module
class MockExternalDetector(BaseExternalDetector):
    """
    Simple detector for testing plumbing.
    Returns one centered box per image.
    """

    def __init__(self, score: float = 0.9, label: int = 1) -> None:
        super().__init__()
        self.score = score
        self.label = label

    def predict(self, images: list[torch.Tensor]):
        outputs = []
        for image in images:
            _, h, w = image.shape
            box = torch.tensor([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=torch.float32)
            score = torch.tensor([self.score], dtype=torch.float32)
            label = torch.tensor([self.label], dtype=torch.long)
            outputs.append(
                {
                    "boxes": box,
                    "scores": score,
                    "labels": label,
                }
            )
        return outputs
