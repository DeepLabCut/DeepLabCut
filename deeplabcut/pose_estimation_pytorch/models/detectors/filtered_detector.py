import torch
from torch import nn


class FilteredDetector(nn.Module):
    def __init__(self, base_model: nn.Module, class_id: int):
        """
        Wrap a torchvision detector to return predictions only for a single class.

        Args:
            base_model: A torchvision-style object detector.
            class_id: The integer class ID to keep (e.g., 1 for 'person' in COCO).
        """
        super().__init__()
        self.base_model = base_model
        self.class_id = class_id

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """
        Arguments:
            images: list of input images as Tensors

        Returns:
            List of dicts, each containing boxes/scores/labels filtered to the specified class.
        """
        with torch.no_grad():
            outputs = self.base_model(images)

        filtered_outputs = []
        for output in outputs:
            mask = output["labels"] == self.class_id
            filtered_output = {
                "boxes": output["boxes"][mask],
                "scores": output["scores"][mask],
                "labels": output["labels"][mask],
            }
            filtered_outputs.append(filtered_output)

        losses = {}
        return losses, filtered_outputs
