#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import deeplabcut.pose_estimation_pytorch.models.detectors as detectors
from deeplabcut.pose_estimation_pytorch.runners import PoseRunner
from deeplabcut.pose_estimation_pytorch.runners.base import RUNNERS, Runner


@RUNNERS.register_module
class DetectorRunner(Runner[detectors.BaseDetector]):
    """Runner for object detection"""

    def __init__(
        self,
        model: detectors.BaseDetector,
        optimizer: torch.optim.Optimizer,
        max_individuals: int,
        snapshot_prefix: str = "detector-snapshot",
        **kwargs,
    ):
        """

        Args:
            model:
            optimizer:
            max_individuals:
            **kwargs: Runner kwargs
        """
        super().__init__(model, optimizer, snapshot_prefix=snapshot_prefix, **kwargs)
        self.max_individuals = max_individuals

    def step(
        self,
        batch: dict[str, Any],
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Runner must be in train or eval mode, but {mode} was found."

        Returns:
            dict: {
                'total_loss': torch.Tensor,
                'aux_loss_1': torch.Tensor,
                ...,
            }
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"DetectorSolver must be in train or eval mode, but {mode} was found."
            )

        if mode == "train":
            self.optimizer.zero_grad()
        else:
            # Override base class
            # No losses returned in train mode;
            #   see https://stackoverflow.com/a/65347721
            # Should be safe as BN is frozen;
            #   see https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333/12
            self.model.train()

        images = batch["image"]
        images = images.to(self.device)

        target = self.model.get_target(
            batch["annotations"]
        )  # (batch_size, channels, h, w)
        for item in target:  # target is a list here
            for key in item:
                if item[key] is not None:
                    item[key] = torch.tensor(item[key]).to(self.device)

        losses, _ = self.model(images, target)
        losses["total_loss"] = sum(loss_part for loss_part in losses.values())
        if mode == "train":
            losses["total_loss"].backward()
            self.optimizer.step()

        return {k: v.detach().cpu().numpy() for k, v in losses.items()}

    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": "poses": np.ndarray},
            ]
        """
        # TODO: iterates over batch one element at a time
        batch_size = 1
        batch_predictions = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_inputs = batch_inputs.to(self.device)
            _, raw_predictions = self.model(batch_inputs)

            for b, item in enumerate(raw_predictions):
                # take the top-k bounding boxes as individuals
                batch_predictions.append(
                    {
                        "detection": {
                            "bboxes": item["boxes"][: self.max_individuals]
                            .cpu()
                            .numpy(),
                        },
                    }
                )

        return batch_predictions
