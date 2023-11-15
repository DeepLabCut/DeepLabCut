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

from deeplabcut.pose_estimation_pytorch.models import model as models
from deeplabcut.pose_estimation_pytorch.runners.base import RUNNERS, Runner


@RUNNERS.register_module
class PoseRunner(Runner[models.PoseModel]):
    """Runner for pose estimation"""

    def __init__(
        self, model: models.PoseModel, optimizer: torch.optim.Optimizer, **kwargs
    ):
        """TODO: Update doc to generic (not pose) runner. Constructor of the Runner class.
        Args:
            model: The neural network for solving pose estimation task.
            optimizer: A PyTorch optimizer for updating model parameters.
            kwargs: Runner kwargs

        Returns:
            None

        Notes/TODO:
            Read stride from config file
        """
        super().__init__(model, optimizer, **kwargs)

    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Runner must be in train or eval mode, but {mode} was found."

        Returns:
            dict: {
                "total_loss": aggregate_loss,
                "aux_loss_1": loss_value,
                ...,
            }
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"BottomUpSolver must be in train or eval mode, but {mode} was found."
            )

        if mode == "train":
            self.optimizer.zero_grad()

        batch_inputs = batch["image"]
        batch_inputs = batch_inputs.to(self.device)
        head_outputs = self.model(batch_inputs)

        target = self.model.get_target(batch_inputs, head_outputs, batch["annotations"])

        losses_dict = self.model.get_loss(head_outputs, target)
        if mode == "train":
            losses_dict["total_loss"].backward()
            self.optimizer.step()

        return {k: v.detach().cpu().numpy() for k, v in losses_dict.items()}

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
            batch_outputs = self.model(batch_inputs)
            raw_predictions = self.model.get_predictions(batch_inputs, batch_outputs)

            for b in range(batch_size):
                image_predictions = {}
                for head, head_outputs in raw_predictions.items():
                    image_predictions[head] = {}
                    for pred_name, pred in head_outputs.items():
                        image_predictions[head][pred_name] = pred[b].cpu().numpy()
                batch_predictions.append(image_predictions)

        return batch_predictions
