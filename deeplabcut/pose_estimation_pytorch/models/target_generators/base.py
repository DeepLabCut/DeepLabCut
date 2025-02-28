#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

TARGET_GENERATORS = Registry("target_generators", build_func=build_from_cfg)


class BaseGenerator(ABC, nn.Module):  # TODO: Should this really be a module?
    """Generates target maps from ground truth annotations to train models

    The outputs of the target generator are used to compute losses for model heads. If
    the head outputs "heatmap" and "offset" tensors, then the corresponding generator
    must output target "heatmap" and "offset" tensors. The targets themselves are
    dictionaries, and passed as keyword-arguments to the criterions. This allows to
    pass masks to the criterions.

    Generally, this means that for each head output (such as "heatmap"), a dict will be
    generated with a "target" key (for the target heatmap) and optionally a "weights"
    key (see the WeightedCriterion classes).
    """

    def __init__(self, label_keypoint_key: str = "keypoints"):
        super().__init__()
        self.label_keypoint_key = label_keypoint_key

    @abstractmethod
    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor], labels: dict
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Generates targets

        Args:
            stride: the stride of the model
            outputs: output of a model head
            labels: the labels for the inputs (each tensor should have shape (b, ...))

        Returns:
            a dictionary mapping the heads to the inputs of the criterion
                {
                    "heatmap": {
                        "target": heatmaps,
                        "weights":  heatmap_weights,
                    },
                    "locref": {
                        "target": locref_map,
                        "weights": locref_weights,
                    }
                }
        """


@TARGET_GENERATORS.register_module
class SequentialGenerator(BaseGenerator):
    def __init__(self, generators: list[dict], label_keypoint_key: str = "keypoints"):
        super().__init__(label_keypoint_key)
        self._generators = [TARGET_GENERATORS.build(dict_) for dict_ in generators]

    @property
    def generators(self):
        return self._generators

    def forward(
        self, stride: int, outputs: dict[str, torch.Tensor], labels: dict
    ) -> dict[str, dict[str, torch.Tensor]]:
        dict_ = {}
        for gen in self.generators:
            dict_.update(gen(stride, outputs, labels))
        return dict_

    def __repr__(self):
        generators_repr = ", ".join(repr(gen) for gen in self._generators)
        return (
            f"<{self.__class__.__name__}(generators=[{generators_repr}], "
            f"label_keypoint_key='{self.label_keypoint_key}')>"
        )
