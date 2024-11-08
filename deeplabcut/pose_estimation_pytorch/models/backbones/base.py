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

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

BACKBONES = Registry("backbones", build_func=build_from_cfg)


class BaseBackbone(ABC, nn.Module):
    """Base Backbone class for pose estimation.

    Attributes:
        stride: the stride for the backbone
        freeze_bn_weights: freeze weights of batch norm layers during training
        freeze_bn_stats: freeze stats of batch norm layers during training
    """

    def __init__(
        self,
        stride: int | float,
        freeze_bn_weights: bool = True,
        freeze_bn_stats: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.freeze_bn_weights = freeze_bn_weights
        self.freeze_bn_stats = freeze_bn_stats

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract method for the forward pass through the backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            a feature map for the input, of shape (batch_size, c', h', w')
        """
        pass

    def freeze_batch_norm_layers(self) -> None:
        """Freezes batch norm layers

        Running mean + var are always given to F.batch_norm, except when the layer is
        in `train` mode and track_running_stats is False, see
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
        So to 'freeze' the running stats, the only way is to set the layer to "eval"
        mode.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if self.freeze_bn_weights:
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
                if self.freeze_bn_stats:
                    module.eval()

    def train(self, mode: bool = True) -> None:
        """Sets the module in training or evaluation mode.

        Args:
            mode: whether to set training mode (True) or evaluation mode (False)
        """
        super().train(mode)
        if self.freeze_bn_weights or self.freeze_bn_stats:
            self.freeze_batch_norm_layers()


class HuggingFaceWeightsMixin:
    """Mixin for backbones where the pretrained weights are stored on HuggingFace"""

    def __init__(
        self,
        backbone_weight_folder: str | Path | None = None,
        repo_id: str = "DeepLabCut/DeepLabCut-Backbones",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if backbone_weight_folder is None:
            backbone_weight_folder = Path(__file__).parent / "pretrained_weights"
        else:
            backbone_weight_folder = Path(backbone_weight_folder).resolve()

        self.backbone_weight_folder = backbone_weight_folder
        self.repo_id = repo_id

    def download_weights(self, filename: str, force: bool = False) -> Path:
        """Downloads the backbone weights from the HuggingFace repo

        Args:
            filename: The name of the model file to download in the repo.
            force: Whether to re-download the file if it already exists locally.

        Returns:
            The path to the model snapshot.
        """
        model_path = self.backbone_weight_folder / filename
        if model_path.exists():
            if not force:
                return model_path
            model_path.unlink()

        logging.info(f"Downloading the pre-trained backbone to {model_path}")
        self.backbone_weight_folder.mkdir(exist_ok=True, parents=False)
        output_path = Path(
            hf_hub_download(
                self.repo_id, filename, cache_dir=self.backbone_weight_folder
            )
        )

        # resolve gets the actual path if the output path is a symlink
        output_path = output_path.resolve()
        # move to the target path
        output_path.rename(model_path)

        # delete downloaded artifacts
        uid, rid = self.repo_id.split("/")
        artifact_dir = self.backbone_weight_folder / f"models--{uid}--{rid}"
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        return model_path
