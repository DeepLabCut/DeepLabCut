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
import numpy as np
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)
from deeplabcut.pose_estimation_pytorch.models.modules import (  # ColoredKeypointEncoder,; StackedKeypointEncoder,
    BaseKeypointEncoder,
    KEYPOINT_ENCODERS,
)


@BACKBONES.register_module
class CondPreNet(BaseBackbone):
    """
    Wrapper module that adds a conditional preNet before any backbone.
    This allows to process image and condition features and prepare them for the main backbone.
    """

    def __init__(
        self,
        kpt_encoder: dict | BaseKeypointEncoder,
        backbone: dict | BaseBackbone,
        img_size: tuple[int, int] = (256, 256),
        **kwargs,
    ):
        """
        Initialize the PreNetWrapper.

        Args:
            backbone: The backbone model to wrap.
            img_size: The (height, width) of the input images.
        """
        pretrained = kwargs.pop("pretrained", False)
        if not isinstance(backbone, BaseBackbone):
            backbone["pretrained"] = pretrained
            backbone = BACKBONES.build(backbone)

        super().__init__(stride=backbone.stride, **kwargs)

        if not isinstance(kpt_encoder, BaseKeypointEncoder):
            if "img_size" not in kpt_encoder:
                kpt_encoder["img_size"] = img_size
            kpt_encoder = KEYPOINT_ENCODERS.build(kpt_encoder)
        self.cond_enc = kpt_encoder

        self.backbone = backbone
        self.rgb_preNet = self._make_preNet(
            num_inputs=3, num_outputs=3, input_image=True
        )
        self.cond_preNet = self._make_preNet(
            num_inputs=self.cond_enc.num_channels, num_outputs=3, input_image=False
        )

        self.init_weights()

    def _make_preNet(self, num_inputs, num_outputs, input_image=False):
        if not input_image:  # cond
            preNet = nn.Sequential(
                nn.Conv2d(
                    num_inputs, num_outputs, kernel_size=7, stride=1, padding="same"
                ),
                nn.BatchNorm2d(num_outputs),
            )
        else:
            preNet = nn.Sequential(
                nn.Conv2d(num_inputs, 64, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, num_outputs, kernel_size=7, stride=1, padding="same"),
                nn.BatchNorm2d(num_outputs),
            )
        return preNet

    def forward(self, x: torch.Tensor, cond_kpts: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Forward pass through the conditional preNet + backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            cond_kpts: Conditional keypoints of shape (batch_size, num_joints, 2).

        Returns:
            the feature map
        """
        # create conditional heatmap
        if isinstance(cond_kpts, torch.Tensor):
            cond_kpts = cond_kpts.detach().numpy()

        cond_hm = self.cond_enc(cond_kpts.squeeze(1), x.size()[2:])
        cond_hm = torch.from_numpy(cond_hm).float().to(x.device)
        cond_hm = cond_hm.permute(0, 3, 1, 2)  # (B, C, H, W)

        x0 = self.rgb_preNet(x)
        x1 = self.cond_preNet(cond_hm)
        x = x0 + x1

        return self.backbone(x)

    def init_weights(self):
        """Initialize PreNet weights from a Normal distribution."""
        for prenet in [self.rgb_preNet, self.cond_preNet]:
            for m in prenet.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
