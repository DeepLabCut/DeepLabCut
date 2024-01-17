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

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class Residual(torch.nn.Module):
    """Residual block module.

    This module implements a residual block for the transformer layers.

    Attributes:
        fn: The function to apply in the residual block.
    """

    def __init__(self, fn: torch.nn.Module):
        """Initialize the Residual block.

        Args:
            fn: The function to apply in the residual block.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass through the Residual block.

        Args:
            x: Input tensor.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            Output tensor.
        """
        return self.fn(x, **kwargs) + x


class PreNorm(torch.nn.Module):
    """PreNorm block module.

    This module implements pre-normalization for the transformer layers.

    Attributes:
        dim: Dimension of the input tensor.
        fn: The function to apply after normalization.
        fusion_factor: Fusion factor for layer normalization.
                       Defaults to 1.
    """

    def __init__(self, dim: int, fn: torch.nn.Module, fusion_factor: int = 1):
        """Initialize the PreNorm block.

        Args:
            dim: Dimension of the input tensor.
            fn: The function to apply after normalization.
            fusion_factor: Fusion factor for layer normalization.
                           Defaults to 1.
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass through the PreNorm block.

        Args:
            x: Input tensor.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            Output tensor.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(torch.nn.Module):
    """FeedForward block module.

    This module implements the feedforward layer in the transformer layers.

    Attributes:
        dim: Dimension of the input tensor.
        hidden_dim: Dimension of the hidden layer.
        dropout: Dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """Initialize the FeedForward block.

        Args:
            dim: Dimension of the input tensor.
            hidden_dim: Dimension of the hidden layer.
            dropout: Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the FeedForward block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.net(x)


class Attention(torch.nn.Module):
    """Attention block module.

    This module implements the attention mechanism in the transformer layers.

    Attributes:
        dim: Dimension of the input tensor.
        heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.0.
        num_keypoints: Number of keypoints. Defaults to None.
        scale_with_head: Scale attention with the number of heads.
                         Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
        num_keypoints: int = None,
        scale_with_head: bool = False,
    ):
        """Initialize the Attention block.

        Args:
            dim: Dimension of the input tensor.
            heads: Number of attention heads. Defaults to 8.
            dropout: Dropout rate. Defaults to 0.0.
            num_keypoints: Number of keypoints. Defaults to None.
            scale_with_head: Scale attention with the number of heads.
                             Defaults to False.
        """
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = torch.nn.Linear(dim, dim * 3, bias=False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass through the Attention block.

        Args:
            x: Input tensor.
            mask: Attention mask. Defaults to None.

        Returns:
            Output tensor.
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class TransformerLayer(torch.nn.Module):
    """TransformerLayer block module.

    This module implements the Transformer layer in the transformer model.

    Attributes:
        dim: Dimension of the input tensor.
        depth: Depth of the transformer layer.
        heads: Number of attention heads.
        mlp_dim: Dimension of the MLP layer.
        dropout: Dropout rate.
        num_keypoints: Number of keypoints. Defaults to None.
        all_attn: Apply attention to all keypoints.
                  Defaults to False.
        scale_with_head: Scale attention with the number of heads.
                         Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        num_keypoints: int = None,
        all_attn: bool = False,
        scale_with_head: bool = False,
    ):
        """Initialize the TransformerLayer block.

        Args:
            dim: Dimension of the input tensor.
            depth: Depth of the transformer layer.
            heads: Number of attention heads.
            mlp_dim: Dimension of the MLP layer.
            dropout: Dropout rate.
            num_keypoints: Number of keypoints. Defaults to None.
            all_attn: Apply attention to all keypoints. Defaults to False.
            scale_with_head: Scale attention with the number of heads. Defaults to False.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dropout=dropout,
                                    num_keypoints=num_keypoints,
                                    scale_with_head=scale_with_head,
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, pos: torch.Tensor = None
    ):
        """Forward pass through the TransformerLayer block.

        Args:
            x: Input tensor.
            mask: Attention mask. Defaults to None.
            pos: Positional encoding. Defaults to None.

        Returns:
            Output tensor.
        """
        for idx, (attn, ff) in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_keypoints :] += pos
            x = attn(x, mask=mask)
            x = ff(x)
        return x
