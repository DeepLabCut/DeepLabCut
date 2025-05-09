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
from torch.nn import init
import torchvision.transforms.functional as TF


class CoAMBlock(nn.Module):
    """
    Conditional Attention Module (CoAM) block.
    """

    def __init__(
        self, spat_dims, channel_list, cond_enc, n_heads=1, channel_only=False
    ):
        super(CoAMBlock, self).__init__()
        self.att_layers = []
        self.spat_dims = spat_dims
        self.cond_enc = cond_enc
        d_cond = cond_enc.num_channels
        for i in range(len(spat_dims)):
            att_layer = DAModule(
                d_model=channel_list[i],
                d_cond=d_cond,
                kernel_size=3,
                H=spat_dims[i][1],
                W=spat_dims[i][0],
                n_heads=n_heads,
                channel_only=channel_only,
            )
            self.att_layers.append(att_layer)
        self.att_layers = nn.ModuleList(self.att_layers)

    def forward(self, y_list, cond_hm):
        # if not isinstance(self.cond_enc, (StackedKeypointEncoder, ColoredKeypointEncoder)):
        #     cond_hm = cond_hm[:,0].unsqueeze(1) # we only want one channel of the heatmap
        y_list_att = []
        for i in range(len(y_list)):
            y_att = self.att_layers[i](
                y_list[i],
                TF.resize(cond_hm, (self.spat_dims[i][1], self.spat_dims[i][0])),
            )
            y_list_att.append(y_att)
        return y_list_att


# modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/DANet.py
class PositionAttentionModule(nn.Module):
    def __init__(
        self, d_model=512, d_cond=3, kernel_size=3, H=7, W=7, n_heads=1, self_att=False
    ):
        super().__init__()
        self.cnn = nn.Conv2d(
            d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )
        self.pa = ScaledDotProductAttention(
            in_dim_q=d_model, in_dim_k=d_model, d_k=d_model, d_v=d_model, h=n_heads
        )
        self.self_att = self_att
        if not self_att:
            self.cnn_cond = nn.Conv2d(
                d_cond, d_cond, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
            )
            self.pa = ScaledDotProductAttention(
                in_dim_q=d_cond, in_dim_k=d_model, d_k=d_model, d_v=d_model, h=n_heads
            )

    def forward(self, x, cond=None):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1).permute(0, 2, 1)  # bs,h*w,c

        if not self.self_att:
            _, c_cond, _, _ = cond.shape
            y_cond = self.cnn_cond(cond)
            y_cond = y_cond.view(bs, c_cond, -1).permute(0, 2, 1)
            y = self.pa(y_cond, y, y)  # bs,h*w,c

        else:
            y = self.pa(y, y, y)

        return y


class ChannelAttentionModule(nn.Module):
    def __init__(
        self, d_model=512, d_cond=3, kernel_size=3, H=7, W=7, n_heads=1, self_att=False
    ):
        super().__init__()
        self.cnn = nn.Conv2d(
            d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )
        self.self_att = self_att
        if not self_att:
            self.cnn_cond = nn.Conv2d(
                d_cond, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
            )
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=n_heads)

    def forward(self, x, cond=None):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)  # bs,c,h*w

        if not self.self_att:
            y_cond = self.cnn_cond(cond)
            y_cond = y_cond.view(bs, c, -1)
            y = self.pa(y_cond, y, y)  # bs,c_cond,h*w
        else:
            y = self.pa(y, y, y)  # bs,c,h*w

        return y


class DAModule(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_cond=3,
        kernel_size=3,
        H=7,
        W=7,
        n_heads=1,
        channel_only=False,
    ):
        super().__init__()
        self.channel_only = channel_only
        if not channel_only:
            self.position_attention_module = PositionAttentionModule(
                d_model=d_model,
                d_cond=d_cond,
                kernel_size=kernel_size,
                H=H,
                W=W,
                n_heads=n_heads,
            )
        self.channel_attention_module = ChannelAttentionModule(
            d_model=d_model,
            d_cond=d_cond,
            kernel_size=kernel_size,
            H=H,
            W=W,
            n_heads=n_heads,
        )

    def forward(self, input, cond):

        bs, c, h, w = input.shape

        c_out = self.channel_attention_module(input, cond)
        c_out = c_out.view(bs, c, h, w)

        if self.channel_only:
            return input * c_out

        p_out = self.position_attention_module(input, cond)
        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)

        return input + (p_out + c_out)


class SelfDAModule(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(
            d_model=d_model,
            d_cond=None,
            kernel_size=kernel_size,
            H=H,
            W=W,
            self_att=True,
        )
        self.channel_attention_module = ChannelAttentionModule(
            d_model=d_model,
            d_cond=None,
            kernel_size=kernel_size,
            H=H,
            W=W,
            self_att=True,
        )

    def forward(self, input):

        bs, c, h, w = input.shape

        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)

        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
        c_out = c_out.view(bs, c, h, w)

        return p_out + c_out


class SelfAttentionModule_CoAM(nn.Module):
    def __init__(self, spat_dims, channel_list):
        super(SelfAttentionModule_CoAM, self).__init__()
        self.att_layers = []
        for i in range(len(spat_dims)):
            att_layer = SelfDAModule(
                d_model=channel_list[i],
                kernel_size=3,
                H=spat_dims[i][0],
                W=spat_dims[i][1],
            )
            self.att_layers.append(att_layer)
        self.att_layers = nn.ModuleList(self.att_layers)

    def forward(self, y_list, *args):
        y_list_att = []
        for i in range(len(y_list)):
            y_att = self.att_layers[i](y_list[i])
            y_list_att.append(y_att)
        return y_list_att


# taken from: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SelfAttention.py
class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, in_dim_q, in_dim_k, d_k, d_v, h, dropout=0.1, rev=False):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()

        # 'rev': condition is key/value and orig. feature map is query
        if rev:
            d_model = in_dim_q
        else:
            d_model = in_dim_k
        self.fc_q = nn.Linear(in_dim_q, h * d_k)
        self.fc_k = nn.Linear(in_dim_k, h * d_k)
        self.fc_v = nn.Linear(in_dim_k, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(
        self, queries, keys, values, attention_mask=None, attention_weights=None
    ):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = (
            self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        )  # (b_s, h, nq, d_k)
        k = (
            self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        )  # (b_s, h, d_k, nk)
        v = (
            self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        )  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = (
            torch.matmul(att, v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


# taken from: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SimplifiedSelfAttention.py
class SimplifiedScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, h, dropout=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(
        self, queries, keys, values, attention_mask=None, attention_weights=None
    ):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(
            0, 2, 1, 3
        )  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(
            0, 2, 3, 1
        )  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(
            0, 2, 1, 3
        )  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = (
            torch.matmul(att, v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
