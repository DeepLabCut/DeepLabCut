"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import torch
import torch.nn as nn
from .backbones.vit_pytorch import dlc_base_kpt_TransReID


class build_dlc_transformer(nn.Module):
    def __init__(self, cfg, in_chans, kpt_num, factory):
        super(build_dlc_transformer, self).__init__()
        self.cos_layer = cfg["cos_layer"]
        self.in_planes = 128
        self.kpt_num = kpt_num
        self.base = factory["dlc_transreid"](
            in_chans=in_chans,
            sie_xishu=cfg["sie_coe"],
            drop_path_rate=cfg["drop_path"],
            drop_rate=cfg["drop_out"],
            attn_drop_rate=cfg["att_drop_rate"],
            kpt_num=kpt_num,
        )

        self.classifier = nn.Identity()

        self.bottleneck = nn.Identity()

        self.ID_LOSS_TYPE = "cosface"

    def forward(self, x):
        global_feat = self.base(x)

        feat = self.bottleneck(global_feat)

        q = self.classifier(feat)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        q = q.div(norm)
        return q

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))


__factory_T_type = {
    "dlc_transreid": dlc_base_kpt_TransReID,
}


def make_dlc_model(cfg, feature_dim, kpt_num):

    model = build_dlc_transformer(cfg, feature_dim, kpt_num, __factory_T_type)

    return model
