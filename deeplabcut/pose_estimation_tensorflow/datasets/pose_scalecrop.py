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


from .factory import PoseDatasetFactory
from .pose_deterministic import DeterministicPoseDataset


@PoseDatasetFactory.register("scalecrop")
class ScalecropPoseDataset(DeterministicPoseDataset):
    def __init__(self, cfg):
        super(ScalecropPoseDataset, self).__init__(cfg)
        self.cfg["deterministic"] = False
        self.max_input_sizesquare = cfg.get("max_input_size", 1500) ** 2
        self.min_input_sizesquare = cfg.get("min_input_size", 64) ** 2
        self.locref_scale = 1.0 / cfg["locref_stdev"]
        self.stride = cfg["stride"]
        self.half_stride = self.stride / 2
        self.scale_jitter_lo = cfg.get("scale_jitter_lo", 0.75)
        self.scale_jitter_up = cfg.get("scale_jitter_up", 1.25)

        self.cfg["crop"] = cfg.get("crop", True)
        self.cfg["cropratio"] = cfg.get("cropratio", 0.4)

        # what is the minimal frames size for cropping plus/minus ie.. [-100,100]^2 for an arb. joint
        self.cfg["minsize"] = cfg.get("minsize", 100)
        self.cfg["leftwidth"] = cfg.get("leftwidth", 400)
        self.cfg["rightwidth"] = cfg.get("rightwidth", 400)
        self.cfg["topheight"] = cfg.get("topheight", 400)
        self.cfg["bottomheight"] = cfg.get("bottomheight", 400)
