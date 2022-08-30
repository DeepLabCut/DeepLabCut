"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import abc
import numpy as np


class BasePoseDataset(metaclass=abc.ABCMeta):
    # TODO Finish implementing actual abstract class
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def load_dataset(self):
        ...

    @abc.abstractmethod
    def next_batch(self):
        ...

    def sample_scale(self):
        if self.cfg.get("deterministic", False):
            np.random.seed(42)
        scale = self.cfg["global_scale"]
        if "scale_jitter_lo" in self.cfg and "scale_jitter_up" in self.cfg:
            scale_jitter = np.random.uniform(
                self.cfg["scale_jitter_lo"], self.cfg["scale_jitter_up"]
            )
            scale *= scale_jitter
        return scale
