"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import abc


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
