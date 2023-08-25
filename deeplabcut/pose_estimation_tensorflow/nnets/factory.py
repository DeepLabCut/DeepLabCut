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
import warnings


class PoseNetFactory:
    _nets = dict()

    @classmethod
    def register(cls, type_):
        def wrapper(net):
            if type_ in cls._nets:
                warnings.warn("Overwriting existing network {}.")
            cls._nets[type_] = net
            return net

        return wrapper

    @classmethod
    def create(cls, cfg):
        if cfg.get("stride", 8) < 8:
            net_type = "multi"
        else:
            net_type = cfg["net_type"]
        key = cls._find_matching_key(cls._nets, net_type)
        if key is None:
            raise ValueError(f"Unsupported network of type {net_type}")
        net = cls._nets.get(key)
        return net(cfg)

    @staticmethod
    def _find_matching_key(dict_, key):
        try:
            match = next(k for k in dict_ if k in key)
        except StopIteration:
            match = None
        return match
