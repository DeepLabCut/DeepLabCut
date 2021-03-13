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
        net = cls._nets.get(net_type)
        if net is None:
            raise ValueError(f"Unsupported network of type {net_type}")
        return net(cfg)
