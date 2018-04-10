from nnet.pose_net import PoseNet


def pose_net(cfg):
    cls = PoseNet
    return cls(cfg)
