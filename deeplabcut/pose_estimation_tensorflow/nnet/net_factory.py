"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


def pose_net(cfg):
    net_type = cfg.net_type
    if "mobilenet" in net_type:  # multi currently not supported
        if (
            cfg.get("stride", 8) < 8
        ):  # this supports multianimal (with PAFs) or pairwise prediction
            from deeplabcut.pose_estimation_tensorflow.nnet.pose_netmulti import PoseNet

            cls = PoseNet
        else:
            print("Initializing MobileNet")
            from deeplabcut.pose_estimation_tensorflow.nnet.pose_net_mobilenet import (
                PoseNet,
            )
            cls = PoseNet

    elif "resnet" in net_type:
        if (
            cfg.get("stride", 8) < 8
        ):  # this supports multianimal (with PAFs) or pairwise prediction
            print(
                "Initialing PAFDLC with multiscale deconvolution!", cfg.get("stride", 8)
            )
            from deeplabcut.pose_estimation_tensorflow.nnet.pose_netmulti import PoseNet

            cls = PoseNet
        else:
            print("Initializing ResNet")
            from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import PoseNet

            cls = PoseNet
    elif 'efficientnet' in net_type:
        if (
            cfg.get("stride", 8) < 8
        ):  # this supports multianimal (with PAFs) or pairwise prediction
            from deeplabcut.pose_estimation_tensorflow.nnet.pose_netmulti import PoseNet

            cls = PoseNet
        else:
            print("Initializing Efficientnet")
            from deeplabcut.pose_estimation_tensorflow.nnet.pose_net_efficientnet import PoseNet
            cls = PoseNet
    else:
        raise Exception('Unsupported class of network: "{}"'.format(net_type))

    return cls(cfg)
