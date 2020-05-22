'''
Adopted: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

def pose_net(cfg):
    net_type = cfg.net_type
    if 'mobilenet' in net_type:
        print("Initializing MobileNet")
        from deeplabcut.pose_estimation_tensorflow.nnet.pose_net_mobilenet import PoseNet
        cls = PoseNet
    elif 'resnet' in net_type:
        print("Initializing ResNet")
        from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import PoseNet
        cls = PoseNet
    else:
        raise Exception("Unsupported class of network: \"{}\"".format(net_type))

    return cls(cfg)
