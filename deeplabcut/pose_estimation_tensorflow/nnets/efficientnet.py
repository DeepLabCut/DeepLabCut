"""
DeepLabCut 2.1.9 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

Effnet added by T. Biasi & AM
Efficient Nets added by T. Biasi & AM
See https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html

"""

import tensorflow as tf
import deeplabcut.pose_estimation_tensorflow.backbones.efficientnet_builder as eff
from .base import BasePoseNet
from .factory import PoseNetFactory


@PoseNetFactory.register("efficientnet")
class PoseEfficientNet(BasePoseNet):
    def __init__(self, cfg):
        super(PoseEfficientNet, self).__init__(cfg)
        if "use_batch_norm" not in self.cfg:
            self.cfg["use_batch_norm"] = False
        if "use_drop_out" not in self.cfg:
            self.cfg["use_drop_out"] = False

    def extract_features(self, inputs, use_batch_norm=False, use_drop_out=False):
        im_centered = self.center_inputs(inputs)
        im_centered /= tf.constant(eff.STDDEV_RGB, shape=[1, 1, 3])
        with tf.compat.v1.variable_scope("efficientnet"):
            eff_net_type = self.cfg["net_type"].replace("_", "-")
            net, end_points = eff.build_model_base(
                im_centered,
                eff_net_type,
                use_batch_norm=use_batch_norm,
                drop_out=use_drop_out,
            )
        return net, end_points

    def get_net(self, inputs, use_batch_norm=False, use_drop_out=False):
        net, _ = self.extract_features(inputs, use_batch_norm, use_drop_out)
        return self.prediction_layers(net)

    def test(self, inputs):
        heads = self.get_net(
            inputs, self.cfg["use_batch_norm"], self.cfg["use_drop_out"]
        )
        return self.add_inference_layers(heads)
