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
"""
Test script for super animal adaptation
"""
import os

import deeplabcut
from deeplabcut.modelzoo.weight_initialization import build_weight_init

print(deeplabcut.__file__)
if __name__ == "__main__":

    superanimal_name = "superanimal_topviewmouse"
    basepath = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(basepath, "openfield-Pranav-2018-10-30", "config.yaml")
    model_name = "hrnet_w32"
    detector_name = "fasterrcnn_resnet50_fpn_v2"

    weight_init = build_weight_init(
        cfg=config_path,
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        with_decoder=False,
    )
    deeplabcut.create_training_dataset(config_path, weight_init=weight_init)

    deeplabcut.train_network(
        config_path,
        epochs=1,
        superanimal_name=superanimal_name,
        superanimal_transfer_learning=True,
    )
