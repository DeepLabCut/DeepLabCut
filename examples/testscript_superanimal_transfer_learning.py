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
import deeplabcut
import os

print(deeplabcut.__file__)
if __name__ == "__main__":

    superanimal_name = "superanimal_topviewmouse"
    basepath = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(basepath, "openfield-Pranav-2018-10-30", "config.yaml")

    deeplabcut.create_training_dataset(config_path, superanimal_name=superanimal_name)

    deeplabcut.train_network(
        config_path,
        maxiters=10,
        superanimal_name=superanimal_name,
        superanimal_transfer_learning=True,
    )
