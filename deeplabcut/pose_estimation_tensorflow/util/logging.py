#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Adapted from DeeperCut by Eldar Insafutdinov
# https://github.com/eldar/pose-tensorflow
#
# Licensed under GNU Lesser General Public License v3.0
#
"""
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""
import logging
import os


def setup_logging():
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(
        filename=os.path.join("log.txt"),
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        format=FORMAT,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
