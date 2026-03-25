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

import os

from deeplabcut.utils.auxiliaryfunctions import (
    get_deeplabcut_path,
    read_plainconfig,
)

dlcparent_path = get_deeplabcut_path()
reid_config = os.path.join(dlcparent_path, "reid_cfg.yaml")
cfg = read_plainconfig(reid_config)
