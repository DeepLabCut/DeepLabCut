"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os
from deeplabcut.utils.auxiliaryfunctions import (
    read_plainconfig,
    get_deeplabcut_path,
)


dlcparent_path = get_deeplabcut_path()
reid_config = os.path.join(dlcparent_path, "reid_cfg.yaml")
cfg = read_plainconfig(reid_config)
