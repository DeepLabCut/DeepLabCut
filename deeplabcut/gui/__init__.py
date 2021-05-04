"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""
import os
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path


DLC_PATH = get_deeplabcut_path()
MEDIA_PATH = os.path.join(DLC_PATH, "gui", "media")
LOGO_PATH = os.path.join(MEDIA_PATH, "logo.png")
