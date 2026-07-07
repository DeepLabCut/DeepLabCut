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
from importlib.resources import files

os.environ["QT_API"] = "pyside6"
import qtpy  # Necessary unused import to properly store the env variable

BASE_DIR = os.path.dirname(__file__)

ASSETS = files("deeplabcut.gui.assets")
LOGO_PATH = ASSETS / "logo.png"
LOGO_TRANSPARENT_PATH = ASSETS / "logo_transparent.png"
WELCOME_PATH = ASSETS / "welcome.png"
