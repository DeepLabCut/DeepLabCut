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
import json
import os


def parse_available_supermodels():
    import deeplabcut

    dlc_path = deeplabcut.utils.auxiliaryfunctions.get_deeplabcut_path()
    json_path = os.path.join(dlc_path, "modelzoo", "models.json")
    with open(json_path) as file:
        return json.load(file)
