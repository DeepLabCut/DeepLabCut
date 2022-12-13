import json
import os
from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path


def parse_available_supermodels():
    dlc_path = get_deeplabcut_path()
    json_path = os.path.join(
        dlc_path, "modelzoo", "models.json"
    )
    with open(json_path) as file:
        return json.load(file)
