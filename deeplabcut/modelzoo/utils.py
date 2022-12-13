import json
import os
from deeplabcut.utils import auxiliaryfunctions


def parse_available_supermodels():
    dlc_path = auxiliaryfunctions.get_deeplabcut_path()
    json_path = os.path.join(
        dlc_path, "modelzoo", "models.json"
    )
    with open(json_path) as file:
        return json.load(file)
