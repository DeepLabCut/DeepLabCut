import json
import os

def parse_available_supermodels():
    import deeplabcut
    dlc_path = deeplabcut.utils.auxiliaryfunctions.get_deeplabcut_path()
    json_path = os.path.join(
        dlc_path, "modelzoo", "models.json"
    )
    with open(json_path) as file:
        return json.load(file)
