import os
from deeplabcut.utils.auxiliaryfunctions import (
    read_plainconfig,
    get_deeplabcut_path,
)


dlcparent_path = get_deeplabcut_path()
reid_config = os.path.join(dlcparent_path, "reid_cfg.yaml")
cfg = read_plainconfig(reid_config)
