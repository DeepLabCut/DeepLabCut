from pathlib import Path

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.modelzoo.utils import (
    create_conversion_table,
    read_conversion_table_from_csv,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.config import (
    create_config_from_modelzoo,
    write_pytorch_config_for_memory_replay,
)
from deeplabcut.utils.pseudo_label import keypoint_matching

dlc_proj_root = Path("/mnt/md0/shaokai/daniel3mouse")
config_path = str(dlc_proj_root / "config.yaml")
superanimal_name = "superanimal_topviewmouse"
model_name = "hrnetw32"
shuffle = 0
max_individuals = 3
device = "cuda"

# keypoint matching before create training dataset
# keypoint matching creates pseudo prediction and a conversion table

keypoint_matching(
    config_path,
    superanimal_name,
    model_name,
)

# keypoint matching creates a memory_replay folder in the root. The conversion table can be read from there
conversion_table_path = dlc_proj_root / "memory_replay" / "conversion_table.csv"

table = create_conversion_table(
    config=config_path,
    super_animal=superanimal_name,
    project_to_super_animal=read_conversion_table_from_csv(conversion_table_path),
)

weight_init = WeightInitialization(
    dataset=superanimal_name,
    conversion_array=table.to_array(),
    with_decoder=True,
    memory_replay=True,
)


deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[shuffle],
    net_type="top_down_hrnet_w32",
    weight_init=weight_init,
    engine=Engine.PYTORCH,
    userfeedback=False,
)

# passing pose_threshold controls the behavior of memory replay. We discard predictions that are lower than the threshold
deeplabcut.train_network(config_path, shuffle=shuffle, device=device, pose_threshold = 0.1)

