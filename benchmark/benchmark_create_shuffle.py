"""Training models on DLC benchmark datasets

In a first step, shuffles can be created for your projects (pass an empty list and no
shuffles are created).

Then you can train models using RunParameters. I usually create the shuffles first,
modify the PyTorch configuration files to add a logger and modify the data augmentation
for whatever I'm doing, and then start my training runs. A logger can be added with:
```
logger:
 type: 'WandbLogger'
 project_name: 'dlc3-ff5f2af-fish'
 run_name: 'dekr-w32-shuffle3'
```

Which specifies to log the run to wandb, (including the project and with which name each
shuffle should be logged).

For single animal projects, benchmark splits were created using the
`create_train_test_splits.py` file. This script creates a JSON file for DLC projects
specifying train/test indices, which can then be passed in the ShuffleCreationParameters
to create new shuffles with the splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import deeplabcut

from projects import (
    MA_DLC_BENCHMARKS,
    MA_DLC_DATA_ROOT,
    SA_DLC_BENCHMARKS,
    SA_DLC_DATA_ROOT,
)
from utils import create_shuffles, Project


@dataclass
class ShuffleCreationParameters:
    """Parameters to create a shuffle

    Attributes:
        project: the project for which to create shuffles
        train_fraction: the training fraction to use to create the shuffles
        net_types: the architectures to create
        num_shuffles: the number of shuffles to create for each net type
        splits_file: if you have specific train/test splits to use for your project,
             they can be used by passing the path to the file containing the splits.
             See the create_train_test_splits.py file for more information about this.
    """

    project: Project
    train_fraction: float
    net_types: tuple[str, ...] | list[str]
    num_shuffles: int = 1
    splits_file: Path | None = None

    def __post_init__(self):
        self.trainset_index = self.project.cfg["TrainingFraction"].index(
            self.train_fraction
        )


def main(shuffles_to_create: list[ShuffleCreationParameters]) -> None:
    """Creates new shuffles for DeepLabCut projects

    Args:
        shuffles_to_create: the shuffles to create
    """
    for m in shuffles_to_create:
        m.project.update_iteration_in_config()
        if m.splits_file is not None:
            for net_type in m.net_types:
                create_shuffles(
                    project=m.project,
                    splits_file=m.splits_file,
                    trainset_index=m.trainset_index,
                    net_type=net_type,
                )
        else:
            deeplabcut.create_training_model_comparison(
                str(m.project.config_path()),
                trainindex=m.trainset_index,
                num_shuffles=m.num_shuffles,
                net_types=list(m.net_types),
            )


if __name__ == "__main__":
    main(
        shuffles_to_create=[
            ShuffleCreationParameters(
                project=MA_DLC_BENCHMARKS["fish"],
                train_fraction=0.95,
                net_types=("top_down_hrnet_w18", "dekr_w32", "dlcrnet_stride32_ms5"),
            ),
            ShuffleCreationParameters(
                project=SA_DLC_BENCHMARKS["fly"],
                train_fraction=0.8,
                net_types=("resnet_50", "hrnet_w18", "hrnet_w32"),
                splits_file=(SA_DLC_DATA_ROOT / "saDLC_benchmarking_splits.json"),
            ),
        ]
    )
