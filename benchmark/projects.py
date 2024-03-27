"""DeepLabCut projects to benchmark"""

from __future__ import annotations

from pathlib import Path

from utils import Project

#MA_DLC_DATA_ROOT = Path("/home/niels/datasets/ma_dlc")
MA_DLC_DATA_ROOT = Path("/home/lucas/datasets")
SA_DLC_DATA_ROOT = Path("/home/lucas/datasets/single_animal_dlc")

MA_DLC_BENCHMARKS = {
    "trimouse": Project(
        root=MA_DLC_DATA_ROOT,
        name="trimice-dlc-2021-06-22",
        iteration=1,
    ),
    "fish": Project(
        root=MA_DLC_DATA_ROOT,
        name="fish-dlc-2021-05-07",
        iteration=30,
    ),
    "parenting": Project(
        root=MA_DLC_DATA_ROOT,
        name="pups-dlc-2021-03-24",
        iteration=1,
    ),
    "marmoset": Project(
        root=MA_DLC_DATA_ROOT,
        name="marmoset-dlc-2021-05-07",
        iteration=1,
    ),
}

SA_DLC_BENCHMARKS = {
    "fly": Project(
        root=SA_DLC_DATA_ROOT,
        name="Fly-Kevin-2019-03-16",
        iteration=2,
    ),
    "openfield": Project(
        root=SA_DLC_DATA_ROOT,
        name="openfield-Pranav-2018-08-20",
        iteration=2,
    ),
}
