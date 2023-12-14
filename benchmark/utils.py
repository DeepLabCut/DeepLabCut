"""Util methods and classes for DeepLabCut Benchmarking"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import deeplabcut as dlc
import deeplabcut.pose_estimation_pytorch.apis.utils as api_utils
import deeplabcut.pose_estimation_pytorch.runners.utils as runner_utils
import deeplabcut.utils.auxiliaryfunctions as af


@dataclass
class Project:
    """
    Attributes:
        root: the path where the project folder is stored
        name: the name of the project
        iteration: the iteration of the project
    """
    root: Path
    name: str
    iteration: int

    def __post_init__(self) -> None:
        self._cfg = None

    @property
    def cfg(self) -> dict:
        if self._cfg is None:
            self._cfg = dlc.utils.auxiliaryfunctions.read_config(
                self.config_path()
            )
        return self._cfg

    @property
    def date(self) -> str:
        return self.cfg["date"]

    @property
    def path(self) -> Path:
        return self.root / self.name

    @property
    def shuffle_prefix(self) -> str:
        return self.task + self.date

    @property
    def task(self) -> str:
        return self.cfg["Task"]

    def config_path(self) -> str:
        return str(self.root / self.name / "config.yaml")

    def update_iteration_in_config(self) -> None:
        dlc.auxiliaryfunctions.edit_config(
            self.config_path(),
            {"iteration": self.iteration},
        )

    def get_shuffle_folder(self, model_prefix: str | None = None):
        base_dir = self.root / self.name
        if model_prefix is not None:
            base_dir = base_dir / model_prefix

        model_dir = base_dir / "dlc-models" / f"iteration-{self.iteration}"
        return model_dir

    def get_shuffle_path(
        self,
        shuffle_index: int,
        trainset_index: int,
        model_prefix: str | None = None
    ) -> Path:
        base_dir = self.get_shuffle_folder(model_prefix=model_prefix)
        train_fraction = (100 * self.cfg["TrainingFraction"][trainset_index])
        shuffle_name = f"{self.shuffle_prefix}-trainset{train_fraction}shuffle{shuffle_index}"
        return base_dir / shuffle_name


@dataclass
class Shuffle:
    project: Project
    train_fraction: float
    index: int
    model_prefix: str | None = None

    def __post_init__(self):
        self.model_prefix_ = self.model_prefix if self.model_prefix is not None else ""
        self.model_folder = self.project.path / af.get_model_folder(
            self.train_fraction, self.index, self.project.cfg, modelprefix=self.model_prefix_
        )
        self.trainset_folder = af.get_training_set_folder(self.project.cfg)
        self._metadata = None
        self._pytorch_cfg = None

    @property
    def pytorch_cfg(self) -> dict:
        if self._pytorch_cfg is None:
            cfg_path = self.model_folder / "train" / "pytorch_config.yaml"
            self._pytorch_cfg = af.read_plainconfig(str(cfg_path))

        return self._pytorch_cfg

    @property
    def test_indices(self):
        self._lazy_load_metadata()
        return self._metadata[2]

    @property
    def train_indices(self):
        self._lazy_load_metadata()
        return self._metadata[1]

    @property
    def trainset_index(self) -> int:
        return self.project.cfg["TrainingFraction"].index(self.train_fraction)

    def snapshots(self, detector: bool = False) -> list[Path]:
        start_str = "snapshot"
        if detector:
            start_str = "detector-snapshot"

        all_snapshots = [
            p for p in (self.model_folder / "train").iterdir()
            if p.name.startswith(start_str) and p.suffix == ".pt"
        ]
        return sorted(all_snapshots, key=lambda s: int(s.stem.split("-")[-1]))

    def scorer(self, index: int | None = None, epochs: int | None = None) -> str:
        if (index is None and epochs is None) or (index is not None and epochs is not None):
            raise ValueError(f"Exactly one of (index, epochs) must be given: had {index}, {epochs}")

        if index is None:
            index = self.epochs_to_snapshot_index(epochs)

        dlc_scorer, _ = runner_utils.get_dlc_scorer(
            str(self.project.path),
            self.project.cfg,
            self.train_fraction,
            self.index,
            self.model_prefix_,
            index,
        )
        return dlc_scorer

    def ground_truth(self) -> pd.DataFrame:
        path_gt = self.project.path / self.trainset_folder / f"CollectedData_{self.project.cfg['scorer']}.h5"
        df_ground_truth = pd.read_hdf(path_gt)
        if not isinstance(df_ground_truth, pd.DataFrame):
            raise ValueError(f"Ground truth data did not contain a dataframe: {df_ground_truth}")

        return api_utils.ensure_multianimal_df_format(df_ground_truth)

    def predictions(self, index: int | None = None, epochs: int | None = None) -> pd.DataFrame:
        if (index is None and epochs is None) or (index is not None and epochs is not None):
            raise ValueError(f"Exactly one of (index, epochs) must be given: had {index}, {epochs}")

        if index is None:
            index = self.epochs_to_snapshot_index(epochs)

        path_eval = (
            self.project.path /
            "evaluation-results" /
            f"iteration-{self.project.iteration}" /
            self.model_folder.name
        )
        scorer = self.scorer(index=index, epochs=epochs)
        epochs = scorer.split("_")[-1]
        path_predictions = path_eval / f"{scorer}-snapshot-{epochs}.h5"
        df_predictions = pd.read_hdf(path_predictions)
        if not isinstance(df_predictions, pd.DataFrame):
            raise ValueError(f"Predictions data did not contain a dataframe: {df_predictions}")

        return df_predictions

    def epochs_to_snapshot_index(self, epochs: int) -> int:
        paths = self.snapshots()
        snapshot_epochs = [int(s.stem.split("-")[-1]) for s in paths]
        try:
            index = snapshot_epochs.index(epochs)
        except ValueError:
            raise ValueError(
                f"Could not find a snapshot trained for {epochs} epochs in {self}."
                f" Found the following snapshots: {[s.name for s in paths]}"
            )

        return index

    def _lazy_load_metadata(self) -> None:
        if self._metadata is None:
            self._metadata = _get_model_folder(
                project_path=self.project.path,
                project_config=self.project.cfg,
                trainset_folder=self.trainset_folder,
                train_fraction=self.train_fraction,
                shuffle_index=self.index,
            )


def create_shuffles(
    project: Project,
    splits_file: Path,
    trainset_index: int,
    net_type: str,
) -> None:
    """Creates shuffles for a project using predefined train/test splits

    Creates train/test splits according to what is defined in a file (can be created
    with `create_train_test_splits.py`). If there are already shuffles for this
    iteration of the project, the index of the first shuffle created will be 1 more
    than the current max (i.e., if shuffle1 and shuffle2 already exist, the first
    shuffle created will be called ...-shuffle3).

    The splits file must have format:
        {
            "project_name": {
                "train_fraction": [
                    {"train": list[int], "test": list[int]}  # image indices in the train and test set
            }
        }

    Example file:
        {
            "openfield-Pranav-2018-08-20": {
                0.8: [
                    {"train": [0, 1, 3, 4], "test": [2]},  # split 1
                    {"train": [0, 1, 2, 3], "test": [4]},  # split 2
                    {"train": [0, 1, 2, 3], "test": [4]},  # split 3
                ]
            },
            "Fly-Kevin-2019-03-16": {
                0.8: [
                    {"train": [0, 1, 3, 4, 5, 6, 7, 8], "test": [2, 9]},
                    {"train": [0, 1, 2, 3, 6, 7, 8, 9], "test": [4, 5]}
                ]
                0.9: [
                    {"train": [0, 1, 2, 3, 5, 6, 7, 8, 9], "test": [4]},
                ]
            }
        }

    Args:
        project: the project to create shuffles for
        splits_file: the splits containing the train and test indices
        trainset_index: the index of the training fractions to create the shuffles with
        net_type: the type of neural net to create the shuffles with
    """
    shuffle_folder = project.get_shuffle_folder(model_prefix=None)
    shuffle_indices = []
    if shuffle_folder.exists():
        existing_shuffles = [
            p
            for p in project.get_shuffle_folder(model_prefix=None).iterdir()
            if p.is_dir()
        ]
        shuffle_indices = [int(s.name.split("shuffle")[1]) for s in existing_shuffles]

    if len(shuffle_indices) == 0:
        next_index = 1
    else:
        next_index = max(*shuffle_indices) + 1

    train_fraction = project.cfg["TrainingFraction"][trainset_index]
    with open(splits_file, "r") as f:
        raw_data = json.load(f)

    splits = raw_data[project.name][str(train_fraction)]
    train_indices = [s["train"] for s in splits]
    test_indices = [s["test"] for s in splits]
    shuffles_to_create = [i for i in range(next_index, next_index + len(train_indices))]

    print(f"Creating training datasets with indices {shuffles_to_create} and splits:")
    for s in splits:
        print(f"  Train: {s['train']}")
        print(f"  Test:  {s['test']}")

    dlc.create_training_dataset(
        project.config_path(),
        Shuffles=shuffles_to_create,
        trainIndices=train_indices,
        testIndices=test_indices,
        net_type=net_type,
        augmenter_type="imgaug",
    )


def _get_model_folder(
    project_path: Path,
    project_config: dict,
    trainset_folder: str,
    train_fraction: float,
    shuffle_index: int,
) -> tuple[dict, list[int], list[int]]:
    _, metadata_filename = af.get_data_and_metadata_filenames(
        trainset_folder, train_fraction, shuffle_index, project_config,
    )
    metadata = af.load_metadata(str(project_path / metadata_filename))
    return metadata[0], [int(i) for i in metadata[1]], [int(i) for i in metadata[2]]
