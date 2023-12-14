"""Creates train/test splits for DeepLabCut Single Animal benchmarks"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import numpy as np

from projects import SA_DLC_BENCHMARKS
from utils import Project


def create_splits(
    seed: int,
    num_samples: int,
    train_fractions: list[float],
    num_splits: int,
) -> dict[float, list[dict[str, list[int]]]]:
    splits = {}
    gen = np.random.default_rng(seed=seed)
    for train_frac in train_fractions:
        splits[train_frac] = []
        print(f"Percentage of samples used for training: {train_frac}")
        for i in range(num_splits):
            num_train_indices = int(np.floor(train_frac * num_samples))
            samples = gen.choice(num_samples, size=num_train_indices, replace=False)
            train_indices = np.sort(samples).tolist()
            test_indices = [i for i in range(num_samples) if i not in train_indices]
            splits[train_frac].append({
                "train": train_indices,
                "test": test_indices,
            })
            print(f"  Split {i}:")
            print(f"    train: {train_indices}")
            print(f"    test:  {test_indices}")
    return splits


def main(
    projects: list[Project],
    seeds: list[int],
    num_splits: int,
    train_fractions: list[float],
    output_file: Path,
) -> None:
    output_file = output_file.resolve()
    splits_data = {}
    for project, seed in zip(projects, seeds):
        save_dir = output_file.parent / f"data-splits-{project.name}"
        save_dir.mkdir(exist_ok=True)

        cfg = auxiliaryfunctions.read_config(str(project.config_path()))

        # saves .h5 and .csv files containing the full dataframe used
        df = deeplabcut.generate_training_dataset.merge_annotateddatasets(cfg, save_dir)
        num_samples = len(df)

        splits_data[project.name] = create_splits(
            seed=seed,
            num_samples=num_samples,
            train_fractions=train_fractions,
            num_splits=num_splits,
        )

    for k, v in splits_data.items():
        print(f"Dataset: {k}")
        for fraction, splits in v.items():
            print(f"  Percentage of samples used for training: {fraction}")
            for i, s in enumerate(splits):
                print(f"    Split {i}:")
                print(f"      train ({len(s['train'])}): {s['train']}")
                print(f"      test  ({len(s['test'])}): {s['test']}")
        print()

    with open(output_file, "w") as f:
        json.dump(splits_data, f, indent=2)


if __name__ == "__main__":
    main(
        projects=[SA_DLC_BENCHMARKS["fly"], SA_DLC_BENCHMARKS["openfield"]],
        seeds=[0, 1],
        num_splits=3,
        train_fractions=[0.8, 0.95],
        output_file=Path("saDLC_benchmarking_splits.json"),
    )
