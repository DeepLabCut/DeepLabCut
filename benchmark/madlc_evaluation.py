from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.benchmark.benchmarks import (
    FishBenchmark,
    MarmosetBenchmark,
    ParentingMouseBenchmark,
    TriMouseBenchmark,
)


def check_bodyparts(gt_bodyparts: set[str], predicted_bodyparts: set[str]) -> None:
    """Needed for the fish: dfin1 and dfin2 are not predicted"""
    valid_bodyparts = set()
    missing_bodyparts = set()
    for bpt in gt_bodyparts:
        if bpt in predicted_bodyparts:
            valid_bodyparts.add(bpt)
        else:
            missing_bodyparts.add(bpt)

    extra_bodyparts = predicted_bodyparts - valid_bodyparts
    if len(extra_bodyparts) > 0:
        print(
            "WARNING: Found bodyparts in predictions that are not in ground truth:"
            f"{list(extra_bodyparts)}"
        )
    if len(missing_bodyparts) > 0:
        print(
            f"WARNING: Some GT bodyparts have no predictions: {list(missing_bodyparts)}"
        )


def parse_dlc_df(
    df: pd.DataFrame,
    ground_truth_bodyparts: set[str],
) -> dict[str, list[dict]]:
    scorers = set(df.columns.get_level_values(0))
    if len(scorers) > 1:
        raise ValueError(
            f"There should only be 1 scorer in the predictions DF. Found {scorers}"
        )
    scorer = scorers.pop()

    individuals = set(df.columns.get_level_values(1))
    bodyparts = set(df.columns.get_level_values(2))
    check_bodyparts(ground_truth_bodyparts, bodyparts)

    data = {}
    for row, df_image in df.iterrows():
        if isinstance(row, str):
            image_path = row
        elif isinstance(row, Iterable):
            image_path = str(Path(*row))
        else:
            raise ValueError(f"Cannot parse row {row}")

        data[image_path] = []
        df_all_individuals = df_image.loc[scorer]
        for idv in individuals:
            df_idv = df_all_individuals.loc[idv]
            keypoints = {}
            scores = []
            for bpt in ground_truth_bodyparts:
                if bpt in df_idv.index.get_level_values(0):
                    df_bpt = df_idv.loc[bpt]
                    scores.append(df_bpt.likelihood)
                    keypoints[bpt] = (df_bpt.x, df_bpt.y)
                else:
                    keypoints[bpt] = (np.nan, np.nan)

            if len(keypoints) > 0:
                data[image_path].append(
                    {
                        "pose": keypoints,
                        "score": np.mean(scores),
                    }
                )

    return data


class DLC3Benchmark:
    """A benchmark for DLC3 Models"""

    def __init__(self, models: dict[str, str]) -> None:
        super().__init__()
        self._names = list(models.keys())
        self.data = {}
        for name, predictions in models.items():
            df_predictions = pd.read_hdf(predictions)
            if not isinstance(df_predictions, pd.DataFrame):
                raise ValueError(
                    f"Failed to parse {predictions} - not a dataframe: {df_predictions}"
                )
            self.data[name] = parse_dlc_df(df_predictions, set(self.keypoints))

    def names(self):
        """An iterable of model names to evaluate."""
        return self._names

    def get_predictions(self, name: str):
        return self.data[name]


class DLC3FishBenchmark(DLC3Benchmark, FishBenchmark):
    code = "link/to/your/code.git"


class DLC3MarmosetBenchmark(DLC3Benchmark, MarmosetBenchmark):
    code = "link/to/your/code.git"


class DLC3ParentingBenchmark(DLC3Benchmark, ParentingMouseBenchmark):
    code = "link/to/your/code.git"


class DLC3TrimouseBenchmark(DLC3Benchmark, TriMouseBenchmark):
    code = "link/to/your/code.git"


def name_to_snapshot_index(filename: str) -> int:
    return int(Path(filename).stem.split("-")[-1])


def main(output_dir: Path, test_hash: str):
    experiments = [p for p in (output_dir / test_hash).iterdir() if p.is_dir()]
    experiments = sorted(experiments, key=lambda s: int(s.stem.split("shuffle")[-1]))
    for exp in experiments:
        benchmark_name = exp.name.split("-")[0]
        benchmark_factory = BENCHMARKS[benchmark_name]

        print(120 * "-")
        print(f"Results for {exp}")
        models = {p.name: p for p in exp.iterdir() if p.suffix == ".h5"}

        b = benchmark_factory(models=models)
        for model in sorted(models.keys(), key=lambda k: name_to_snapshot_index(k)):
            result = b.evaluate(model)
            print(
                f"{result.method_name}, {result.benchmark_name}: "
                f"{result.mean_avg_precision:.4f} mAP, "
                f"{result.root_mean_squared_error:.2f} RMSE"
            )


BENCHMARKS = {
    "fishMay7": DLC3FishBenchmark,
    "pupsMar24": DLC3ParentingBenchmark,
    "trimiceJun22": DLC3TrimouseBenchmark,
    "marmosetMay7": DLC3MarmosetBenchmark,
}


if __name__ == "__main__":
    main(
        output_dir=Path("outputs"),
        test_hash="2023_12_07_fc2f00e2",
    )
