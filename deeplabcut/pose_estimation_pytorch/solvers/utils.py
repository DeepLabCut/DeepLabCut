import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.utils import create_folder


def get_dlc_scorer(train_fraction, shuffle, model_prefix, test_cfg, train_iterations):
    model_folder = get_model_folder(train_fraction, shuffle, model_prefix, test_cfg)
    snapshots = get_snapshots(Path(model_folder))
    snapshot = snapshots[train_iterations]
    snapshot_epochs = int(snapshot.split("-")[-1])

    dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.get_scorer_name(
        test_cfg,
        shuffle,
        train_fraction,
        snapshot_epochs,
        modelprefix=model_prefix,
    )

    return dlc_scorer, dlc_scorer_legacy


def get_evaluation_folder(train_fraction, shuffle, model_prefix, test_cfg):
    evaluation_folder = os.path.join(
        test_cfg["project_path"],
        str(
            auxiliaryfunctions.get_evaluation_folder(
                train_fraction, shuffle, test_cfg, modelprefix=model_prefix
            )
        ),
    )
    create_folder(evaluation_folder)
    return evaluation_folder


def get_model_folder(train_fraction, shuffle, model_prefix, test_cfg):
    model_folder = os.path.join(
        test_cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
                train_fraction, shuffle, test_cfg, modelprefix=model_prefix
            )
        ),
    )
    create_folder(model_folder)
    return model_folder


def get_snapshots(model_folder: Path) -> List[str]:
    snapshots = [
        f.stem
        for f in (model_folder / "train").iterdir()
        if f.name.startswith("snapshot") and f.suffix == ".pt"
    ]
    return sorted(snapshots, key=lambda s: int(s.split("-")[-1]))


def get_result_filename(evaluation_folder, dlc_scorer, dlc_scorerlegacy, model_path):
    _, results_filename, _ = auxiliaryfunctions.check_if_not_evaluated(
        evaluation_folder, dlc_scorer, dlc_scorerlegacy, os.path.basename(model_path)
    )
    return results_filename


def get_model_path(model_folder: str, load_epoch: int):
    model_paths = glob.glob(f"{model_folder}/train/snapshot*")
    sorted_paths = sort_paths(model_paths)
    model_path = sorted_paths[load_epoch]
    return model_path


def get_detector_path(model_folder: str, load_epoch: int):
    detector_paths = glob.glob(f"{model_folder}/train/detector-snapshot*")
    sorted_paths = sort_paths(detector_paths)
    detector_path = sorted_paths[load_epoch]
    return detector_path


def sort_paths(paths: list):
    sorted_paths = sorted(
        paths, key=lambda i: int(os.path.basename(i).split("-")[-1][:-3])
    )
    return sorted_paths


def build_predictions_df(
    dlc_scorer: str,
    individuals: List[str],
    bodyparts: List[str],
    df_index: pd.Index,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Builds a predictions dataframe in the DLC format

    Builds a DataFrame in the DeepLabCut format, with MultiIndex columns. If there is
    only one individual, the column levels are ("scorer", "bodyparts", "coords"). If
    there are multiple individuals, the column levels are ("scorer", "individuals",
    "bodyparts", "coords").

    Args:
        dlc_scorer: the DLC scorer that generated the predictions
        individuals: the names of individuals in the project
        bodyparts: the names of bodyparts in the project
        df_index: the index to apply to the dataframe
        predictions: the predictions made by the scorer. should be of shape
        (len(df_index), len(bodyparts), 3) if len(individuals) == 1, and otherwise
        (len(df_index), len(individuals), len(bodyparts), 3)

    Returns:
        the dataframe containing the predictions in DLC format
    """
    num_individuals = len(individuals)
    if num_individuals == 1:
        # Single animal prediction dataframe
        index = pd.MultiIndex.from_product(
            [
                [dlc_scorer],
                bodyparts,
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "bodyparts", "coords"],
        )
    else:
        # Multi animal prediction dataframe
        index = pd.MultiIndex.from_product(
            [
                [dlc_scorer],
                individuals,
                bodyparts,
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    return pd.DataFrame(predictions, columns=index, index=df_index)


def get_paths(
    train_fraction: float = 0.95,
    shuffle: int = 0,
    model_prefix: str = "",
    cfg: dict = None,
    train_iterations: int = 99,
    method: str = "bu",
):
    dlc_scorer, dlc_scorer_legacy = get_dlc_scorer(
        train_fraction, shuffle, model_prefix, cfg, train_iterations
    )
    evaluation_folder = get_evaluation_folder(
        train_fraction, shuffle, model_prefix, cfg
    )

    model_folder = get_model_folder(train_fraction, shuffle, model_prefix, cfg)

    model_path = get_model_path(model_folder, train_iterations)

    detector_path = None
    if method.lower() == "td":
        detector_path = get_detector_path(model_folder, train_iterations)

    return {
        "dlc_scorer": dlc_scorer,
        "dlc_scorer_legacy": dlc_scorer_legacy,
        "evaluation_folder": evaluation_folder,
        "model_folder": model_folder,
        "model_path": model_path,
        "detector_path": detector_path,
    }


def get_results_filename(evaluation_folder, dlc_scorer, dlc_scorer_legacy, model_path):
    results_filename = get_result_filename(
        evaluation_folder, dlc_scorer, dlc_scorer_legacy, model_path
    )
    return results_filename
