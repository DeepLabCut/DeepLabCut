#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import glob
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import deeplabcut.pose_estimation_pytorch.utils as pytorch_utils
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.runners import Task


def verify_paths(
    paths: List[str], pattern: str = r"^(.*)?snapshot-(\d+)\.pt$"
) -> List[str]:
    """Verify the input list of strings if each string follows the regular expression pattern.

    Args:
        paths: List of paths
        pattern: Regular expression pattern for the path

    Returns:
        valid_paths: List of strings from `paths` that follow the given pattern.

    Raises:
        Warning: Thrown if an invalid path is in `paths`. Notifies user of each
                 incorrectly-formatted string found in `paths`.

    Example:
    Inputs:
        paths = ['proj/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-5.pt',
                'proj/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-10.pt',
                'proj/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-1.pt']
        pattern = r"^(.*)?snapshot-(\d+)\.pt$"
    Output:
        valid_paths = ['proj/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-1.pt',
                        'proj/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-5.pt',
                        'proj/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-10.pt']
    """
    valid_paths = [x for x in paths if re.match(pattern, x)]
    invalid_paths = [x for x in paths if x not in valid_paths]

    if len(invalid_paths) > 0:
        warnings.warn("Invalid paths found and ignored:" + "\n".join(invalid_paths))

    return valid_paths


def sort_paths(
    paths: List[str], pattern: str = r"^(.*)?snapshot-(\d+)\.pt$"
) -> List[str]:
    """Sort a list of paths following a specific regular expression pattern.

    Default pattern for each path in list: path/to/snapshot-epoch_number.pt
    Paths not following this format will be ignored, not included in the
    list of paths sorted, and a warning will be issued providing the list of invalid paths

    Args:
        paths: List of string paths (of the snapshots)
        pattern: Regular expression pattern for the file path format of model snapshots

    Returns:
        sorted_paths: List of valid string paths sorted in ascending epoch number order

    Examples:
        1) Input:
                paths = ["/path/to/snapshot-100.pt",
                "/path/to/snapshot-10.pt",
                "/path/to/snapshot-5.pt",
                "/path/to/snapshot-50.pt"]
                pattern = r"^(.*)?snapshot-(\d+)\.pt$"

            Output:
                sorted_paths = ["/path/to/snapshot-5.pt",
                "/path/to/snapshot-10.pt",
                "/path/to/snapshot-50.pt",
                "/path/to/snapshot-100.pt"]

        2)  Input:
                paths = ["path/to/snapshot-5.pt","path/to/snapshot-1.pt"]
                pattern = r"^(.*)?snapshot-(\d+)\.pt$"

            Output:
                sorted_paths = ["path/to/snapshot-1.pt","path/to/snapshot-5.pt"]

        3)  Input:
                paths = ["path/to/snapshots-5.pt","path/to/snapshot-1.pt"]

            Output: sorted_paths = ["path/to/snapshot-1.pt"]
                Warning: "Invalid paths found and ignored: path/to/snapshots-5.pt"

        4)  Input:
                paths = ["path\to\snapshot-5.pt","path\to\snapshot-1.pt"]

            Output:
                sorted_paths = ["path\to\snapshot-1.pt","path\to\snapshot-5.pt"]

        5)  Input:
                paths = ["path/to/snapshot-5.weights","path/to/snapshot-1.pt"]

            Output:
                sorted_paths = ["path/to/snapshot-1.pt"]
                Warning: "Invalid paths found and ignored: path/to/snapshots-5.weights"
    """
    verified_paths = verify_paths(paths, pattern)
    sorted_paths = sorted(
        verified_paths, key=lambda i: int(re.match(pattern, i).group(2))
    )
    return sorted_paths


def get_detector_path(model_folder: str, load_epoch: int) -> str:
    """Given model_folder, load_epoch number, returns the detector path (str).

    Merely calls the verify_directory function with the detector flag

    Args:
        model_folder: String path to the model folder
        load_epoch: snapshot epoch number for the model that you want to use

    Returns:
        Path of the detector directory with the given epoch id

    Example:
        Input:
            model_folder = 'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/'
            load_epoch = 10

        Output:
            'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/detector-snapshot-10.pt'
    """
    return get_verified_path(model_folder, load_epoch, mode="detector")


def get_dlc_scorer(
    project_path: str,
    test_cfg: dict,
    train_fraction: float,
    shuffle: int,
    model_prefix: str,
    train_iterations: int,
) -> Tuple[str, str]:
    """Return dlc_scorer given the ff parameters:
    train_faction, shuffle, model_prefix, test_cfg, and train_iterations.

    Args:
        project_path:
        train_fraction: fraction of the dataset assigned for training
        shuffle: shuffle id
        model_prefix: keep as default (included for backwards compatibility); default value is ""
        test_cfg: contents of the config file in a dict
        train_iterations: the iteration number of the snapshot

    Returns:
        dlc_scorer: the scorer/network name for the particular set of given parameters
        dlc_scorer_legacy: dlc_scorer version that starts with DeepCut instead of DLC

    Example:
        Input:
            train_fraction = 0.95
            shuffle = 1
            model_prefix = ""
            test_cfg = dict from auxiliaryfunctions.read_cfg(configpath)
            train_iterations = 10
        Output:
            ('DLC_model_w32_behaviordateshuffle1_10','DeepCut_model_w32_behaviordateshuffle1_10')

    """
    model_folder = get_model_folder(
        project_path, test_cfg, train_fraction, shuffle, model_prefix
    )
    snapshots = get_snapshots(Path(model_folder))
    snapshot = snapshots[train_iterations]
    snapshot_epochs = int(snapshot.split("-")[-1])

    (dlc_scorer, dlc_scorer_legacy) = auxiliaryfunctions.get_scorer_name(
        test_cfg, shuffle, train_fraction, snapshot_epochs, modelprefix=model_prefix
    )

    return dlc_scorer, dlc_scorer_legacy


def get_snapshots(model_folder: Path) -> List[str]:
    """Get snapshots in a given Path

    Args:
        model_folder: path containing the snapshots

    Returns:
        List of snapshot paths
    """
    snapshots = [
        f.stem
        for f in (model_folder / "train").iterdir()
        if f.name.startswith("snapshot") and f.suffix == ".pt"
    ]
    return sorted(snapshots, key=lambda s: int(s.split("-")[-1]))


def get_verified_path(directory_path: str, load_epoch: int, mode: str = "model") -> str:
    """Helper function for the get_model_path and get_detector_path functions.

    Verifies the directories and returns the specific directory given the parameters:
    directory_path, load_epoch, and mode ("model" for
    model_path and "detector" for detector_path)

    Args:
        directory_path: String path to the model folder
        load_epoch: snapshot epoch number for the model that you want to use
        mode: "model" for loading dlc-models; "detector" for loading detector snapshots

    Returns:
        Path of the directory with the given epoch id and mode (model or detector)

    Raises:
        FileNotFoundError:
            a) when given diirectory does not exist
            b) when the desired snapshot does not exist in the folder
            c) when there are no snapshots in the model_folder
            d) when there are no snapshots following the valid format in the directory

    Example:
        Input:
            model_folder = 'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/'
            load_epoch = 1
            mode = "model"

        Output:
            'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-10.pt'
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Path {directory_path} does not exist.")

    directory_paths = []
    mode_prefix = ""

    # Assigns the proper prefix and paths given the verification mode: for either model paths or detector paths
    if mode == "detector":
        mode_prefix = "detector-"
    directory_paths = glob.glob(
        os.path.join(directory_path, "train", f"{mode_prefix}snapshot*")
    )
    # else:
    #     directory_paths = glob.glob(f"{directory_path}/train/snapshot*")

    # If there are no snapshots inside the given directory, raise a FileNotFoundError
    if len(directory_paths) == 0:
        raise FileNotFoundError(
            f"Path {directory_path} exists, but there are no snapshots in it. "
            "Make sure that the {mode}_folder given has a filetree with files "
            "of the form <{mode}_path>/{mode_prefix}snapshot*."
        )

    if load_epoch >= len(directory_paths):
        raise FileNotFoundError(
            f"Model {directory_path}{mode_prefix}snapshot for the given load_epoch does not exist."
            "Make sure that the {mode}_folder given has a filetree with the correct model."
        )
    sorted_paths = []
    if mode == "detector":
        sorted_paths = sort_paths(
            directory_paths, r"^(.*)?detector-snapshot-(\d+)\.pt$"
        )
    else:
        sorted_paths = sort_paths(directory_paths)

    if len(sorted_paths) == 0:
        raise FileNotFoundError(
            f"Path {directory_path} exists, but the snapshots inside it are all in an invalid format. "
            "Make sure that the snapshots are named in the ff format: "
            "<{mode}_path>/{mode_prefix}snapshot-epoch_no.pt"
        )

    return sorted_paths[load_epoch]


def get_results_filename(
    evaluation_folder: str, dlc_scorer: str, dlc_scorerlegacy: str, model_path: str
) -> str:
    """Returns the file path of the results given by the ff parameters:
    evaluation_folder, dlc_scorer, dlc_scorerlegacy, and model_path.

    Also, checks and informs the user if the network given has already been evaluated.

    Args:
        evaluation_folder: path of the evaluation folder
        dlc_scorer: dlc_scorer name (str)
        dlc_scorerlegacy: dlc_scorerlegacy (str); dlc_scorer name that starts with 'DeepCut' instead of 'DLC'
        model_path: path of the model used

    Returns:
        results_filename: file path (string) of the results

    Example:
        Input:
            evaluation_folder = 0.95
            dlc_scorer = 1
            dlc_scorerlegacy = ""
            model_path = "proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/"
        Output:
            'proj_name/evaluation-results/iteration-0/behaviordate-trainset95shuffle1/DLC_dekr_w32_behaviordateshuffle1_1-snapshot-10.h5'
    """
    (_, results_filename, _) = auxiliaryfunctions.check_if_not_evaluated(
        evaluation_folder, dlc_scorer, dlc_scorerlegacy, os.path.basename(model_path)
    )

    return results_filename


def get_model_folder(
    project_path: str, cfg: dict, train_fraction: float, shuffle: int, model_prefix: str
) -> str:
    """Returns the model folder path given the ff parameters:
    train_faction, shuffle, model_prefix, and test_cfg

    Args:
        project_path:
        cfg: contents of the config file in a dict
        train_fraction: fraction of the dataset assigned for training
        shuffle: shuffle id
        model_prefix: keep as default (included for backwards compatibility); default value is ""

    Returns:
        model_folder: the path of the model folder

    Example:
        Input:
            train_fraction = 0.95
            shuffle = 1
            model_prefix = ""
            test_cfg = dict from auxiliaryfunctions.read_cfg(configpath)
        Output:
            'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1'

    """
    model_folder = os.path.join(
        project_path,
        str(
            auxiliaryfunctions.get_model_folder(
                train_fraction, shuffle, cfg, modelprefix=model_prefix
            )
        ),
    )

    if not os.path.exists(model_folder):
        pytorch_utils.create_folder(model_folder)

    return model_folder


def get_model_path(model_folder: str, load_epoch: int) -> str:
    """Given model_folder and load_epoch number, returns the model path (str).

    Merely calls the verify_directory function

    Args:
        model_folder: String path to the model folder
        load_epoch: snapshot epoch number for the model that you want to use

    Returns:
        Path of the model directory with the given epoch id

    Example:
        Input:
            model_folder = 'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/'
            load_epoch = 10

        Output:
            'proj_name/dlc-models/iteration-0/behaviordate-trainset95shuffle1/train/snapshot-10.pt'
    """
    return get_verified_path(model_folder, load_epoch)


def get_evaluation_folder(
    train_fraction: float, shuffle: int, model_prefix: str, test_cfg: dict
) -> str:
    """Returns the evaluation folder path given the ff parameters:
    train_faction, shuffle, model_prefix, and test_cfg.

    Args:
        train_fraction: fraction of the dataset assigned for training
        shuffle: shuffle id
        model_prefix: keep as default (included for backwards compatibility); default value is ""
        test_cfg: contents of the config file in a dict

    Returns:
        evaluation_folder: the path of the evaluation folder

    Example:
        Input:
            train_fraction = 0.95
            shuffle = 1
            model_prefix = ""
            test_cfg = dict from auxiliaryfunctions.read_cfg(configpath)
        Output:
            'proj_name/evaluation-results/iteration-0/behaviordate-trainset95shuffle1'
    """
    evaluation_folder = os.path.join(
        test_cfg["project_path"],
        str(
            auxiliaryfunctions.get_evaluation_folder(
                train_fraction, shuffle, test_cfg, modelprefix=model_prefix
            )
        ),
    )
    return evaluation_folder


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
            [[dlc_scorer], bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
    else:
        # Multi-animal prediction dataframe
        index = pd.MultiIndex.from_product(
            [[dlc_scorer], individuals, bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    return pd.DataFrame(predictions, columns=index, index=df_index)


def build_entire_pred_df(
    dlc_scorer: str,
    individuals: List[str],
    bodyparts: List[str],
    df_index: pd.Index,
    predictions: np.ndarray,
    unique_bodyparts: List[str],
    unique_predictions: Optional[np.ndarray],
) -> pd.DataFrame:
    num_individuals = len(individuals)
    if num_individuals == 1 or len(unique_bodyparts) == 0 or unique_predictions is None:
        return build_predictions_df(
            dlc_scorer, individuals, bodyparts, df_index, predictions
        )

    animals_df = build_predictions_df(
        dlc_scorer, individuals, bodyparts, df_index, predictions
    )
    unique_df = build_predictions_df(
        dlc_scorer, ["single"], unique_bodyparts, df_index, unique_predictions
    )
    new_cols = pd.MultiIndex.from_tuples(
        [(col[0], "single", col[1], col[2]) for col in unique_df.columns],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    unique_df.columns = new_cols
    predictions_df = animals_df.merge(unique_df, left_index=True, right_index=True)
    return predictions_df


def get_paths(
    project_path: str,
    train_fraction: float = 0.95,
    shuffle: int = 0,
    model_prefix: str = "",
    cfg: dict = None,
    train_iterations: int = 99,
    task: Task = Task.BOTTOM_UP,
):
    dlc_scorer, dlc_scorer_legacy = get_dlc_scorer(
        project_path, cfg, train_fraction, shuffle, model_prefix, train_iterations
    )
    evaluation_folder = get_evaluation_folder(
        train_fraction, shuffle, model_prefix, cfg
    )

    model_folder = get_model_folder(
        project_path, cfg, train_fraction, shuffle, model_prefix
    )

    model_path = get_model_path(model_folder, train_iterations)

    detector_path = None
    if task == Task.TOP_DOWN:  # always take the last detector
        detector_path = get_detector_path(model_folder, -1)

    return {
        "dlc_scorer": dlc_scorer,
        "dlc_scorer_legacy": dlc_scorer_legacy,
        "evaluation_folder": evaluation_folder,
        "model_folder": model_folder,
        "model_path": model_path,
        "detector_path": detector_path,
    }
