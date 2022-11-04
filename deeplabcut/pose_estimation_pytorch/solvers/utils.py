import glob
import os
import pandas as pd
from typing import List

import numpy as np

from deeplabcut import auxiliaryfunctions
from ..utils import create_folder


def get_dlc_scorer(train_fraction, shuffle, model_prefix, test_cfg, train_iterations):
    dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.GetScorerName(
        test_cfg,
        shuffle,
        train_fraction,
        train_iterations,
        modelprefix=model_prefix,
    )

    return dlc_scorer, dlc_scorer_legacy


def get_evaluation_folder(train_fraction, shuffle, model_prefix, test_cfg):
    evaluation_folder = os.path.join(
        test_cfg["project_path"],
        str(
            auxiliaryfunctions.GetEvaluationFolder(
                train_fraction,
                shuffle,
                test_cfg,
                modelprefix=model_prefix
            )
        ),
    )
    create_folder(evaluation_folder)
    return evaluation_folder


def get_model_folder(train_fraction, shuffle, model_prefix, test_cfg):
    model_folder = os.path.join(
        test_cfg["project_path"],
        str(
            auxiliaryfunctions.GetModelFolder(
                train_fraction,
                shuffle,
                test_cfg,
                modelprefix=model_prefix
            )
        ),
    )
    create_folder(model_folder)
    return model_folder


def get_result_filename(evaluation_folder,
                        dlc_scorer,
                        dlc_scorerlegacy,
                        model_path):
    _, results_filename, _ = auxiliaryfunctions.CheckifNotEvaluated(evaluation_folder,
                                                                    dlc_scorer,
                                                                    dlc_scorerlegacy,
                                                                    os.path.basename(model_path))

    return results_filename


def get_model_path(model_folder: str,
                   load_epoch: int = -1):
    model_paths = glob.glob(f'{model_folder}/train/snapshot*')
    sorted_paths = sort_paths(model_paths)
    model_path = sorted_paths[load_epoch]
    return model_path


def sort_paths(paths: list):
    sorted_paths = sorted(paths, key=lambda i: int(os.path.basename(i).split('-')[-1][:-3]))
    return sorted_paths


def get_rmse(prediction,
             target: pd.DataFrame,
             pcutoff: int=-1,
             bodyparts: List[str] =None):
    scorer_pred = prediction.columns[0][0]
    scorer_target = target.columns[0][0]
    mask = prediction[scorer_pred].xs("likelihood", level=1, axis=1) >= pcutoff
    if bodyparts:
        diff = (target[scorer_target][bodyparts] - prediction[scorer_pred][bodyparts]) ** 2
    else:
        diff = (target[scorer_target] - prediction[scorer_pred]) ** 2
    mse = diff.xs("x", level=1, axis=1) + diff.xs("y", level=1, axis=1)
    rmse = np.sqrt(mse)
    rmse_p = np.sqrt(mse[mask])

    return rmse, rmse_p
