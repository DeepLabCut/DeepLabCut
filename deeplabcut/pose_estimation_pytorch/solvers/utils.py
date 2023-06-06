import glob
import numpy as np
import os
import pandas as pd
from deeplabcut import auxiliaryfunctions
from typing import List, Union

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
                   load_epoch: int):
    model_paths = glob.glob(f'{model_folder}/train/snapshot*')
    sorted_paths = sort_paths(model_paths)
    model_path = sorted_paths[load_epoch]
    return model_path


def sort_paths(paths: list):
    sorted_paths = sorted(paths, key=lambda i: int(os.path.basename(i).split('-')[-1][:-3]))
    return sorted_paths


def save_predictions(names, cfg, data_index,
                     predicted_poses,
                     results_filename):
    if not os.path.exists(names['evaluation_folder']):
        os.makedirs(names['evaluation_folder'])

    results_path = f'{results_filename}'
    num_animals = len(cfg.get('individuals', ['single']))
    if num_animals == 1:
        # Single animal prediction deataframe
        index = pd.MultiIndex.from_product(
            [
                [names['dlc_scorer']],
                cfg["bodyparts"],
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "bodyparts", "coords"],
        )
    else:
        # Multi animal prediction dataframe
        index = pd.MultiIndex.from_product(
            [
                [names['dlc_scorer']],
                cfg['individuals'],
                cfg["multianimalbodyparts"],
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    predicted_data = pd.DataFrame(
        predicted_poses, columns=index, index=data_index
    )

    predicted_data.to_hdf(
        results_path, "df_with_missing"
    )

    return predicted_data


def get_paths(train_fraction: float = 0.95,
              shuffle: int = 0,
              model_prefix: str = "",
              cfg: dict = None,
              train_iterations: int = 99):
    dlc_scorer, dlc_scorer_legacy = get_dlc_scorer(train_fraction,
                                                   shuffle,
                                                   model_prefix,
                                                   cfg,
                                                   train_iterations)
    evaluation_folder = get_evaluation_folder(train_fraction,
                                              shuffle,
                                              model_prefix,
                                              cfg)

    model_folder = get_model_folder(train_fraction,
                                    shuffle,
                                    model_prefix,
                                    cfg)

    model_path = get_model_path(model_folder, train_iterations)

    return {
        'dlc_scorer': dlc_scorer,
        'dlc_scorer_legacy': dlc_scorer_legacy,
        'evaluation_folder': evaluation_folder,
        'model_folder': model_folder,
        'model_path': model_path
    }


def get_results_filename(evaluation_folder,
                         dlc_scorer,
                         dlc_scorer_legacy,
                         model_path):
    results_filename = get_result_filename(evaluation_folder,
                                           dlc_scorer,
                                           dlc_scorer_legacy,
                                           model_path)

    return results_filename
