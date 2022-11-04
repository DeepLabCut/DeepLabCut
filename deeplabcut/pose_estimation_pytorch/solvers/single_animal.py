import os

import pandas as pd
import torch

from .base import BottomUpSolver
from ..registry import Registry, build_from_cfg
from ...pose_estimation_tensorflow import Plotting
from ...utils import auxiliaryfunctions

SINGLE_ANIMAL_SOLVER = Registry('single_animal_solver',
                                build_func=build_from_cfg)


@SINGLE_ANIMAL_SOLVER.register_module
class BottomUpSingleAnimalSolver(BottomUpSolver):

    def evaluate(self,
                 dataset,
                 model_prefix: str = '',
                 train_fraction: int = 0.95,
                 train_iterations: int = 49,
                 shuffle: int = 0,
                 plotting: bool = False):
        target_df = dataset.dataframe
        self.names = self._get_paths(train_fraction=train_fraction,
                                     model_prefix=model_prefix,
                                     shuffle=shuffle,
                                     cfg=dataset.cfg,
                                     train_iterations=train_iterations)

        results_filename = self._get_results_filename(self.names['evaluation_folder'],
                                                      self.names['dlc_scorer'],
                                                      self.names['dlc_scorer_legacy'],
                                                      self.names['model_path'][:-3])
        self.model.load_state_dict(torch.load(self.names['model_path']))

        predicted_poses = self.inference(dataset)
        predicted_df = self.save_predictions(target_df.index,
                                             predicted_poses.reshape(target_df.index.shape[0], -1),
                                             results_filename)
        if plotting:
            foldername = f'{self.names["evaluation_folder"]}/LabeledImages_{self.names["dlc_scorer"]}-{train_iterations}'
            auxiliaryfunctions.attempttomakefolder(foldername)
            combined_df = predicted_df.merge(target_df,
                                             left_index=True,
                                             right_index=True)
            Plotting(dataset.cfg,
                     dataset.cfg['bodyparts'],
                     self.names['dlc_scorer'],
                     predicted_df.index,
                     combined_df,
                     foldername)

        rmse, rmes_p = self.get_scores(predicted_df, target_df)
        print(f'RMSE: {rmse}, RMSE pcutoff: {rmes_p}')

    def save_predictions(self,
                         data_index,
                         predicted_poses,
                         results_filename):
        if not os.path.exists(self.names['evaluation_folder']):
            os.makedirs(self.names['evaluation_folder'])

        results_path = f'{results_filename}'
        index = pd.MultiIndex.from_product(
            [
                [self.names['dlc_scorer']],
                self.cfg["all_joints_names"],
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "bodyparts", "coords"],
        )

        predicted_data = pd.DataFrame(
            predicted_poses, columns=index, index=data_index
        )

        predicted_data.to_hdf(
            results_path, "df_with_missing", format="table", mode="w"
        )

        return predicted_data
