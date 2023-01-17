from abc import ABC, abstractmethod
from typing import Optional
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDataset
from .utils import *
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_prediction


class Solver(ABC):

    def __init__(self,
                 model: PoseModel,
                 criterion: torch.nn,
                 optimizer: torch.optim.Optimizer,
                 cfg: Dict,
                 device: str = 'cpu',
                 scheduler: Optional = None,
                 logger: Optional = None):

        """Solver base class.

        A solvers contains helper methods for bundling a model, criterion and optimizer.

        Parameters
        ----------
        model: The neural network for solving pose estimation task.
        criterion: The criterion computed from the difference between the prediction
            and the target.
        optimizer: A PyTorch optimizer for updating model parameters.
        cfg: DeepLabCut pose_cfg for training.
            See https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_cfg.yaml for more details.
        scheduler: Optional. Scheduler for adjusting the lr of the optimizer.
        """
        if cfg is None:
            raise ValueError('')
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.history = {'train_loss': [],
                        'eval_loss': []}
        self.logger = logger
        if self.logger:
            logger.log_config(cfg)
        self.model.to(device)
        self.stride = 8  # TODO: stride from config?

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            train_fraction: float = 0.95,
            shuffle: int = 0,
            model_prefix: str = '',
            *,
            epochs: int = 10000) -> None:
        """
        Train model for the specified number of steps.

        Parameters
        ----------
        train_loader: Data loader, which is an iterator over train instances.
            Each batch contains image tensor and heat maps tensor input samples.
        valid_loader: Data loader used for validation of the model.
        train_fraction: TODO discuss (mb better specify with config)
        shuffle: TODO discuss (mb better specify with config)
        model_prefix: TODO discuss (mb better specify with config)
        epochs: The number of training iterations.
        """
        model_folder = get_model_folder(train_fraction,
                                        shuffle,
                                        model_prefix,
                                        train_loader.dataset.cfg)
        for i in tqdm(range(epochs)):
            train_loss = self.epoch(train_loader, mode='train')
            if self.scheduler:
                self.scheduler.step()
            valid_loss = self.epoch(valid_loader, mode='eval')
            save_path = f'{model_folder}/train/snapshot-{i}.pt'
            torch.save(self.model.state_dict(), save_path)
            print(f'Epoch {i + 1}/{epochs}, '
                  f'train loss {train_loss}, '
                  f'valid loss {valid_loss}')

    def get_scores(self,
                   prediction: pd.DataFrame,
                   target: pd.DataFrame,
                   bodyparts: List = None):
        if self.cfg.get( 'pcutoff'):
            pcutoff = self.cfg['pcutoff']
            rmse, rmse_p = get_rmse(prediction, target, pcutoff,
                                    bodyparts = bodyparts)
        else:
            rmse, rmse_p = get_rmse(prediction, target,
                                    bodyparts = bodyparts)

        return np.nanmean(rmse), np.nanmean(rmse_p)

    def epoch(self,
              loader: torch.utils.data.DataLoader,
              mode: str = 'train') -> np.array:
        """

        Parameters
        ----------
        loader: Data loader, which is an iterator over instances.
            Each batch contains image tensor and heat maps tensor input samples.
        mode:
        Returns
        -------
        epoch_loss: Average of the loss over the batches.
        """
        if mode not in ['train', 'eval']:
            raise ValueError(f'Solver must be in train or eval mode, but {mode} was found.')
        to_mode = getattr(self.model, mode)
        to_mode()
        epoch_loss = []
        for batch in loader:
            loss = self.step(batch, mode)
            epoch_loss.append(loss)
        epoch_loss = np.mean(epoch_loss)
        self.history[f'{mode}_loss'].append(epoch_loss)

        return epoch_loss

    @abstractmethod
    def step(self,
             batch: Tuple[torch.Tensor, torch.Tensor],
             *args) -> Optional:
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  dataset: PoseDataset) -> np.array:
        # todo add scale
        predicted_poses = []
        for item in dataset:
            if isinstance(item, tuple) or isinstance(item, list):
                item = item[0]
            else:
                item = item
            item = item.to(self.device)
            output = self.model(item)
            pose = get_prediction(self.cfg, output, self.stride)
            predicted_poses.append(pose)
        predicted_poses = np.concatenate(predicted_poses)
        return predicted_poses

    @staticmethod
    def _get_paths(train_fraction: float = 0.95,
                   shuffle: int = 0,
                   model_prefix: str = "",
                   cfg: dict = None,
                   train_iterations: int = 9):

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

    @staticmethod
    def _get_results_filename(evaluation_folder,
                              dlc_scorer,
                              dlc_scorer_legacy,
                              model_path):

        results_filename = get_result_filename(evaluation_folder,
                                               dlc_scorer,
                                               dlc_scorer_legacy,
                                               model_path)

        return results_filename


class BottomUpSolver(Solver):
    """
    Base solvers for bottom up pose estimation.
    """

    def step(self,
             batch: Tuple[torch.Tensor, torch.Tensor],
             mode: str = 'train') -> np.array:
        """Perform a single epoch gradient update or validation step.

        Parameters
        ----------
        batch: Tuple of input image(s) and target(s) for train or valid single step.
        mode: `train` or `eval`

        Returns
        -------
        batch loss
        """
        if mode not in ['train', 'eval']:
            raise ValueError(f'Solver must be in train or eval mode, but {mode} was found.')
        if mode == 'train':
            self.optimizer.zero_grad()
        image, keypoints = batch
        image = image.to(self.device)
        prediction = self.model(image)
        target = self.model.get_target(keypoints, prediction[0].shape[2:])  # (batch_size, channels, h, w)

        for key in target:
            if target[key] is not None:
                target[key] = target[key].to(self.device)

        total_loss, heatmap_loss, locref_loss = self.criterion(prediction, target)
        if self.logger:
            self.logger.log(f'{mode} total loss', total_loss)
            self.logger.log(f'{mode} heatmap loss', heatmap_loss)
            self.logger.log(f'{mode} locref loss', locref_loss)
        if mode == 'train':
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach().cpu().numpy()


class TopDownSolver(Solver):
    # TODO
    pass
