from abc import ABC, abstractmethod
from typing import Optional
from typing import Tuple, Dict

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDataset
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_prediction
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from .utils import *


class Solver(ABC):

    def __init__(self,
                 model: PoseModel,
                 criterion: torch.nn,
                 optimizer: torch.optim.Optimizer,
                 predictor: BasePredictor,
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
        self.predictor = predictor
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

        save_epochs = 30  # TODO: read this value from config file
        for i in range(epochs):
            train_loss = self.epoch(train_loader, mode='train', step=i + 1)
            if self.scheduler:
                self.scheduler.step()
            print(f'Training for epoch {i + 1} done, starting eval on validation data')
            valid_loss = self.epoch(valid_loader, mode='eval', step=i + 1)

            if (i + 1) % save_epochs == 0:
                print(f"Finished epoch {i + 1}; saving model")
                torch.save(
                    self.model.state_dict(),
                    f"{model_folder}/train/snapshot-{i + 1}.pt",
                )

            print(
                f'Epoch {i + 1}/{epochs}, '
                f'train loss {train_loss}, '
                f'valid loss {valid_loss}'
            )
        
        if epochs % save_epochs != 0:
            print(f"Finished epoch {epochs}; saving model")
            torch.save(
                self.model.state_dict(),
                f"{model_folder}/train/snapshot-{epochs}.pt",
            )

    def epoch(self,
              loader: torch.utils.data.DataLoader,
              mode: str = 'train',
              step: Optional[int] = None) -> np.array:
        """

        Parameters
        ----------
        loader: Data loader, which is an iterator over instances.
            Each batch contains image tensor and heat maps tensor input samples.
        mode: "train" or "eval"
        step: the global step in processing, used to log metrics.
        Returns
        -------
        epoch_loss: Average of the loss over the batches.
        """
        if mode not in ['train', 'eval']:
            raise ValueError(f'Solver mode must be train or eval, found mode={mode}.')
        to_mode = getattr(self.model, mode)
        to_mode()
        epoch_loss = []
        metrics={
            'total_loss': [],
            'heatmap_loss': [],
            'locref_loss': [],
        }
        for i, batch in enumerate(loader):
            loss, htmp_loss, locref_loss = self.step(batch, mode)
            epoch_loss.append(loss)

            metrics['total_loss'].append(loss)
            metrics['heatmap_loss'].append(htmp_loss)
            metrics['locref_loss'].append(locref_loss)

            if (i+1)%self.cfg['display_iters'] == 0:
                print(f"Number of iterations : {i+1}, loss : {loss}, lr : {self.optimizer.param_groups[0]['lr']}")
        epoch_loss = np.mean(epoch_loss)
        self.history[f'{mode}_loss'].append(epoch_loss)

        if self.logger:
            for key in metrics.keys():
                self.logger.log(
                    f'{mode} {key}',
                    np.nanmean(metrics[key]),
                    step=step,
                )

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
        image = batch['image']
        image = image.to(self.device)
        prediction = self.model(image)
        
        target = self.model.get_target(batch['annotations'], prediction, image.shape[2:])  # (batch_size, channels, h, w)
        for key in target:
            if target[key] is not None:
                target[key] = torch.Tensor(target[key]).to(self.device)

        total_loss, heatmap_loss, locref_loss = self.criterion(prediction, target)
        if mode == 'train':
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach().cpu().numpy(), heatmap_loss.detach().cpu().numpy(), locref_loss.detach().cpu().numpy(), #rmse #, rmse_pcutoff


class TopDownSolver(Solver):
    # TODO
    pass
