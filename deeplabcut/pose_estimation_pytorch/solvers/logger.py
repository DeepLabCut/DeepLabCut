from typing import Optional

import wandb as wb

from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

LOGGER = Registry('single_animal_solver',
                  build_func=build_from_cfg)

@LOGGER.register_module
class WandbLogger:
    """
    Wandb logger to track experiments and log data.
    (https://docs.wandb.ai/guides)
    """

    def __init__(self,
                 project_name: str = 'deeplabcut',
                 run_name: str = 'tmp',
                 model: PoseModel = None) -> None:
        """
        Initialization of wandb logger.

        Parameters
        ----------
        project_name: the name of the wandb project
        run_name: the name of the wandb run
        model: model to log
        """
        self.run = wb.init(project=project_name,
                           name=run_name)
        if model is None:
            raise ValueError('Specify the model to track!')
        self.run.watch(model)

    def log(self,
            key: str = None,
            value: str = None,
            step: Optional[int] = None) -> None:
        """
        Use this method to log data from runs, such as scalars, images, video, histograms, plots, and tables.

        Parameters
        ----------
        key: name of the logged value
        value: data to log
        step: the global step in processing
        """
        if key is None or value is None:
            raise ValueError(f'Nothing to log. Key: {key} and value: {value} expected to be scalar, table or image.')
        self.run.log({key: value}, step=step)

    def save(self):
        self.run.save(self.run.run.dir)

    def log_config(self,
                   config: dict = None) -> None:
        """
        Use this method to save

        Parameters
        ----------
        config: experiment config
        """
        self.run.config.update(config)
