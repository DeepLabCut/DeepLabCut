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

from typing import Optional

import deeplabcut.pose_estimation_pytorch.registry as deeplabcut_pose_estimation_pytorch_registry
import wandb as wb
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel

LOGGER = deeplabcut_pose_estimation_pytorch_registry.Registry(
    "single_animal_solver",
    build_func=deeplabcut_pose_estimation_pytorch_registry.build_from_cfg,
)


@LOGGER.register_module
class WandbLogger:
    """Wandb logger to track experiments and log data.

    Refer to: https://docs.wandb.ai/guides for more information on wandb.

    Attributes:
        run (wandb.Run): The wandb run object associated with the current experiment.

    """

    def __init__(
        self,
        project_name: str = "deeplabcut",
        run_name: str = "tmp",
        model: PoseModel = None,
    ) -> None:
        """Initialize the WandbLogger class.

        Args:
            project_name: The name of the wandb project. Defaults to "deeplabcut".
            run_name: The name of the wandb run. Defaults to "tmp".
            model: The model to log. Defaults to None.

        Example:
            logger = WandbLogger(project_name="my_project", run_name="exp1", model=my_model)

        """

        self.run = wb.init(project=project_name, name=run_name)
        if model is None:
            raise ValueError("Specify the model to track!")
        self.run.watch(model)

    def log(
        self, key: str = None, value: str = None, step: Optional[int] = None
    ) -> None:
        """Logs data from runs, such as scalars, images, video, histograms, plots, and tables.

        Args:
            key The name of the logged value.
            value: Data to log.
            step: The global step in processing. Defaults to None.

        Example:
            logger = WandbLogger()
            logger.log(key="loss", value=0.123, step=100)

        """
        if key is None or value is None:
            raise ValueError(
                f"Nothing to log. Key: {key} and value: {value} expected to be scalar, table or image."
            )
        self.run.log({key: value}, step=step)

    def save(self):
        """Syncs all files to wandb with the policy specified.

        Notes:
            self.run: A run is a unit of computation logged by wandb.
            self.run.run.dir: The directory where files associated with the run are saved.

        Example:
            logger = WandbLogger()
            # Training and logging
            logger.save()
        """
        self.run.save(self.run.dir)

    def log_config(self, config: dict = None) -> None:
        """Updates the current run with the given config dict.

        Notes:
            self.run: A run is a unit of computation logged by wandb.
            self.run.config: Config object associated with this run.

        Args:
            config: Experiment config file.

        Example:
            logger = WandbLogger()
            config = {"learning_rate": 0.001, "batch_size": 32}
            logger.log_config(config)

        """
        self.run.config.update(config)
