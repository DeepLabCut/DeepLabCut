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

from abc import ABC, abstractmethod

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg
from torch import nn

PREDICTORS = Registry("predictors", build_func=build_from_cfg)


class BasePredictor(ABC, nn.Module):
    """The base Predictor class.

    This class is an abstract base class (ABC) for defining predictors used in the DeepLabCut Toolbox.
    All predictor classes should inherit from this base class and implement the forward method.
    Regresses keypoint coordinates from model's output maps 

    Attributes:
        num_animals: Number of animals in the project. Should be set in subclasses.

    Example:
        # Create a subclass that inherits from BasePredictor and implements the forward method.
        class MyPredictor(BasePredictor):
            def __init__(self, num_animals):
                super().__init__()
                self.num_animals = num_animals

            def forward(self, outputs):
                # Implement the forward pass of your custom predictor here.
                pass
    """

    def __init__(self):
        super().__init__()

        self.num_animals = None

    @abstractmethod
    def forward(self, outputs):
        """Abstract method for the forward pass of the Predictor.

        Args:
            outputs: Output tensors from previous layers.

        Returns:
            Tensor: Output tensor.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        pass
