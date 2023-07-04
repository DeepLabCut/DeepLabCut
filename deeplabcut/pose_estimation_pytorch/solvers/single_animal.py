import os
import pandas as pd
import torch

from deeplabcut.pose_estimation_pytorch.solvers.base import BottomUpSolver, SOLVERS


@SOLVERS.register_module
class BottomUpSingleAnimalSolver(BottomUpSolver):
    """
    To be extended if needed
    """

    pass
