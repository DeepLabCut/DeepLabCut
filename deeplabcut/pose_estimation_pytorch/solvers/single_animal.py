import os
import pandas as pd
import torch

from .base import BottomUpSolver
from ..registry import Registry, build_from_cfg
from ..utils import *
from ...pose_estimation_tensorflow import Plotting
from ...utils import auxiliaryfunctions

SINGLE_ANIMAL_SOLVER = Registry('single_animal_solver',
                                build_func=build_from_cfg)


@SINGLE_ANIMAL_SOLVER.register_module
class BottomUpSingleAnimalSolver(BottomUpSolver):
    """
    To be extended if needed
    """
    pass