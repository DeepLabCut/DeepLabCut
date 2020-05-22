"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os, pickle
import pandas as pd
import numpy as np
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config
from pathlib import Path

def extractindividualsandbodyparts(cfg):
    individuals=cfg['individuals']
    if len(cfg['uniquebodyparts'])>0:
        individuals.extend('single')
    return individuals,cfg['uniquebodyparts'],cfg['multianimalbodyparts']
