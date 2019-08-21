"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path
from deeplabcut import DEBUG
import matplotlib.pyplot as plt

def create_new_project_3d(working_directory=None):
    """Creates a new project directory, sub-directories and a basic configuration file for 3d project. 
    The configuration file is loaded with the default values. Adjust the parameters to your project's needs.

    Parameters
    ----------
    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string.

    """
    
