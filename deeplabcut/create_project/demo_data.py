"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

import os
from pathlib import Path
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

def load_demo_data(config,createtrainingset=True):
  """
  Loads the demo data. Make sure that you are in the same directory where you have downloaded or cloned the deeplabcut.

  Parameter
    ----------
    config : string
        Full path of the config.yaml file of the provided demo dataset as a string.

    Example
    --------
    >>> deeplabcut.load_demo_data('config.yaml')
    --------
  """
  config = Path(config).resolve()
  config = str(config)
  
  transform_data(config)
  if createtrainingset:
      print("Loaded, now creating training data...")
      deeplabcut.create_training_dataset(config, num_shuffles=1)

def transform_data(config):
    """
    This function adds the full path to labeling dataset.
    It also adds the correct path to the video file in the config file.
    """
    import pandas as pd

    cfg = auxiliaryfunctions.read_config(config)
    project_path = str(Path(config).parents[0])

    cfg['project_path'] = project_path
    if 'Reaching' in project_path:
        video_file = os.path.join(project_path, 'videos','reachingvideo1.avi')
    elif 'openfield' in project_path:
        video_file = os.path.join(project_path, 'videos','m4s1.mp4')
    else:
        print("This is not an offical demo dataset.")
    
    if 'WILL BE AUTOMATICALLY UPDATED BY DEMO CODE' in cfg['video_sets'].keys():
        cfg['video_sets'][str(video_file)] = cfg['video_sets'].pop('WILL BE AUTOMATICALLY UPDATED BY DEMO CODE')

    auxiliaryfunctions.write_config(config,cfg)
