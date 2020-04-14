import numpy as np
import os

os.environ['DLClight'] = 'True'
import pickle
import deeplabcut
import pandas as pd

from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
from easydict import EasyDict as edict
from itertools import product
from tqdm import tqdm

