#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:25:28 2019

@author: alex
"""

import numpy as np

from pathlib import Path
import os, sys

# os.environ['DLClight']='True'
import deeplabcut
import shutil

VIDS = [
    "/home/alex/Dropbox/InterestingCode/social_datasets/4tanmay/silversideschooling-Valentina-2019-07-14/videos/deeplc.menidia.school4.59rpm.S11.D.short.avi"
]
config = "/home/alex/Dropbox/InterestingCode/social_datasets/4tanmay/silversideschooling-Valentina-2019-07-14/config.yaml"
# config='/home/alex/Dropbox/InterestingCode/social_datasets/4tanmay/silversideschooling-Valentina-2019-07-14/config.yaml'
deeplabcut.extract_outlier_frames(
    config, VIDS, shuffle=0, outlieralgorithm="jump", epsilon=5, videotype=".avi"
)
# NOTE: commented out some bodyparts in config.yaml (as I did not do inference for all)
# deeplabcut.refine_labels(config)
