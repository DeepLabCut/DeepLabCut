"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os, pickle, yaml
import pandas as pd
from pathlib import Path
import numpy as np

import ruamel.yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def create_config_template_maDLC():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    import ruamel.yaml
    yaml_str = """\
# Project definitions (do not edit)
    Task:
    scorer:
    date:
    \n
# Project path (change when moving around)
    project_path:
    \n
# Multi-Animal Project Area
    multianimalproject:
    individuals:
    multianimalbodyparts:
    uniquebodyparts:
    \n
# Original DLC bodypart naming area
    bodyparts:
    \n
# Annotation data set configuration (and individual video cropping parameters)
    video_sets:
    start:
    stop:
    numframes2pick:
    \n
# Plotting & Skeleton configuration
    skeleton:
    skeleton_color:
    pcutoff:
    dotsize:
    alphavalue:
    colormap:
    \n
# Training,Evaluation and Analysis configuration
    TrainingFraction:
    iteration:
    resnet:
    snapshotindex:
    batch_size:
    \n
# Cropping Parameters (for analysis and outlier frame detection)
    cropping:
def create_config_template_maDLC():
#if cropping is true for analysis, then set the values here:
    x1:
    x2:
    y1:
    y2:
    \n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
    corner2move2:
    move2corner:
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return(cfg_file,ruamelFile)

def write_config_maDLC(configname,cfg):
    """
    Write structured config file.
    """
    with open(configname, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file,ruamelFile = create_config_template_maDLC()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]
