#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

It should take about 4:00 minutes to run this in a CPU.
It should take about 1:30 minutes on a GPU (incl. downloading the ResNet weights)

It produces nothing of interesting scientifically.
"""

task='TEST' # Enter the name of your experiment Task
scorer='Alex' # Enter the name of the experimenter/labeler

import os

os.environ['DLClight']='True'
import deeplabcut


