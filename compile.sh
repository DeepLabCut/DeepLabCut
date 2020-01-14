#!/bin/sh
cd deeplabcut/pose_estimation_tensorflow/lib/nms_cython
python3 setup.py build_ext --inplace
