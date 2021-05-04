#!/bin/sh
#cd to where DLC is installed, and then: dist-packages/deeplabcut/pose_estimation_tensorflow/lib/nms_cython
#i.e. in my case:
cd /usr/local/lib/python3.6/dist-packages/deeplabcut/pose_estimation_tensorflow/lib/nms_cython
python3 setup.py build_ext --inplace
