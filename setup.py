#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabcut",
    version="2.0.9",
    author="A. & M. Mathis Labs",
    author_email="alexander.mathis@bethgelab.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexEMG/DeepLabCut",
    install_requires=['certifi','chardet~=3.0.4','click','easydict~=1.7',
                      'h5py~=2.7','imageio~=2.3.0','intel-openmp',
                      'ipython~=6.0.0','ipython-genutils~=0.2.0',
                      'matplotlib~=3.0.3','moviepy~=0.2.3.5','numpy~=1.14.5','opencv-python~=3.4',
                      'pandas>=0.21.0','patsy','python-dateutil~=2.7.3','pyyaml>=5.1','requests',
                      'ruamel.yaml~=0.15','setuptools','scikit-image~=0.14.0','scikit-learn~=0.19.2',
                      'scipy~=1.1.0','six~=1.11.0','statsmodels~=0.9.0','tables',
                      'tensorpack~=0.9.7.1',
                      'tqdm>4.29','wheel~=0.31.1'],
    scripts=['deeplabcut/pose_estimation_tensorflow/models/pretrained/download.sh'],
    packages=setuptools.find_packages(),
    data_files=[('deeplabcut',['deeplabcut/pose_cfg.yaml','deeplabcut/pose_estimation_tensorflow/models/pretrained/pretrained_model_urls.yaml'])],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points="""[console_scripts]
            dlc=dlc:main""",
)

#https://stackoverflow.com/questions/39590187/in-requirements-txt-what-does-tilde-equals-mean
