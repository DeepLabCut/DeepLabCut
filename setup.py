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
    version="2.0.8",
    author="A. & M. Mathis Labs",
    author_email="alexander.mathis@bethgelab.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexEMG/DeepLabCut",
    install_requires=['certifi','chardet','click','easydict',
                      'h5py','imageio','intel-openmp',
                      'ipython','ipython-genutils',
                      'matplotlib','moviepy','numpy',
                      'pandas','patsy','python-dateutil','pyyaml','requests',
                      'setuptools','scikit-image','scikit-learn','scikit-video',
                      'scipy','six','statsmodels','tables',
                      'tensorpack>=0.9.7.1',
                      'tqdm>4','wheel'],
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
