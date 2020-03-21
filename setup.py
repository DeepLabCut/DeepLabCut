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
    version="2.1.6.3",
    author="A. & M. Mathis Labs",
    author_email="alexander.mathis@bethgelab.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexEMG/DeepLabCut",
    install_requires=['certifi','chardet','click','easydict',
                      'h5py~=2.7','intel-openmp','imgaug',
                      'ipython','ipython-genutils',
                      'matplotlib==3.0.3','moviepy','numpy==1.16.4','opencv-python~=3.4',
                      'pandas','patsy','python-dateutil','pyyaml>=5.1','requests',
                      'ruamel.yaml~=0.15','setuptools','scikit-image','scikit-learn',
                      'scipy','six','statsmodels','tables',
                      'tensorpack>=0.9.7.1',
                      'tqdm','wheel'],
    scripts=['deeplabcut/pose_estimation_tensorflow/models/pretrained/download.sh'],
    packages=setuptools.find_packages(),
    data_files=[('deeplabcut',['deeplabcut/pose_cfg.yaml','deeplabcut/pose_estimation_tensorflow/models/pretrained/pretrained_model_urls.yaml','deeplabcut/gui/media/logo.png','deeplabcut/gui/media/dlc_1-01.png'])],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points="""[console_scripts]
            dlc=dlc:main""",
)

#https://www.python.org/dev/peps/pep-0440/#compatible-release
