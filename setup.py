#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0-2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabcut",
    version="2.2rc2",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut",
    install_requires=[
        "python-dateutil",
        "ipython",
        "ipython-genutils",
        "wheel",
        "certifi",
        "chardet",
        "click",
        "cython",
        "filterpy",
        "h5py",
        "ruamel.yaml>=0.15.0",
        "intel-openmp",
        "imgaug",
        "numba==0.51.1",
        "matplotlib==3.1.3",
        "networkx",
        "numpy~=1.17.3",
        "opencv-python-headless",
        "pandas>=1.0.1",
        "patsy",
        "pyyaml",
        "setuptools",
        "scikit-image>=0.17",
        "scikit-learn",
        "scipy>=1.4",
        "six",
        "statsmodels>=0.11",
        "tables",
        "tensorpack",
        "tqdm",
        "moviepy<=1.0.1",
        "Pillow>=7.1",
    ],
    extras_require={"gui": ["wxpython<4.1"]},
    scripts=["deeplabcut/pose_estimation_tensorflow/models/pretrained/download.sh"],
    packages=setuptools.find_packages(),
    data_files=[
        (
            "deeplabcut",
            [
                "deeplabcut/pose_cfg.yaml",
                "deeplabcut/inference_cfg.yaml",
                "deeplabcut/pose_estimation_tensorflow/models/pretrained/pretrained_model_urls.yaml",
                "deeplabcut/gui/media/logo.png",
                "deeplabcut/gui/media/dlc_1-01.png",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/nms_grid.pyx",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/nms_grid.cpp",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/include/nms_scoremap.hxx",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/include/andres/marray.hxx",
            ],
        )
    ],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points="""[console_scripts]
            dlc=dlc:main""",
)

# https://www.python.org/dev/peps/pep-0440/#compatible-release
