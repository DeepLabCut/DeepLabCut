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
import importlib
import platform

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    required = fr.read().splitlines()

if platform.machine() == "ppc64le":
    conda_modules = ["cv2", "tensorflow", "imgaug"]
    err = "Missing requirement {}. Please install it using conda"
    for modname in conda_modules:
        try:
            importlib.import_module(modname)
        except ModuleNotFoundError as exc:
            raise ImportError(err.format(modname)) from exc
        except Exception as exc:
            raise exc

setuptools.setup(
    name="deeplabcut",
    version="2.2.0.2",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut",
    install_requires=required,
    extras_require={
        "gui": ["wxpython<4.1"]
    },
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
            ],
        )
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points="""[console_scripts]
            dlc=dlc:main""",
)

# https://www.python.org/dev/peps/pep-0440/#compatible-release
