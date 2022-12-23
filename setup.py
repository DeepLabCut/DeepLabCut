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

with open("README.md", encoding="utf-8", errors="replace") as fh:
    long_description = fh.read()


setuptools.setup(
    name="deeplabcut",
    version="2.3.0",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut",
    install_requires=[
        "dlclibrary",
        "filterpy>=1.4.4",
        "ruamel.yaml>=0.15.0",
        "imgaug>=0.4.0",
        "imageio-ffmpeg",
        "numba>=0.54",
        "matplotlib>=3.3",
        "networkx>=2.6",
        "numpy>=1.18.5",
        "pandas>=1.0.1,!=1.5.0",
        "scikit-image>=0.17",
        "scikit-learn>=1.0",
        "scipy>=1.4",
        "statsmodels>=0.11",
        "tables>=3.7.0",
        "torch",
        "tensorpack>=0.11",
        "tf_slim>=1.1.0",
        "tqdm",
        "pyyaml",
        "Pillow>=7.1",
    ],
    extras_require={
        "gui": [
            "pyside6<6.3.2",
            "qdarkstyle==3.1",
            "napari-deeplabcut>=0.0.9",
        ],
        "openvino": ["openvino-dev==2022.1.0"],
        "docs": ["numpydoc"],
        "tf": ["tensorflow>=2.0,<=2.10"],  # Last supported TF version on Windows Native
        "apple_mchips": [],
        "modelzoo": ["huggingface_hub"],
    },
    scripts=["deeplabcut/pose_estimation_tensorflow/models/pretrained/download.sh"],
    packages=setuptools.find_packages(),
    data_files=[
        (
            "deeplabcut",
            [
                "deeplabcut/pose_cfg.yaml",
                "deeplabcut/inference_cfg.yaml",
                "deeplabcut/reid_cfg.yaml",
                "deeplabcut/pose_estimation_tensorflow/models/pretrained/pretrained_model_urls.yaml",
                "deeplabcut/gui/style.qss",
                "deeplabcut/gui/media/logo.png",
                "deeplabcut/gui/media/dlc_1-01.png",
                "deeplabcut/gui/assets/logo.png",
                "deeplabcut/gui/assets/logo_transparent.png",
                "deeplabcut/gui/assets/welcome.png",
                "deeplabcut/gui/assets/icons/help.png",
                "deeplabcut/gui/assets/icons/help2.png",
                "deeplabcut/gui/assets/icons/new_project.png",
                "deeplabcut/gui/assets/icons/new_project2.png",
                "deeplabcut/gui/assets/icons/open.png",
                "deeplabcut/gui/assets/icons/open2.png",
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
