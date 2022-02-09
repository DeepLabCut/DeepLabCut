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
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
root = Path(__file__).parent
filename = str(root / "requirements.txt")
with open(filename) as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setuptools.setup(
    name="deeplabcut",
    version="2.2.0.6",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut",
    install_requires=requirements,
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
