# DeepLabCut PyTorch API Documentation

<div align="center">

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1628250004229-KVYD7JJVHYEFDJ32L9VJ/DLClogo2021.jpg?format=1000w" width="95%">
</p>

</div>

[![PyPI](https://img.shields.io/pypi/v/deeplabcut?label=PyPI)](https://pypi.org/project/deeplabcut)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deeplabcut)
[![License](https://img.shields.io/github/license/DeepLabCut/DeepLabCut)](https://github.com/DeepLabCut/DeepLabCut/blob/main/LICENSE)

This documentation is designed for maintainers, developers, and expert users who want to understand and extend the PyTorch backend of DeepLabCut 3.0+. It provides detailed information about the architecture, APIs, and practical examples for building and training state-of-the-art pose estimation models.

## Overview

The [`deeplabcut.pose_estimation_pytorch`][] package provides a complete framework for training and deploying deep learning models for pose estimation. The API is designed to be modular, flexible, and extensible, allowing developers to easily add new models, customize training pipelines, and integrate with existing workflows.