# Technical (Hardware) Considerations

## Quick summary:
[On our install page](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/installation.md#technical-considerations)
we highlight that for GPU computing through standard installation you need a NVIDIA GPU, with at least 8 GB of memory. If you have an Intel or AMD GPU, and are on windows, there is an alternative method of installation available which is shown on the [installation tips page](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/recipes/installTips.md) under "How to install Deeplabcut for Intel and AMD GPUs". 
Note, some info is repeated here, and will be updated as systems and hardware changes.

### Computer:

For reference, we use e.g. Dell workstations (79xx series) with **Ubuntu 16.04 LTS, 18.04 LTS, or 20.04 LTS** and run a Docker container that has TensorFlow, etc. installed (https://github.com/DeepLabCut/Docker4DeepLabCut2.0).

### Computer Hardware:

Ideally, you will use a strong GPU with *at least* 8GB memory such as the [NVIDIA GeForce 1080 Ti,  2080 Ti, or 3090](https://www.nvidia.com/en-us/shop/geforce/?page=1&limit=9&locale=en-us).  A GPU is not strictly necessary, but on a CPU the (training and evaluation) code is considerably slower (10x) for ResNets, but MobileNets and EfficientNets are slightly faster. Still, a GPU will give you a massive speed boost. You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/DeepLabCut/DeepLabCut/issues/47) or Google Colaboratory.

### Camera Hardware:

The software is very robust to track data from any camera (cell phone cameras, grayscale, color; captured under infrared light, different manufacturers, etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/).

### Software:

**Operating System:** Linux (Ubuntu), MacOS* (Mojave), or Windows 10. However, the authors strongly recommend Ubuntu! *MacOS does not support NVIDIA GPUs (easily), so we only suggest this option for CPU use or a case where the user wants to label data, refine data, etc and then push the project to a cloud resource for GPU computing steps, or use MobileNets.

**Anaconda/Python3:** Anaconda: a free and open source distribution of the Python programming language (download from https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2.


**TensorFlow** You will need [TensorFlow](https://www.tensorflow.org/) (we used version 1.0 in the paper, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.15, and 2.0 to 2.5**; we recommend TF2.5 now) for Python 3.7, 3.8, or 3.9 with GPU support.

To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow (see: [Mathis & Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). However, this is the preferred path if you want to test DeepLabCut on your own computer/data before purchasing a GPU, with the added benefit of a straightforward installation! Otherwise, use our COLAB notebooks for GPU access for testing.

Docker: We highly recommend advaced users use the supplied [Docker container](https://github.com/MMathisLab/Docker4DeepLabCut2.0).
NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)
