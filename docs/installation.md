# How To Install DeepLabCut2.0:

## First, Technical Considerations:

- Computer: 

     - For reference, we use e.g. Dell workstations (79xx series) with **Ubuntu 16.04 LTS** and run a Docker container that has TensorFlow, etc. installed (https://github.com/MMathisLab/Docker4DeepLabCut2.0). 

- Computer Hardware:
     - Ideally, you will use a strong GPU with at least 8GB memory such as the [NVIDIA GeForce 1080 Ti](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080/).  A GPU is not necessary, but on a CPU the (training and evaluation) code is considerably slower (100x). You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/AlexEMG/DeepLabCut/issues/47).

- Camera Hardware:
     - The software is very robust to track data from pretty much any camera (grayscale, color, or graysale captured under infrared light etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/)

     
- Software: 
     - Operating System: Linux (Ubuntu) or Windows (7 or newer). However, the authors recommend Ubuntu
     - Anaconda/Python3: Anaconda: a free and open source distribution of the Python programming language (download from: \https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2. 
     - pip install deeplabcut (see below!) 
     - TensorFlow (see below!)
       - You will need [TensorFlow](https://www.tensorflow.org/) (we used version 1.0 in the paper, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.4, 1.8, and  1.10**) for Python 3 with GPU support. 
        - To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow (see: [Mathis & Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). However, this is the preferred path if you want to test DeepLabCut on your data before purchasing a GPU, with the added benefit of a straightforward installation! 
     - Docker: We highly recommend using the supplied [Docker container](https://github.com/MMathisLab/Docker4DeepLabCut2.0). 
     NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)
     
    
# INSTALLATION:
 
 There are several modes of installation, and the user should decide to either use a **system-wide** (see [note below](/docs/installation.md#system-wide-considerations)) or **Anaconda environment** based pip installation (recommended), or the supplied **Docker container** (recommended for more advanced users). The simplest installation is the pip installation, which is  described first.
 
 **All the following commands will be run in the app ``terminal`` in Ubuntu, and called ``cmd`` in Windows. Please first open the terminal (search ``terminal`` or ``cmd``).**
 
## Anaconda:  
[Anaconda](https://anaconda.org/anaconda/python) is perhaps the easiest way to install Python and additional packages across various operating systems. First create an [Anaconda](https://anaconda.org/anaconda/python) environment.  With Anaconda you create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine by running in a terminal:

**LINUX:**
```
conda create -n <nameyourenvironment> python=3.6
source activate <nameyourenvironment>
```
**Windows:** 
```
conda create -n <nameyourenvironment> python=3.6
activate <nameyourenvironment>
```
Once the environment was activated, the user can install DeepLabCut. In the terminal type: 
```
pip install deeplabcut 
```
 * if you have ever used pip to install deeplabcut (or other packges), use ``--ignore-installed`` to be sure you are grabbing the latest! i.e. ``pip install --ignore-installed deeplabcut``

**Then,**

Windows: ```pip install -U wxPython ``` 

Linux: ```pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl```

## Install TensorFlow - with GPU support or CPU support:
As users can use a GPU or CPU, TensorFlow is not installed with the command ``pip install deeplabcut``. 
Here is more information on how to best install TensorFlow with pip: https://www.tensorflow.org/install/pip

CPU ONLY: 

``pip install --ignore-installed tensorflow==1.10``

GPU: 

Install [TensorFlow](https://www.tensorflow.org/). In the Nature Neuroscience paper we used **TensorFlow 1.0 with CUDA (Cuda 8.0)**. Some other versions of TensorFlow have been tested, but use at your own risk (i.e. these versions have been tested 1.2, 1.4, 1.8 or 1.10-1.11, but might require different CUDA versions)! Please check your driver/CUDA/TensorFlow version [on this Stackoverflow post](https://stackoverflow.com/questions/30820513/what-is-version-of-cuda-for-nvidia-304-125/30820690#30820690).

If you have a GPU, you should then **install the NVIDIA CUDA package and an appropriate driver for your specific GPU.** Please follow the instructions found here: https://www.tensorflow.org/install/gpu, and more [tips below](). The order of operations matters. 

Some tips for installing **TensorFlow 1.8** will follow here:

**FIRST**, install a driver for your GPU (we recommend the 384.xx) Find DRIVER HERE: https://www.nvidia.com/download/index.aspx
- check which driver is installed by typing this into the terminal: ``nvidia-smi``

**SECOND**, install CUDA (9.0 here): https://developer.nvidia.com/cuda-90-download-archive

**THIRD**, install TensorFlow: 

Package for pip install:

``pip install tensorflow-gpu==1.8`` —with GPU support (Ubuntu and Windows)

Note, the version is specified by using: ``==1.8``

**FOURTH**, Please check your CUDA and [TensorFlow installation](https://www.tensorflow.org/install/) with the lines below:

Start a python session: 
`` ipython``

``import tensorflow as tf``

``sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))``

You can test that your GPU is being properly engaged with these additional [tips](https://www.tensorflow.org/programmers_guide/using_gpu).

# Troubleshooting: 

TensorFlow:
Here are some additional resources users have found helpful (posted without endorsement):

- https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690

<p align="center">
<img src="/docs/images/cuda_driver.png" width="50%">
</p>

- https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

<p align="center">
<img src="/docs/images/tensorflow_cuda_cudnn_version_chart.png" width="50%">
</p>

- http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html

- https://developer.nvidia.com/cuda-toolkit-archive

- http://www.python36.com/install-tensorflow-gpu-windows/


FFMEG:

- A few Windows users report needing to install ffmeg as described here: https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows (A potential error could occur when making new videos). 

DEEPLABCUT: 

- if you git clone or download this folder, and are inside of it then ``import deeplabcut`` will import the package from there rather than from the latest on PyPi!

## System-wide considerations:

If you perform the system wide installation, and the computer has other Python packages or TensorFlow versions installed that conflict, this will overwrite them. If you have a dedicated machine for DeepLabCut, this is fine. If there are other applications that require different versions of libraries, then one would potentially break those applications. The solution to this problem is to create a virtual environment, a self-contained directory that contains a Python installation for a particular version of Python, plus additional packages. One way to manage virtual environments is to use conda environments (for which you need Anaconda installed). 

## You're ready to Run DeepLabCut! 

Now you can use Jupyer Notebooks, Spyder, and to train just use the terminal, to run all the code!
          
 Return to [readme](../README.md).

