# How To Install DeepLabCut: <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1609805496320-48N5Y3NEBIVVNUIXPNBV/ke17ZwdGBToddI8pDm48kEPc72vD8ARkQNSjpzTzPRsUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8GRo6ASst2s6pLvNAu_PZdLjGeaj0GkoPWeOP-8DYHB5lK4wgKtPMRocsaeGU4PClrIJgRK3oXroL8Ygt-EThXU/Intall.png?format=750w" width="250" title="DLC" alt="DLC" align="right" vspace = "50">

DeepLabCut can be run on Windows, Linux, or MacOS (see more details at [technical considerations](/docs/installation.md#technical-considerations)). We also recommend watching this short video on [how to navigate our docs](https://www.youtube.com/watch?v=A9qZidI7tL8) and how to [test your installation](https://www.youtube.com/watch?v=IOWtKn3l33s)!

 Please note, there are several modes of installation, and the user should decide to either use a **system-wide** (see [note below](/docs/installation.md#system-wide-considerations)), **Anaconda environment** based installation (recommended), or the supplied **Docker container** (recommended for Ubuntu advanced users). One can of course also use other Python distributions than Anaconda, but **Anaconda is the easiest route.** 
 
 ## Step 1: You need to have Python 3 installed, and we highly recommend using Anaconda to do so. 
 
 ### Simply download the appropriate files here: https://www.anaconda.com/distribution/
 
- Anaconda is perhaps the easiest way to install Python and additional packages across various operating systems. With Anaconda you create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine. 

## Step 2: [Easy install: please use our supplied Anaconda environments](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md).
Please click [here](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md) to get these files (and more info). 

**CRITICAL:** If you have a GPU, you should FIRST then **install the NVIDIA CUDA (10 or LOWER) package and an appropriate driver for your specific GPU**, then you can use the supplied conda file. Please follow the instructions found here https://www.tensorflow.org/install/gpu, and more tips below, to install the correct version of CUDA and your graphic card driver. The order of operations matters.

### The most common "new user" hurdle is installing and using your GPU:
- Here we provide notes on how to install and check your GPU use with TensorFlow (which is used by DeepLabCut and already installed with the Anaconda files above). Thus, you do not need to independently install tensorflow.

In the Nature Neuroscience paper, we used **TensorFlow 1.0 with CUDA (Cuda 8.0)**; in the Nature Protocols paper, we tested up through **TensorFlow 1.12 with CUDA 10**. Some other versions of TensorFlow have been tested (i.e. these versions have been tested 1.2, 1.4, 1.8 and 1.10-1.15, but might require different CUDA versions - CUDA 10.+ is NOT supported)! Currently, TensorFlow 2.0 is not supported. Please check your driver/cuDNN/CUDA/TensorFlow versions [on this StackOverflow post](https://stackoverflow.com/questions/30820513/what-is-version-of-cuda-for-nvidia-304-125/30820690#30820690).

**Currently, TensorFlow 2.X is NOT supported!**

Here is an example on how to install **the GPU driver + CUDA 10 + TensorFlow 1.13.1** will follow here:

**FIRST**, install a driver for your GPU (and compatable up to CUDA 10). Find DRIVER HERE: https://www.nvidia.com/download/index.aspx
- check which driver is installed by typing this into the terminal: ``nvidia-smi``.

**SECOND**, install CUDA 10 (higher versions are not currently supported): https://developer.nvidia.com/ (Note that cuDNN, https://developer.nvidia.com/cudnn, is supplied inside the anaconda environment files, so you don't need to install it again).

**THIRD:** Git Clone this repo to get the conda env. Launch the provided Anaconda Environment File "DLC-GPU" (see [here!](https://github.com/DeepLabCut/DeepLabCut/tree/master/conda-environments)) 
 - note, tensorflow is installed inside this env.

**FOURTH**, Please check your CUDA and [TensorFlow installation](https://www.tensorflow.org/install/) with the lines below:

- In the terminal, run:

  `nvcc -V` to check your installed version(s). 

- The best practice is to then run the supplied `testscript.py` (this is inside the examples folder you acquired when you git cloned the repo). Here is more information/a short [video on running the testscript](https://www.youtube.com/watch?v=IOWtKn3l33s).

- Additionally, if you want to use the bleeding edge, with yout git clone you also get the latest code. While inside the main DeepLabCut folder, you can run `./reinstall.sh` to be sure it's installed (more here: https://github.com/DeepLabCut/DeepLabCut/wiki/How-to-use-the-latest-GitHub-code)

- You can test that your GPU is being properly engaged with these additional [tips](https://www.tensorflow.org/programmers_guide/using_gpu).

- Ubuntu users might find this [installation guide](https://github.com/DeepLabCut/Docker4DeepLabCut2.0/wiki/Installation-of-NVIDIA-driver-and-CUDA-10) for a fresh ubuntu install useful as well. 

## Troubleshooting:

TensorFlow:
Here are some additional resources users have found helpful (posted without endorsement):

- https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e46ca1ae6cfbb5c5d1ee0/1547585235033/cuda_driver.png?format=750w" width="50%">
</p>

- https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e4a53c2241b2d3e3f3c04/1547586322898/tensorflow_cuda_cudnn_version_chart.png?format=500w" width="50%">
</p>

- http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html

- https://developer.nvidia.com/cuda-toolkit-archive

- http://www.python36.com/install-tensorflow-gpu-windows/


FFMEG:

- A few Windows users report needing to install re-install ffmeg (after windows updates) as described here: https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows (A potential error could occur when making new videos). On Ubuntu, the command is: `sudo apt install ffmpeg`

DEEPLABCUT:

- if you git clone or download this folder, and are inside of it then ``import deeplabcut`` will import the package from there rather than from the latest on PyPi!

## You're ready to Run DeepLabCut!

Now you can use Jupyter Notebooks, Spyder, and to train just use the terminal, to run all the code!

## System-wide considerations:

If you perform the system-wide installation, and the computer has other Python packages or TensorFlow versions installed that conflict, this will overwrite them. If you have a dedicated machine for DeepLabCut, this is fine. If there are other applications that require different versions of libraries, then one would potentially break those applications. The solution to this problem is to create a virtual environment, a self-contained directory that contains a Python installation for a particular version of Python, plus additional packages. One way to manage virtual environments is to use conda environments (for which you need Anaconda installed).

## Technical Considerations:

- Computer:

     - For reference, we use e.g. Dell workstations (79xx series) with **Ubuntu 16.04 LTS, 18.04 LTS, or 20.04 LTS** and run a Docker container that has TensorFlow, etc. installed (https://github.com/DeepLabCut/Docker4DeepLabCut2.0).

- Computer Hardware:
     - Ideally, you will use a strong GPU with *at least* 8GB memory such as the [NVIDIA GeForce 1080 Ti or 2080 Ti](https://www.nvidia.com/en-us/shop/geforce/?page=1&limit=9&locale=en-us).  A GPU is not necessary, but on a CPU the (training and evaluation) code is considerably slower (10x) for ResNets, but MobileNets are faster (see WIKI). You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/DeepLabCut/DeepLabCut/issues/47) or Google Colaboratory.

- Camera Hardware:
     - The software is very robust to track data from any camera (cell phone cameras, grayscale, color; captured under infrared light, different manufacturers, etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/).

- Software:
     - Operating System: Linux (Ubuntu), MacOS* (Mojave), or Windows 10. However, the authors strongly recommend Ubuntu! *MacOS does not support NVIDIA GPUs (easily), so we only suggest this option for CPU use or a case where the user wants to label data, refine data, etc and then push the project to a cloud resource for GPU computing steps, or use MobileNets.
     - Anaconda/Python3: Anaconda: a free and open source distribution of the Python programming language (download from https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2.
     - `pip install deeplabcut`
     - TensorFlow
       - You will need [TensorFlow](https://www.tensorflow.org/) (we used version 1.0 in the paper, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.14**) for Python 3 with GPU support.
        - To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow (see: [Mathis & Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). However, this is the preferred path if you want to test DeepLabCut on your own computer/data before purchasing a GPU, with the added benefit of a straightforward installation! Otherwise, use our COLAB notebooks for GPU access for testing.
     - Docker: We highly recommend advaced users use the supplied [Docker container](https://github.com/MMathisLab/Docker4DeepLabCut2.0).
     NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)



Return to [readme](../README.md).
