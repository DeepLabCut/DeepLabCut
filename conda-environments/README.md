# Quick Anaconda Install for Windows 10, MacOS, Ubuntu 18.04!
### Please use one (or more) of the supplied Anaconda environments for a fast and easy installation process.

(0) Be sure you have Anaconda 3 installed! https://www.anaconda.com/distribution/, and get familiar with using "cmd" or terminal!

(1) Either go to www.deeplabcut.org to download the correct environment file:

<p align="center">
<img src= https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1559946970288-746C71KJ8S0QHRWX6J7K/ke17ZwdGBToddI8pDm48kELMDNUlnb4MkVstKRkN11VZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PIgSbWMtmBz9ZfqPOQ4lPb9Tf93k5QH8_hDHLJHb65L6A/installimage.png?format=1000w width="60%">
</p>

or download or git clone this repo (in the terminal/cmd program, while **in a folder** you wish to place DeepLabCut 
type ``git clone https://github.com/AlexEMG/DeepLabCut.git`` Now, "cd", i.e. go into, the folder named ``conda-environments``

(2) Now, depending on which file you want to use (if **with GPUs**, see extra note below), open the program **terminal** or cmd where you placed the file (i.e. ``cd conda-environments``) and then type: 

``conda env create -f dlc-macOS-CPU.yaml``

or 

``conda env create -f dlc-windowsCPU.yaml``

or 

``conda env create -f dlc-windowsGPU.yaml``

or 

``conda env create -f dlc-ubuntu-GPU.yaml``

(3) Enter your environment by running:

- Ubuntu/MacOS: ``source activate nameoftheenv`` (i.e. ``source activate dlc-macOS-CPU``)
- Windows: ``activate nameoftheenv`` (i.e. ``activate dlc-windowsGPU``)

Now you should see (nameofenv) on the left of your teminal screen, i.e. ``(dlc-macOS-CPU) YourName-MacBook...``

(4) If you plan to use Jupyter Notebooks **once you are inside the environment** you need to run this line one time to link to Jupyter: ``conda install nb_conda``

Great, that's it! 

Now just follow the user guide, to get DeepLabCut up and running in no time!

Just as a reminder, you can exit the environment anytime and (later) come back! So the environments really allow you to manage multiple packages that you might want to install on your computer. 

Here are some conda environment management tips: https://kapeli.com/cheat_sheets/Conda.docset/Contents/Resources/Documents/index

**GPUs:** The ONLY thing you need to do first if have an NVIDIA driver installed, and CUDA (currently, TensorFlow 1.13 is installed inside the env, so you can install CUDA 10 and an appropriate driver).
- DRIVERS: https://www.nvidia.com/Download/index.aspx
- CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-cuda-enabled-system
