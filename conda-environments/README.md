# Quick Anaconda Install!
### Please use one (or more) of the supplied Anaconda environments for a fast and easy installation process.

(0) Be sure you have Anaconda 3 installed! https://www.anaconda.com/distribution/, and get familiar with using "cmd" or terminal!

(1) Either go to www.deeplabcut.org (at the bottom of the page) to download the correct environment file:

<p align="center">
<img src= https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1582763070742-Q45NTO6B5NVXISBQU9TI/ke17ZwdGBToddI8pDm48kCn9JOE-Zo6yZRQwL29ZJRUUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYy7Mythp_T-mtop-vrsUOmeInPi9iDjx9w8K4ZfjXt2dnJa5czOiI-P3uZePqbYB3W1QVt6sqQ11VFBgt-Giz29CjLISwBs8eEdxAxTptZAUg/condaexample.png?format=2500w width="60%">
</p>

or download or git clone this repo (in the terminal/cmd program, while **in a folder** you wish to place DeepLabCut
type ``git clone https://github.com/AlexEMG/DeepLabCut.git``).\
Now, always in Terminal (or Command Prompt), go to the folder named ``conda-environments`` using the command "cd" (which stands for change directory). For example, if you downloaded or cloned the repo onto your Desktop, the command may look like:\
``cd C:\Users\YourUserName\Desktop\DeepLabCut\conda-environments``\
To get the location right, a cool trick is to drag the folder and drop it into Terminal. Alternatively, you can (on Windows) hold SHIFT and right-click > Copy as path, or (on Mac) right-click and while in the menu press the OPTION key to reveal Copy as Pathname.

(2) Now, depending on which file you want to use (if **with GPUs**, see extra note below!!!), open the program **terminal** or cmd where you placed the file (i.e. ``cd conda-environments``) and then type:

``conda env create -f DLC-CPU.yaml``

or

``conda env create -f DLC-GPU.yaml``

(3) You can now use this environment from anywhere on your comptuer (i.e. no need to go back into the conda- folder). Just enter your environment by running:

- Ubuntu/MacOS: ``source/conda activate nameoftheenv`` (i.e. on your Mac: ``conda activate DLC-CPU``)
- Windows: ``activate nameoftheenv`` (i.e. ``activate DLC-GPU``)

Now you should see (nameofenv) on the left of your teminal screen, i.e. ``(dlc-macOS-CPU) YourName-MacBook...``
NOTE: no need to run pip install deeplabcut, as it is already installed!!! :)

However, if you ever want to update your DLC, just run `pip install --upgrade deeplabcut` once you are inside your env. If you want to use a beta release, then you need to specify the specific version you want, such as `pip install deeplabcut==2.2b8`. Once installed, you can check the version by running `import deeplabcut` `deeplabcut.__version__`. Don't be afraid to update, DLC is backwards compatible with your 2.0+ projects and performance continues to get better and new features are added nearly monthly.

Here are some conda environment management tips: https://kapeli.com/cheat_sheets/Conda.docset/Contents/Resources/Documents/index

**Pro Tip:** A great way to test your installation is to use our provided testscripts. This would mean you need to be up-to-date with the lastest code though! Please see [here](https://github.com/DeepLabCut/DeepLabCut/wiki/How-to-use-the-latest-GitHub-code) on how to get the latest GitHub code, and how to test your installation by following this video: https://www.youtube.com/watch?v=IOWtKn3l33s

**Great, that's it!**

Simply run ``ipython`` or ``pythonw`` (**macOS only**) to lauch the terminal, ``jupyter notebook`` to lauch a browser session, or ``ipython/pythonw, import deeplabcut, deeplabcut.launch_dlc()`` to use our Project Manager GUI! **Many more details** [**here**](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/UseOverviewGuide.md)!


# Creating your own customized conda env (recommended route for Linux: Ubuntu, CentOS, Mint, etc.)

Some users might want to create their own customize env. -  Here is an example.

In the terminal type:

`conda create -n DLC python=3.7 tensorflow=1.13.1` 

(this would be for CPU-based tensorflow; for GPU support use `tensorflow-gpu=1.13.1`).

The only thing you then need to add to the env is deeplabcut (`pip install deeplabcut`) and wxPython, which is OS dependent.  
For Windows and MacOS, you just run `pip install -U wxPython<4.1.0` but for linux you need the specific wheel (https://wxpython.org/pages/downloads/index.html).

# Using DLC:

Now just [**follow the user guide**](https://www.nature.com/articles/s41596-019-0176-0) to get DeepLabCut up and running in no time!

Just as a reminder, you can exit the environment anytime and (later) come back! So the environments really allow you to manage multiple packages that you might want to install on your computer.

**GPUs:** The ONLY thing you need to do **first** if you have an NVIDIA GPU, NVIDIA driver installed, and CUDA <=10 (currently, TensorFlow 1.13.1 is installed inside the env, so you can install up to CUDA 10 and an appropriate driver). Please note that only NVIDA GPUs are supported.
- DRIVERS: https://www.nvidia.com/Download/index.aspx
- CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-cuda-enabled-system
