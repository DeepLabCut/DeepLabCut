(how-to-install)=
# How To Install DeepLabCut

- **DeepLabCut can be run on Windows, Linux, or MacOS as long as you have Python 3.10 installed**
  - (see also [technical considerations](tech-considerations-during-install) and if you run into issues also check out the [Installation Tips](https://deeplabcut.github.io/DeepLabCut/docs/recipes/installTips.html) page).
- 🚧 Please note, there are several modes of installation:
  -  please decide to either use a [**conda environment**](https://deeplabcut.github.io/DeepLabCut/docs/installation.html#conda-the-installation-process-is-as-easy-as-this-figure) based installation (**recommended**),
  -  or the supplied [**Docker container**](docker-containers) (recommended for Ubuntu advanced users).
- 🚀 Please note, you will get the best performance with using a **GPU**!
  - Please see the section on [GPU support](https://deeplabcut.github.io/DeepLabCut/docs/installation.html#gpu-support) to install your GPU driver and CUDA.

```{Hint} Familiar with python packages and conda? Quick Install Guide:

This assumes you have `conda`/`mamba` installed and this will install DeepLabCut in a fresh
environment. If you have an NVIDIA GPU, install PyTorch according to [their instructions
](https://pytorch.org/get-started/locally/) (with your desired CUDA version) - you just
need your GPU drivers installed.

```bash
conda create -n DEEPLABCUT python=3.12
conda activate DEEPLABCUT
conda install -c conda-forge pytables==3.8.0

# install PyTorch with your desired CUDA version (or for CPU only) - check [their
](https://pytorch.org/get-started/locally/) website:
# GPU version of pytorch for CUDA 11.3
conda install pytorch cudatoolkit=11.3 -c pytorch


# install the latest version of DeepLabCut
pip install --pre deeplabcut
# or if you want to use the GUI
pip install --pre deeplabcut[gui]

# ONLY IF YOU HAVE A CUDA GPU - check that PyTorch can access your GPU; this
# should print `True`
python -c "import torch; print(torch.cuda.is_available())"
```

- Why do we install [pytables](https://www.pytables.org/usersguide/installation.html) with
`conda` and not `pip`? Because it requires some libraries that not all users will have
installed, and conda will ensure that they are installed as well.

- If you're familiar with the command line and want TensorFlow support, look [below](
deeplabcut-with-tf-install) for a fresh installation that has worked for us (on Linux)
and makes it possible to use the GPU with both PyTorch and TensorFlow.


## CONDA: The installation process is as easy as this figure! -->

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/71e5d954-75a0-4534-9fa6-7ecc4bf1b76d/installDLC.png?format=1500w" width="250" title="DLC" alt="DLC" align="right" vspace = "50">

#### 🚨 Before you start with our conda file, do you have a GPU?
````{admonition} 🚨 Click here for more information!
:class: dropdown
- We recommend having a GPU if possible!
- You **need to decide if you want to use a CPU or GPU for your models**: (Note, you can also use the CPU-only for project management and labeling the data! Then, for example, use Google Colaboratory GPUs for free (read more [here](https://github.com/DeepLabCut/DeepLabCut/tree/master/examples#demo-4-deeplabcut-training-and-analysis-on-google-colaboratory-with-googles-gpus) and there are a lot of helper videos on [our YouTube channel!](https://www.youtube.com/playlist?list=PLjpMSEOb9vRFwwgIkLLN1NmJxFprkO_zi)).

  - **CPU?** Great, jump to the next section below!

  - **NVIDIA GPU?**  If you want to use your own GPU (i.e., a GPU is in your workstation), then you need to be sure you have a CUDA compatible GPU, CUDA, and cuDNN installed. Please note, which CUDA you install depends on what version of PyTorch you want to use. So, please check "GPU Support" below carefully. **Note, DeepLabCut is up to date with the latest CUDA and PyTorch!**
  
  - **Apple M-chip GPU?** Be sure to install miniconda3, and your GPU will be used by default.
````

### Step 1: Install Python via Anaconda

#### Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#), or use miniconda3 for MacOS users (see below)

- Anaconda is an easy way to install Python and additional packages across various operating systems. With Anaconda you create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine.

```{Hint}
Download anaconda for your operating system: [anaconda.com/download/
](https://www.anaconda.com/download/)
```

- IF you use a M1 or M2 chip in your MacBook with v12.5+ (typically 2020 or newer machines), we recommend **miniconda3,** which operates with the same principles as anaconda. This is straight forward and explained in detail here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html. But in short, open the program "terminal" and copy/paste and run the code that is supplied below.

#### 💡 miniconda for Mac
````{admonition} Click the button to see code for miniconda for Mac
:class: dropdown
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_4.12.0-MacOSX-arm64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda init zsh
````

### Step 2: Build an Env using our Conda file!

You simply need to have this `.yaml` file anywhere locally on your computer. So, let's download it!

```{Hint}
Windows users: Be sure you have `git` installed along with anaconda: https://gitforwindows.org/
```

- TO DIRECTLY DOWNLOAD THE CONDA FILE conda:

  - click ➡️ for [CONDA FILE](https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml#:~:text=Raw%20file%20content-,Download,-%E2%8C%98) and then click the "..." and select Download
    <img width="274" alt="Screen Shot 2023-09-13 at 10 33 32 PM" src="https://github.com/DeepLabCut/DeepLabCut/assets/28102185/ec4295a5-e85c-4ce7-8c16-e6517a2cfa22">

-  **Now, in Terminal (or Anaconda Command Prompt for Windows users)**, if you clicked to download, go to your downloads folder.

```{Hint}
Windows users: Be sure to open the program terminal/cmd/anaconda prompt with a RIGHT-click, "open as admin"
```

```{Hint}
:class: dropdown
If you cloned the repo onto your Desktop, the command may look like:
``cd C:\Users\YourUserName\Desktop\DeepLabCut\conda-environments``
You can (on Windows) hold SHIFT and right-click > Copy as path, or (on Mac) right-click and while in the menu press the OPTION key to reveal Copy as Pathname.
```
Be sure you are in the folder that has the `.yaml` file, then run:

``conda env create -f DEEPLABCUT.yaml``


- You can now use this environment from anywhere on your computer (i.e., no need to go back into the conda- folder). Just enter your environment by running:
     - Ubuntu/MacOS: ``source/conda activate nameoftheenv`` (i.e. on your Mac: ``conda activate DEEPLABCUT``)
     - Windows: ``activate nameoftheenv`` (i.e. ``activate DEEPLABCUT``)

Now you should see (`nameofenv`) on the left of your terminal screen, i.e. ``(DEEPLABCUT) YourName-MacBook...``
NOTE: no need to run pip install deeplabcut, as it is already installed!!! :)

(deeplabcut-with-tf-install)=
#### 💡 Notice: PyTorch and TensorFlow Support within DeepLabCut

````{admonition} DeepLabCut TensorFlow Support
:class: dropdown
As of June 2024 we have a PyTorch Engine backend and we will be depreciating the 
TensorFlow backend by the end of 2024. Currently, if you want to use TensorFlow, you 
need to run `pip install deeplabcut[tf]` in order to install the correct version of 
TensorFlow in your conda env. Please note, we will be providing bug fixes, but we will 
not be supporting new TensorFlow versions beyond 2.10 (Windows), and 2.12 for other OS.

Installing TensorFlow and getting it to have access to the GPU can be a bit tricky. 
Check TensorFlow's [compatibility matrix](https://www.tensorflow.org/install/source#gpu)
to know which version of CUDA and cuDNN you should install.

We have found that installing DeepLabCut with the following commands works well for
Linux users to install PyTorch 2.3.1, TensorFlow 2.12, CUDA 11.8 and cuDNN 8 in a Conda
environment:

```bash
conda create -n deeplabcut-with-tf "python=3.10"
conda activate deeplabcut-with-tf

# Install the desired TensorFlow version, built for CUDA 11.8 and cuDNN 8
pip install "tensorflow==2.12" "tensorpack>=0.11" "tf_slim>=1.1.0"

# Install PyTorch with a version using CUDA 11.8 and cuDNN 8
pip install "torch==2.3.1" torchvision --index-url https://download.pytorch.org/whl/cu118

# Create symbolic links to NVIDIA shared libraries for TensorFlow
#   -> as described in their installation docs:
#      https://www.tensorflow.org/install/pip#step-by-step_instructions

pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

pip install  --pre deeplabcut
```
````

**Great, that's it! DeepLabCut is installed!** 🎉💜


### Step 3: Really, that's it! Let's run DeepLabCut

Head over to the [User Guide Overview](https://deeplabcut.github.io/DeepLabCut/docs/UseOverviewGuide.html) for information. 

🎉 Launch DeepLabCut in your new env by running `python -m deeplabcut`

## Other ways to install DeepLabCut and additional tips

### Alternatively, you can git clone this repo and install from source!
i.e., if the download did not work or you just want to have the source code handy!

- **Windows/Linux/MacBooks:** git clone this repo (in the terminal/cmd program, while **in a folder** you wish to place DeepLabCut
To git clone type: ``git clone https://github.com/DeepLabCut/DeepLabCut.git``). Note, this can be anywhere, even downloads is fine.)
- Then follow the same steps as in Step 2 above, adjusting for the file now being in the downloaded folder.

### PIP:

- Everything you need to build custom models within DeepLabCut (i.e., use our source code and our dependencies) can be installed with `pip install 'deeplabcut[gui]'` (for GUI support w/PyTorch) or without the gui: `pip install 'deeplabcut'`.
- If you want to use the SuperAnimal models, then please use `pip install 'deeplabcut[gui,modelzoo]'`. 

## DOCKER:

- We also have docker containers. Docker is the most reproducible way to use and deploy code. Please see our dedicated docker package and page [here](https://deeplabcut.github.io/DeepLabCut/docs/docker.html).

## Pro Tips:

More [installation ProTips](installation-tips) are also available.

If you ever want to update your DLC, just run `pip install --upgrade deeplabcut` once
you are inside your env. If you want to use a specific release, then you need to specify
the version you want, such as `pip install deeplabcut==3.0`. Once installed, you can
check the version by running `import deeplabcut` `deeplabcut.__version__`. Don't be 
afraid to update, DLC is backwards compatible with your 2.0+ projects and performance 
continues to get better and new features are added nearly monthly.

**All of the data you labelled in version 2.X is also compatible with version 3+ and the
PyTorch engine**! There is no change in the workflow or the way labels are handled: the
big changes happen under-the-hood! If you've been working with DeepLabCut 2.X and want
to learn more about moving to the PyTorch engine, checkout our docs on [moving from 
TensorFlow to PyTorch](dlc3-user-guide)

Here are some conda environment management tips: [kapeli.com: Conda Cheat Sheet](
https://kapeli.com/cheat_sheets/Conda.docset/Contents/Resources/Documents/index)

**Pro Tip:** If you want to modify code and then test it, you can use our provided 
testscripts. This would mean you need to be up-to-date with the latest GitHub-based code
though! Please see [here](installation-tips) on how to get the latest GitHub code, and
how to test your installation by following this video: 
https://www.youtube.com/watch?v=IOWtKn3l33s.

## Creating your own customized conda env (recommended route for Linux: Ubuntu, CentOS, Mint, etc.)

*Note in a fresh ubuntu install, you will often have to run: ``sudo apt-get install gcc python3-dev`` to install the GNU Compiler Collection and the python developing environment.

Some users might want to create their own customize env. -  Here is an example.

In the terminal type:

`conda create -n DLC python=3.10`

**Current version:** The only thing you then need to add to the env is deeplabcut (
`pip install deeplabcut`) or `pip install 'deeplabcut[gui]'` which has a napari based
GUI.


## **GPU Support:**

The ONLY thing you need to do **first** if you have an NVIDIA GPU and the matching NVIDIA CUDA+driver installed.
- CUDA: https://developer.nvidia.com/cuda-downloads (just follow the prompts here!)
- DRIVERS: https://www.nvidia.com/Download/index.aspx

### The most common "new user" hurdle is installing and using your GPU, so don't get discouraged!

**CRITICAL:** If you have a GPU, you should FIRST **install an appropriate driver for 
your specific GPU**, then you can use the supplied conda file. You'll need an NVIDIA GPU
which is compatible with CUDA. To see a list of CUDA-enabled NVIDIA GPUs, please [see 
their website](https://developer.nvidia.com/cuda-gpus).

- Here we provide notes on how to install and check your GPU use with TensorFlow (which
is used by DeepLabCut and already installed with the Anaconda files above). Thus, you do
not need to independently install tensorflow.

**FIRST**, install a driver for your GPU. Find DRIVER HERE: 
https://www.nvidia.com/download/index.aspx

- Check which driver is installed by typing this into the terminal: ``nvidia-smi``.

**SECOND**, install CUDA: https://developer.nvidia.com/ (Note that cuDNN, https://developer.nvidia.com/cudnn, is supplied inside the anaconda environment files, so you don't need to install it again).

**THIRD:** Follow the steps above to get the `DEEPLABCUT` conda file and install it!

#### Notes:

- **As of version 3.0+ we moved to PyTorch. The Last supported version of TensorFlow is 
2.10  (window users) and 2.12 for others (we have not tested beyond this).**
- Please be mindful different versions of TensorFlow require different CUDA versions.
- As the combination of TensorFlow and CUDA matters, we strongly encourage you to 
**check your driver/cuDNN/CUDA/TensorFlow versions** [on this StackOverflow post](
https://stackoverflow.com/questions/30820513/what-is-version-of-cuda-for-nvidia-304-125/30820690#30820690
).
- To check your GPU is working, in the terminal, run:

`nvcc -V` to check your installed version(s).

- The best practice is to then run the supplied `testscript_pytorch_single_animal.py` 
(or `testscript.py` for the TensorFlow engine); this is inside the examples folder you
acquired when you git cloned the repo. Here is more information/a short 
[video on running the testscript](https://www.youtube.com/watch?v=IOWtKn3l33s).
- Additionally, if you want to use the bleeding edge, with yout git clone you also get
the latest code. While inside the main DeepLabCut folder, you can run `./reinstall.sh`
to be sure it's installed (more [here](installation-tips))
- You can test that your GPU is being properly engaged with these additional [tips](
https://www.tensorflow.org/programmers_guide/using_gpu).
- Ubuntu users might find this [installation guide](
https://deeplabcut.github.io/DeepLabCut/docs/recipes/installTips.html#installation-on-ubuntu-20-04-lts
) for a fresh ubuntu install useful as well.

## Troubleshooting:

TensorFlow:
Here are some additional resources users have found helpful (posted without endorsement):

- https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e46ca1ae6cfbb5c5d1ee0/1547585235033/cuda_driver.png?format=750w" width="50%">
</p>

- https://www.tensorflow.org/install/source#gpu

- http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html

- https://developer.nvidia.com/cuda-toolkit-archive

- http://www.python36.com/install-tensorflow-gpu-windows/


FFMPEG:

- A few Windows users report needing to install re-install ffmpeg (after windows updates) as described here: https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows (A potential error could occur when making new videos). On Ubuntu, the command is: `sudo apt install ffmpeg`

DEEPLABCUT:

- if you git clone or download this folder, and are inside of it then ``import deeplabcut`` will import the package from there rather than from the latest on PyPi!

(system-wide-considerations-during-install)=
## System-wide considerations:

If you perform the system-wide installation, and the computer has other Python packages or TensorFlow versions installed that conflict, this will overwrite them. If you have a dedicated machine for DeepLabCut, this is fine. If there are other applications that require different versions of libraries, then one would potentially break those applications. The solution to this problem is to create a virtual environment, a self-contained directory that contains a Python installation for a particular version of Python, plus additional packages. One way to manage virtual environments is to use conda environments (for which you need Anaconda installed).

(tech-considerations-during-install)=
## Technical Considerations:

- Computer:

     - For reference, we use e.g. Dell workstations (79xx series) with **Ubuntu 16.04 LTS, 18.04 LTS, 20.04 LTS, 22.04 LTS** and for versions prior to 2.2, we run a Docker container that has TensorFlow, etc. installed (https://github.com/DeepLabCut/Docker4DeepLabCut2.0). Now we use the new Docker containers supplied on this repo (linux support only), also available through [DockerHub](https://hub.docker.com/r/deeplabcut/deeplabcut) or the [`deeplabcut-docker`](https://pypi.org/project/deeplabcut-docker/) helper script.

- Computer Hardware:
     - Ideally, you will use a strong NVIDIA GPU with *at least* 8GB memory.  A GPU is not necessary, but on a CPU the (training and evaluation) code is considerably slower (10x) for ResNets, but MobileNets are faster (see WIKI). You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/DeepLabCut/DeepLabCut/issues/47) or Google Colaboratory.

- Camera Hardware:
     - The software is very robust to track data from any camera (cell phone cameras, grayscale, color; captured under infrared light, different manufacturers, etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/).

- Software:
     - Operating System: Linux (Ubuntu), MacOS* (Mojave), or Windows 10. However, the authors strongly recommend Ubuntu! *MacOS does not support NVIDIA GPUs (easily), so we only suggest this option for CPU use or a case where the user wants to label data, refine data, etc and then push the project to a cloud resource for GPU computing steps, or use MobileNets.
     - Anaconda/Python3: Anaconda: a free and open source distribution of the Python programming language (download from https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2.
     - `pip install deeplabcut`
     - TensorFlow
       - If you want to use a pre3.0 version, you will need [TensorFlow](https://www.tensorflow.org/) (we used version 1.0 in the Nature Neuroscience paper, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.15, and 2.0 to 2.10**; we recommend TF2.10 now) for Python 3.8, 3.9, 3.10 with GPU support.
        - To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow (see: [Mathis & Warren](https://www.biorxiv.org/content/early/2018/10/30/457242)). However, this is the preferred path if you want to test DeepLabCut on your own computer/data before purchasing a GPU, with the added benefit of a straightforward installation! Otherwise, use our COLAB notebooks for GPU access for testing.
     - Docker: We highly recommend advanced users use the supplied [Docker container](docker-containers)
