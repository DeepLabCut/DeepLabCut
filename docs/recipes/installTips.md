(installation-tips)=
# Installation Tips

## How to use the latest updates directly from GitHub

We often update the master deeplabcut code base on GitHub, and then ~1 a month we push out a stable release on pypi. This is what most users turn to on a daily basis (i.e. pypi is where you get your `pip install deeplabcut` code from! But, sometimes we add things to the repo that are not yet integrated, or you might want to edit the code yourself. Here, we show you how to do this.

### Method 1:

If you want to *use* the latest, you can use pip and add the specific tags, such as `gui`, etc. by modifying and running: 
```
pip install --upgrade 'git+https://github.com/deeplabcut/deeplabcut.git#egg=deeplabcut[gui]'
```

which will download and update deeplabcut, and any dependencies that don't match the new version. If you want to force upgrade all of the dependencies to the latest available versions, too, then use the additional `--upgrade-strategy eager`, i.e.:

```
pip install --upgrade --upgrade-strategy eager 'git+https://github.com/deeplabcut/deeplabcut.git#egg=deeplabcut[gui]'
```

### Method 2: 

If you want to be able to *edit* the source code of DeepLabCut, i.e., maybe add a feature or fix a 🐛, then you need to "clone" the source code:

**Step 1:**

- git clone the repo into a folder on your computer:  

- click on this green button and copy the link:

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1581984907363-G8AFGX4V20Y1XD1PSZAK/ke17ZwdGBToddI8pDm48kGJBV0_F4LE4_UtCip_K_3lZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVE0ejQCe16973Pm-pux3j5_Oqt57D2H0YbaJ3tl8vn_eR926scO3xePJoa6uVJa9B4/gitclone.png?format=500w)

- then in the terminal type: `git clone https://github.com/DeepLabCut/DeepLabCut.git`

**Step 2:**

- Now you will work from the terminal inside this cloned folder:

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1581985288123-V8XUAY0C0ZDNJ5WBHB7Y/ke17ZwdGBToddI8pDm48kIsGBOdR9tS_SxF6KQXIcDtZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpz3c8X74DzCy4P3pv-ZANOdh-3ZL9iVkcryTbbTskaGvEc42UcRKU-PHxLXKM6ZekE/terminal.png?format=750w)

- Now, when you start `ipython` and `import deeplabcut` you are importing the folder "deeplabcut" - so any changes you make, or any changes we made before adding it to the pip package, are here.

- You can also check which deeplabcut you are importing by running: `deeplabcut.__file__`

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1581985466026-94OCSZJ5TL8U52JLB5VU/ke17ZwdGBToddI8pDm48kNdOD5iqmBzHwUaWGKS6qHBZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpyQPoegsR7K4odW9xcCi1MIHmvHh95_BFXYdKinJaRhV61R4G3qaUq94yWmtQgdj1A/importlocal.png?format=750w)

If you make changes to the code/first use the code, be sure you run `./resinstall.sh`, which you find in the main DeepLabCut folder:

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1609353210708-FRNREI7HUNS4GLDSJ00G/ke17ZwdGBToddI8pDm48kAya1IcSd32bok4WHvykeicUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYy7Mythp_T-mtop-vrsUOmeInPi9iDjx9w8K4ZfjXt2dq18t0tDkB2HMfL2JGcLHN27k5rSOPIU8nEAZT0p1MiSCjLISwBs8eEdxAxTptZAUg/Screen+Shot+2020-12-30+at+7.33.16+PM.png?format=2500w)

Then, you can see what version you have with `deeplabcut.__version__`

If you make changes, you can also then utilize our test scripts. Run the desired test script found here (you will need to git clone first): https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/.

i.e., for example:
```
# Testing with the PyTorch engine
python testscript_pytorch_multi_animal.py

# Testing with the TensorFlow engine
python testscript_multianimal.py
```


## Installation on Ubuntu 18.04 LTS

### Here are our tips for an easy installation. This is done on a fresh computer installation (Ubuntu 18.04 LTS)

install gcc:

```
sudo apt install gcc
```

then, download CUDA 10 from here: https://developer.nvidia.com/cuda-downloads and follow the instructions... ie:

```
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run
```
 with the exception that I also (afterwards):

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers autoinstall
```

**then reboot**

Check gcc -version:

```
gcc --version
```

output:
```
gcc (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE
```
Then:
```
sudo apt install nvidia-cuda-toolkit gcc-7
```

```
nvcc --version
```

then you can check:

```
nvidia-smi
```

and you'll see driver version, CUDA version, status of graphics card(s).

## Installation on Ubuntu 20.04 LTS

Hello! Another cookbook entry on how to install your freshly installed 20.04 LTS system for DLC use. Namely, CUDA, drivers, Docker, and anaconda!

### Let's start with CUDA support for your GPU:

`sudo apt install gcc`

then:

```python

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

```

Then:
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers autoinstall
```

then:

`reboot`

re-open terminal and check gcc version:

`gcc --version`

output:
```python
gcc --version
gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
Then finish installation:

`sudo apt install nvidia-cuda-toolkit gcc-9`

Then check:

`nvcc --version`

All set! If error messages, read them carefully as they often tell you how to fix it, or what to google :D

Now you can see CUDA, DRIVER, GPU(s):

`nvidia-smi`

output:

```python
nvidia-smi
Tue Jun 22 18:46:26 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:0B:00.0  On |                  N/A |
|  0%   46C    P8    11W / 200W |    252MiB /  8116MiB |      5%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

```

### Next, Docker!

```
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```
add key: `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`

```
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
Then:
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

some clean up:
`sudo apt autoremove`

now you can run `sudo docker run hello-world`

and get:
```
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

### Next, Anaconda!

Click here to get the ubuntu/linux package: https://www.anaconda.com/products/individual#linux

this downloads a file, save it (I save into downloads)

then `cd Downloads`:

and run:

`bash Anaconda3-2021.05-Linux-x86_64.sh`

and you get:
```python
Welcome to Anaconda3 2021.05

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>
```

Follow prompts!

### Next, DeepLabCut!

Given this is a totally fresh install, here are a few things that I also needed: `sudo apt install libcanberra-gtk-module libcanberra-gtk3-module`

We strongly recommend for Ubuntu users to use Docker (https://hub.docker.com/r/deeplabcut/deeplabcut) - it's a much more reproducible environment.

If you want to use our conda file, then I proceeded below:

I grab the conda file from the website at www.deeplabcut.org. Simply click to download. For me, this goes into Downloads.

So, I open a terminal, `cd Downloads`, and then run: `conda env create -f DEEPLABCUT.yaml`

Follow prompts!

## Troubleshooting: Note, if you get a failed build due to wxPython (note, this does not happen on Ubuntu 18, 16, etc), i.e.:

```{warning}
DeepLabCut no longer uses `wxpython` for its GUI - if you're getting such an error, 
you're likely installing an old version of DeepLabCut.
```

```python
ERROR: Command errored out with exit status 1: /home/mackenzie/anaconda3/envs/DLC-GPU/bin/python -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-0jsmkrr1/wxpython_aeff462b2060421a9cf65df55f63a126/setup.py'"'"'; __file__='"'"'/tmp/pip-install-0jsmkrr1/wxpython_aeff462b2060421a9cf65df55f63a126/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /tmp/pip-record-pzy9q5u2/install-record.txt --single-version-externally-managed --compile --install-headers /home/mackenzie/anaconda3/envs/DLC-GPU/include/python3.7m/wxpython Check the logs for full command output.

failed

CondaEnvException: Pip failed
```
You can either: remove conda env: `conda remove --name DEEPLABCUT --all`, open the DLC-GPU.yaml file (any text editor!) and change `deeplabcut[gui]` to `deeplabcut`. Then run: `conda env create -f DEEPLABCUT.yaml` again...

then you will get:
```python

 Successfully uninstalled decorator-5.0.9
Successfully installed PyWavelets-1.1.1 absl-py-0.13.0 astor-0.8.1 bayesian-optimization-1.2.0 chardet-4.0.0 click-8.0.1 cycler-0.10.0 cython-0.29.23 decorator-4.4.2 deeplabcut-2.1.10.4 filterpy-1.4.5 gast-0.2.2 google-pasta-0.2.0 grpcio-1.38.1 h5py-2.10.0 idna-2.10 imageio-2.9.0 imageio-ffmpeg-0.4.4 imgaug-0.4.0 intel-openmp-2021.2.0 joblib-1.0.1 keras-applications-1.0.8 keras-preprocessing-1.1.2 kiwisolver-1.3.1 llvmlite-0.34.0 markdown-3.3.4 matplotlib-3.1.3 moviepy-1.0.1 msgpack-1.0.2 msgpack-numpy-0.4.7.1 networkx-2.5.1 numba-0.51.1 numexpr-2.7.3 numpy-1.17.5 opencv-python-4.5.2.54 opencv-python-headless-3.4.9.33 opt-einsum-3.3.0 pandas-1.2.5 patsy-0.5.1 pillow-8.2.0 proglog-0.1.9 protobuf-3.17.3 psutil-5.8.0 pytz-2021.1 pyyaml-5.4.1 requests-2.25.1 ruamel.yaml-0.17.9 ruamel.yaml.clib-0.2.2 scikit-image-0.18.1 scikit-learn-0.24.2 scipy-1.7.0 statsmodels-0.12.2 tables-3.6.1 tabulate-0.8.9 tensorboard-1.15.0 tensorflow-estimator-1.15.1 tensorflow-gpu-1.15.5 tensorpack-0.9.8 termcolor-1.1.0 threadpoolctl-2.1.0 tifffile-2021.6.14 tqdm-4.61.1 urllib3-1.26.5 werkzeug-2.0.1 wrapt-1.12.1

done
#
# To activate this environment, use
#
#     $ conda activate DEEPLABCUT
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Activate! `conda activate DEEPLABCUT` and then run: `conda install -c conda-forge wxpython`.

Then run `python -m deeplabcut` which launches the DLC GUI.

## DeepLabCut MacOS M-chip installation environment instructions:

This only assumes you have anaconda installed. Use the `DEEPLABCUT_M1.yaml` conda file
if you have a newer MacBook (with an M1, M2, M3, M4 chip or more later), and follow
these steps:

(1) git clone the deeplabcut cut repo:

```bash
git clone https://github.com/DeepLabCut/DeepLabCut.git
```

(2) in the program terminal run: `cd DeepLabCut/conda-environments`

(3) Then, run:

```bash
conda env create -f DEEPLABCUT.yaml
```

(4) Finally, activate your environment and to launch DLC with the GUI

```bash
conda activate DEEPLABCUT
python -m deeplabcut
```

The GUI will open. Of course, you can also run DeepLabCut in headless mode.

If **you want to use the TensorFlow engine**, you'll need to install the `apple_mchips`
extra with DeepLabCut. You can do so by running:

```bash
pip install deeplabcut[apple_mchips]
```

## How to confirm that your GPU is being used by DeepLabCut

During training and analysis steps, DeepLabCut does not use the GPU processor heavily. To confirm that DeepLabCut is properly using your GPU:

**On Windows**:

(1) Open the task manager. If it looks like the image below, click on "More Details" 

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/a0db3157-2228-4444-8084-36801659f272/installBrandon1.png?format=500w)

(2) That will bring up the following, which still isn't helpful and has caused confusion for users. The %GPU does not reflect DeepLabCut usage.

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/117e3573-60bb-4599-b00b-c75276b24173/installBrandon2.png?format=500w)

(3) Click on the **Performance** tab. On that page, click on the small arrow under GPU (it might start as **3D**, and change it to **CUDA**.  

(4) During training, you should see the **Dedicated GPU memory usage** increase to near maximum, and you should see some activity in the **CUDA** graph. The graph below is the activity while running `testscript.py`.

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/b1d03ca0-f8ba-4a31-a399-6e86856c81b0/installBrandon3.png?format=500w)

(5) If you don't see activity there during training, then your GPU is likely not installed correctly for DeepLabCut. Return to the installation instructions, and be sure you installed CUDA 11+, and ran `conda install cudnn -c conda-forge` after installing DeepLabCut.

## How to install DeepLabCut for Intel and AMD GPUs on Windows for the TensorFlow engine

If you are on Windows 10/11 and have a DirectX 12 compatible GPU from any vendor (AMD, Intel, or Nvidia), you utilise GPU acceleration for inference, with an installation that is consistent between devices. This method uses [Tensorflow-directml](https://github.com/microsoft/tensorflow-directml) which uses DirectML instead of Cuda for ML training and inference.

To check the DirectX version of your installed GPU, type in dxdiag into windows search and select the run command. In system information, the bottom item of the list shows your DirectX version. In addition to this ensure your standard GPU drivers are up-to-date. Updating drivers by any official means (Nvidia Geforce experience, AMD radeon software, direct from the vendor website) is fine.

The following instructions are using conda and pip for environment management, executing within the Anaconda prompt program that was installed along with Anaconda python. The # lines are not to be typed, they are for guidance.

```shell
conda create --name dlc_dml python=3.7
conda activate dlc_dml
#specific versions noted that are validated as of DLC 2.2.0.6, but other versions may work:
pip install 'deeplabcut[gui]'==2.2.0.6
pip install tensorflow-directml==1.15.5
pip install pip install imageio==2.9.0
conda install ffmpeg==4.2.2
```

Attention: Please note the order of execution of these commands are important, as pip's dependency manager may change package versions to incorrect ones if done in the wrong version.
