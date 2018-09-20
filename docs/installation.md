## Technical Considerations 

For reference, we use e.g. Dell workstations (79xx series) with Ubuntu 16.04 LTS and run a Docker container that has TensorFlow, etc. installed (https://github.com/AlexEMG/Docker4DeepLabCut). The code also runs on Windows (thanks to  [Richard Warren](https://github.com/rwarren2163) for checking it) or MacOS (some users have already successfully done so). 

- Camera Hardware:
     - The software is very robust to track data from pretty much any camera (grayscale, color, or graysale captured under infrared light etc.). See demos on our [website](https://www.mousemotorlab.org/deeplabcut/)

- Computer Hardware:
     - Ideally, you will use a strong GPU with at least 8GB memory such as the [NVIDIA GeForce 1080 Ti](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080/).  A GPU is not necessary, but on a CPU the (training and evaluation) code is considerably slower (100x). You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/AlexEMG/DeepLabCut/issues/47).
     
- Software: 
     - We highly recommend using the supplied [Docker container](https://github.com/AlexEMG/Docker4DeepLabCut). NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)
     - The toolbox is written in [Python 3](https://www.python.org/). You will need [TensorFlow](https://www.tensorflow.org/) (we used version 1.0 in the paper, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.4, 1.8, and  1.10**) for Python 3 with GPU support. 
     - To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow. However, this is the preferred path if you want to test DeepLabCut on your data before purchasing a GPU, with the added benefit of a straightforward installation. 

## Anaconda environment with TensorFlow on GPU support:

- **Docker: We highly recommend to use the supplied [Docker container](https://github.com/AlexEMG/Docker4DeepLabCut), which has everything including TensorFlow for the GPU preinstalled! NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)**

 - If you cannot use the Docker container, [Anaconda](https://anaconda.org/anaconda/python) is perhaps the easiest way to install Python and additional packages across various operating systems. First create an [Anaconda](https://anaconda.org/anaconda/python) environment with all dependencies, but TensorFlow.  With Anaconda you create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine by running in a terminal:

**LINUX/MacOS:**
```
git clone https://github.com/AlexEMG/DeepLabCut.git
cd deeplabcut/conda-files
conda env create -f dlcdependencies.yml
```
**Windows:** 
```
git clone https://github.com/AlexEMG/DeepLabCut.git
cd deeplabcut/conda-files
conda env create -f dlcDependenciesFORWINDOWS.yaml
```

**Then, for LINUX & Windows:** Install [TensorFlow](https://www.tensorflow.org/). We used **TensorFlow 1.0 with CUDA (Cuda 8.0)**. Therefore, we recommend this version. Some other versions of TensorFlow have been tested, but use at your own risk (i.e. these versions have been tested 1.2, 1.4, 1.8 or 1.10, but might require different CUDA versions! Please check your driver/CUDA/TensorFlow version [on this Stackoverflow post](https://stackoverflow.com/questions/30820513/what-is-version-of-cuda-for-nvidia-304-125/30820690#30820690). Please check your CUDA and [TensorFlow installation](https://www.tensorflow.org/install/) with the line below

      $ sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

You can test that your GPU is being properly engaged with these additional [tips](https://www.tensorflow.org/programmers_guide/using_gpu).

## Anaconda environment with TensorFlow for only a CPU:

[Anaconda](https://anaconda.org/anaconda/python) is perhaps the easiest way to install Python and additional packages across various operating systems. With Anaconda just create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine by running in a terminal:

**LINUX/MacOS:**
```
git clone https://github.com/AlexEMG/DeepLabCut.git
cd deeplabcut/conda-files
conda env create -f dlcdependencieswTF1.2.yml
```

Some conda channels are different for Windows and Rick Warren contributed the following yml file for **Windows** which actually features TensorFlow 1.8:

**Windows:** 
```
git clone https://github.com/AlexEMG/DeepLabCut.git
cd deeplabcut/conda-files
conda env create -f dlcDependenciesFORWINDOWSwTF.yaml
```

That's it! You should now be able to run the .py files from inside the environment in the terminal.

Note that these environment yaml file [might not work across platforms (other versions of Windows, MacOS, ...)](https://stackoverflow.com/questions/39280638/how-to-share-conda-environments-across-platforms). Installing TensorTlow on your CPU is easy on any platform, if the environment yaml with TensorFlow does not work for you, then install the one without, and then follow the instructions here for installing [Tensorflow 1.2](https://www.tensorflow.org/versions/r1.2/install/). 

Once you installed the environment you can activate it, by typing on Linux/MacOS: 
```
source activate DLCdependencies
```

and on Windows do: 
```
activate DLCdependencies
```

Then you can work with all DLC functionalities inside this environment. Enjoy!

## Alternative Python installation (not Anaconda environment based):

If you wish to just install everything directly onto your computer (i.e. you don't need other versions of TF, CUDA, etc), then you can install these packages using the terminal:


 - Install Python3 [e.g. Sypder (or any other IDE) and/or Jupyter Notebook for Python3]
 - Clone (or download) the code we provide

     $ git clone https://github.com/AlexEMG/DeepLabCut

 - You will also need to install the following **Python3** packages (in the terminal type):

```
      $ pip install scipy scikit-image sk-video matplotlib pyyaml easydict 
      $ pip install moviepy imageio tqdm tables
      $ pip install pyqt5 scikit-learn pandas
      $ pip install ffmpeg-python
```
Note if you also have python2 installed, you might want to use pip3 instead (DeepLabCut is suported in Python3.x only). 

Then, install *TensorFlow*:

**Then, for LINUX & Windows:** Install [TensorFlow](https://www.tensorflow.org/). We used **TensorFlow 1.0 with CUDA (Cuda 8.0)**. Therefore, we recommend this version. Some other versions of TensorFlow have been tested, but use at your own risk (i.e. these versions have been tested 1.2, 1.4, 1.8 or 1.10, but might require different CUDA versions! Please check your driver/CUDA/TensorFlow version [on this Stackoverflow post](https://stackoverflow.com/questions/30820513/what-is-version-of-cuda-for-nvidia-304-125/30820690#30820690). Here is a nice guide for [CUDA + cuDNN for TensorFlow installation on Ubuntu](https://medium.com/@ikekramer/installing-cuda-8-0-and-cudnn-5-1-on-ubuntu-16-04-6b9f284f6e77).


Please check your CUDA and [TensorFlow installation](https://www.tensorflow.org/install/) with th line below

      $ sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

You can test that your GPU is being properly engaged with these additional [tips](https://www.tensorflow.org/programmers_guide/using_gpu).

Now you can use Jupyer Notebooks, Spyder, and to train just use the terminal, to run all the code!
          
      
 Return to [readme](../README.md).

