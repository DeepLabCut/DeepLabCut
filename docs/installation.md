## Technical Considerations 

For reference, we use e.g. Dell workstations (79xx series) with Ubuntu 16.04 LTS and run a Docker container that has TensorFlow, etc. installed (https://github.com/AlexEMG/Docker4DeepLabCut). The code also runs on Windows (thanks to  [Richard Warren](https://github.com/rwarren2163) for checking it) or MacOS (some users have already successfully done so). 

- Hardware:
     - Ideally, you will use a strong GPU with at least 8GB memory such as the [NVIDIA GeForce 1080 Ti](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080/). There are no other hardware requirements. In particular, the software is very robust to track data from pretty much any camera (grayscale, color, or graysale captured under infrared light etc.). A GPU is not necessary, but on a CPU the (training and evaluation) code is considerably slower (100x). 
     
- Software: 
     - We recommend using the supplied [Docker container](https://github.com/AlexEMG/Docker4DeepLabCut). NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)
     - The toolbox is written in [Python 3](https://www.python.org/). You will need [TensorFlow](https://www.tensorflow.org/) (we used 1.0 for figures in papers, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.4 and 1.8**) for Python 3 with GPU support. 
     - To note, is it possible to run DeepLabCut on your CPU, but it will be VERY slow. However, this is the preferred path if you want to test DeepLabCut on your data before purchasing a GPU, with the added benefit of a straightforward installation. 

## Simplified installation with conda environments for a CPU:

[Anaconda](https://anaconda.org/anaconda/python) is perhaps the easiest way to install Python and additional packages across various operating systems. With anaconda just create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine by running in a terminal:
```
git clone https://github.com/AlexEMG/DeepLabCut.git
cd DeepLabCut
conda env create -f dlcdependencies.yml
```
Note that this environment does not contain Tensorflow, but all other dependencies. 

Some conda channels are different for Windows and Rick Warren contributed the following yml file for **Windows**:
```
conda env create -f dlcDependenciesFORWINDOWS.yaml
```


## Conda environment with TensorFlow for CPU support installed do (instead):
```
conda env create -f dlcdependencieswTF1.2.yml
```
Again on **Windows** use (instead with TensorFlow 1.8):
```
conda env create -f dlcDependenciesFORWINDOWSwTF.yaml
```

Note that this environment yaml file was created on Ubuntu 16.04, so the installation of TensorFlow [might not work across platforms (Windows, MacOS)](https://stackoverflow.com/questions/39280638/how-to-share-conda-environments-across-platforms). Installing TensorTlow on your CPU is easy on any platform, if the environment yaml with TensorFlow does not work for you, then install the one without and then follow the instructions here for installing [Tensorflow](https://www.tensorflow.org/versions/r1.2/install/). 

Once you installed the environment you can activate it, by typing on Linux/MacOS: 
```
source activate DLCdependencies
```
and on Windows do: 
```
activate DLCdependencies
```

Then you can work with all DLC functionalities inside this environment. 

# TensorFlow Installation with GPU support:

- **Docker: We highly recommend to use the supplied [Docker container](https://github.com/AlexEMG/Docker4DeepLabCut), which has everything including TensorFlow for the GPU preinstalled. NOTE: [this container does not work on windows hosts!](https://github.com/NVIDIA/nvidia-docker/issues/43)**

 - If you do not want to use Docker, here are the dependencies: 

     - Install Sypder (or equivalent IDE) and/or Jupyter Notebook
     - Clone (or download) the code we provide "git clone https://github.com/AlexEMG/DeepLabCut.git"
     - You will also need to install the following Python packages (in the terminal type):
     ```
      $ pip install scipy scikit-image matplotlib pyyaml easydict 
      $ pip install moviepy imageio tqdm tables sk-video pandas requests
      $ git clone https://github.com/AlexEMG/DeepLabCut.git
      ```
Then install [TensorFlow](https://www.tensorflow.org/). Ideally install **TensorFlow 1.0 with CUDA (Cuda 8.0)** (or TensorFlow 1.2, 1.4 or 1.8 / other versions might work too). Please check your CUDA and [TensorFlow installation](https://www.tensorflow.org/install/) with this line (below), and you can test that your GPU is being properly engaged with these additional [tips](https://www.tensorflow.org/programmers_guide/using_gpu).

      $ sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
           
      
 Return to [readme](../README.md).

