# Python/iPython:

All of DeepLabCut can be run from an ipython console in the progam **terminal**! Go [here](/docs/UseOverviewGuide.md) for all the instructions!

# Demo Notebooks:

We recommend using **Anaconda to install Python and Jupyter Notebooks, see the [Installation](master/docs/installation.md) page**. Then, on your local machine using these notebooks to guide you, you can (1) demo our labeled data, (2) create a project, extract frames to lablel, use the GUI to label, and create a training set. You can also then run the training that utilizies TensorFlow with a CPU on your local computer (see the [Installation](docs/installation.md)). Installing TensorFlow for GPU support, is a bit more elaborate, so we suggest running the Training and new Video Analysis inside the [supplied Docker container](https://github.com/MMathisLab/Docker4DeepLabCut2.0) (which works on all Linux systems). 

We suggest making a "Fork" of this repo and/or then place DeepLabCut files in a folder :
``git clone https://github.com/AlexEMG/DeepLabCut`` 
so you can access it locally with **Anaconda.** You can also click the "download" button, rather than using ``git``. Then you can edit the Notebooks as you like!

<p align="center">
<img src="/docs/images/gitclone.png" width="90%">
</p>

### Some quick installation notes about each demo, i.e. you need to do this in order to run the Notebooks.

**Demo 1:** This is designed for using your own computer and a **CPU** (unless you want to install Tensorflow-gpu, see [Installation](docs/installation.md)!).
- Required Installation steps are, in the terminal type (but **please** still check [Installation](master/docs/installation.md)):
  - ``pip install deeplabcut``
  - In Windows: ``pip install -U wxPython``
  - in Linux: ``pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl``
  - ``pip install tensorflow==1.10``
  
**Demo 1.5 & 2:** Required Installation steps are, in the terminal type (but **please** still check [Installation](master/docs/installation.md)):
  - ``pip install deeplabcut``
  - In Windows: ``pip install -U wxPython``
  - in Linux: ``pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl``
  - **Demo 1.5:** This now requires the [DeepLabCut Docker](https://github.com/MMathisLab/Docker4DeepLabCut2.0) or your own GPU set up (see [Installation](master/docs/installation.md)).
  - **Demo 2**  can use either CPUs or a GPU.

### First, activate your Anaconda environment (here called DLC2), and open a portal to Jupyter: 
Linux: `` source activate <yourcondaname>`` then ``jupyter notebook`` 
note, if you have never used Jupter NOtebooks before, you need to once do ``export BROWSER=google-chrome``)

<p align="center">
<img src="/docs/images/enterAnacondaNotebook.png" width="90%">
</p>

## Demo 1: run DeepLabCut [on our data](Demo_labeledexample_MouseReaching.ipynb) (already labeled) with a CPU
 - This will give you a feel of the workflow for DeepLabCut.
 - Done? CPU mastered? Excellent - let's move into GPU computing... 
 
## Demo 1.5: Run DeepLabCut on a [GPU in Docker (linux only)](Docker_TrainNetwork_VideoAnalysis.ipynb)
 - This requires the [DeepLabCut Docker](https://github.com/MMathisLab/Docker4DeepLabCut2.0)!

Note, the notebooks with labeled data: [reaching data](Demo_labeledexample_MouseReaching.ipynb), or [trail-tracking data](Demo_labeledexample_Openfield.ipynb) can be run on a CPU, GPU, etc. The one with the trail-tracking data even achieves good results, when trained for half an hour on a GPU!

## Demo 2: Set up DeepLabCut on [your own data](Demo_yourowndata.ipynb) 
- Now that you're a master of the demos, this Notebook walks you through how to build your own pipeline: 
  - Create a new project
  - Label new data 
  - Create a training set 
  - Then, either use your CPU, or your GPU (the Notebook will guide you at this junction), to train, analyze and perform some basic analysis of your data. 

For GPU-based training and analysis you will need to switch to either our [supplied Docker container](https://github.com/MMathisLab/Docker4DeepLabCut2.0), and modify the [Docker Demo Notebook](Docker_TrainNetwork_VideoAnalysis.ipynb) for your project, or you need to [install TensorFlow with GPU support](master/docs/installation.md) in an Anaconda Env.

## Demo DeepLabCut training and analysis on Colaboratory (in the cloud):

- Alternatively, you can use Google [Colaboratory](https://colab.research.google.com) to demo running the training and analysis of new videos, i.e. the parts that need a GPU!

  - Warning: Colab updates their CUDA/TensorFlow likely faster than we can keep up, so this may not work at all future points in time (and, as a reminder, this whole package is released with a [LICENSE](/LICENSE) that implies no Liability and no Warranty). 

We suggest making a "Fork" of this repo, git clone or download the folder into your google drive, then linking your google account to your GitHub (you'll see how to do this in the Notebook below). Then you can edit the Notebook for your own data too (just put https://colab.research.google.com/ in front of the web address of your own repo)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexEMG/DeepLabCut/blob/master/examples/Colab_TrainNetwork_VideoAnalysis.ipynb)

1. Click Open in Colab to launch the notebook.
2. Make the notebook live by clicking 'Connect' in the Colab toolbar, and then click "Runtime > Change Runtime Type > and select Python3 and GPU as your hardware. Follow the instructions in the Notebook.
3. Be aware, they often don't let you run on their GPUs for very long, so make sure your ``save_inters`` variable is low for this setting.
