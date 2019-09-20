# Demo Jupyter & Colaboratory Notebooks:

You can use DeepLabCut in the cloud without any installation (see [Demo using Google Colaboratory below](/examples#demo-deeplabcut-training-and-analysis-on-google-colaboratory-with-googles-gpus)).

On your computer, we recommend using **Anaconda to install Python and Jupyter Notebooks, see the [Installation](/docs/installation.md) page**. Then, on your local machine using these notebooks to guide you, you can (1) demo our labeled data (or create your own), (2) create a project, extract frames to lablel, use the GUI to label, and create a training set. 

We suggest making a "Fork" of this repo and/or then place DeepLabCut files in a folder:
``git clone https://github.com/AlexEMG/DeepLabCut``
so you can access it locally with **Anaconda.** You can also click the "download" button, rather than using ``git``. Then you can edit the Notebooks as you like!

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e475e4d7a9c7b4f872025/1547585387197/gitclone.png?format=500w" width="90%">
</p>

## Demo 1: run DeepLabCut [on our reaching data](Demo_labeledexample_MouseReaching.ipynb) or [on our open-field data](Demo_labeledexample_Openfield.ipynb)
 - This will give you a feel of the workflow for DeepLabCut. Follow the instructions inside the notebook!

Note, the notebooks with labeled data: [reaching data](Demo_labeledexample_MouseReaching.ipynb), or [open-field data](Demo_labeledexample_Openfield.ipynb) can be run on a CPU, GPU, etc. The one with the trail-tracking data even achieves good results, when trained for half an hour on a GPU!

## Demo 2: Run DeepLabCut on a [GPU in Docker (linux only)](Docker_TrainNetwork_VideoAnalysis.ipynb)
 - This requires the [DeepLabCut Docker](https://github.com/MMathisLab/Docker4DeepLabCut2.0)!

## Demo 3: Set up DeepLabCut on [your own data](Demo_yourowndata.ipynb)
- Now that you're a master of the demos, this Notebook walks you through how to build your own pipeline:
  - Create a new project
  - Label new data
  - Then, either use your CPU, or your GPU (the Notebook will guide you at this junction), to train, analyze and perform some basic analysis of your data.

For GPU-based training and analysis you will need to switch to either our [supplied Docker container](https://github.com/MMathisLab/Docker4DeepLabCut2.0), and modify the [Docker Demo Notebook](Docker_TrainNetwork_VideoAnalysis.ipynb) for your project, or you need to [install TensorFlow with GPU support](/docs/installation.md) in an Anaconda Env, or use Google Colab, more below: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexEMG/DeepLabCut/blob/master/examples/Colab_TrainNetwork_VideoAnalysis.ipynb)

## Demo DeepLabCut training and analysis on Google Colaboratory (with Google's GPUs!):

We suggest making a "Fork" of this repo, git clone or download the folder into your google drive, then linking your google account to your GitHub (you'll see how to do this in the Notebook below). Then you can edit the Notebook for your own data too (just put https://colab.research.google.com/ in front of the web address of your own repo)

- You can use Google [Colaboratory](https://colab.research.google.com) to demo running DeepLabCut on our data. Here is an example colab-ready Jupyter Notebook for the open field data, which you can launch by clicking the badge below: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexEMG/DeepLabCut/blob/master/examples/Colab_DEMO_mouse_openfield.ipynb)

- Using Colab on your data for the training and analysis of new videos, i.e. the parts that need a GPU! 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexEMG/DeepLabCut/blob/master/examples/Colab_TrainNetwork_VideoAnalysis.ipynb)

1. Click Open in Colab to launch the notebook.
2. Make the notebook live by clicking 'Connect' in the Colab toolbar, and then click "Runtime > Change Runtime Type > and select Python3 and GPU as your hardware. Follow the instructions in the Notebook.
3. Be aware, they often don't let you run on their GPUs for very long, so make sure your ``save_inters`` variable is low for this setting.

Here is a demo of us using the Colab Notebooks: https://www.youtube.com/watch?v=qJGs8nxx80A & https://www.youtube.com/watch?v=j13aXxysI2E


*Warning: Colab updates their CUDA/TensorFlow likely faster than we can keep up, so this may not work at all future points in time (and, as a reminder, this whole package is released with a [LICENSE](/LICENSE) that implies no Liability and no Warranty).*

## 3D DeepLabCut:

Ready to take your pose estimation to a new dimension? As of 2.0.7+ we support 3D within the package. Please check out the dedicated 3D_Demo_DeepLabCut.ipynb above for more details!

## Human pre-trained network DEMO:

We provide a COLAB notebook (COLAB_Human_Project_DEMO.ipynb) that allows you to immediately analyze videos using a human pre-trained network. Here, your videos should include frames that are around 300 by 300 pixels (although it does not need to be square) for optimal performance. As of 2.0.9, we provide tools to downsample and shorten videos within the toolbox. This code will also create a new project folder so you can refine, add new bodyparts or label other objects, and re-train. Launch COLAB here: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexEMG/DeepLabCut/blob/master/examples/COLAB_Human_Project_DEMO.ipynb)

# Python/iPython:

All of DeepLabCut can be run from an ipython console in the program **terminal**! Go [here](/docs/UseOverviewGuide.md) for detailed instructions!

We also have some video tutorials to demonstrate how we use Anaconda and Docker via the terminal:

 https://www.youtube.com/watch?v=7xwOhUcIGio &  https://www.youtube.com/watch?v=bgfnz1wtlpo

