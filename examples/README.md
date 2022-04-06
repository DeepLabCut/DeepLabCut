# Demo Jupyter & Colaboratory Notebooks:

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572293604382-W6BWA63LZ9J8R7N0QEA5/ke17ZwdGBToddI8pDm48kIw6YkRUEyoge4858uAJfaMUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYwL8IeDg6_3B-BRuF4nNrNcQkVuAT7tdErd0wQFEGFSnH9wUPiI8bGoX-EQadkbLIJwhzjIpw393-uEwSKO7VZIL9gN_Sb5I_dLwvWryjeCJg/dlc_overview-01.png?format=1000w" width="550" title="DLC" alt="DLC" align="right" vspace = "70">

We provide a Project Manager GUI that will walk you through the major steps and options of the DeepLabCut Toolbox. However, there are more options and features that can be accessed by running the code in an interactive environment, such as Jupyter*. Moreover, if you don't have a GPU, you can create your project on any computer, then move your project to the cloud to use GPUs. To do this, we provide you with Google Colaboratory Notebooks (see [Demo using Google Colaboratory below](/examples#demo-deeplabcut-training-and-analysis-on-google-colaboratory-with-googles-gpus)).


## Demo 1: run DeepLabCut [on our open-field data](JUPYTER/Demo_labeledexample_Openfield.ipynb)
 - This will give you a feel of the workflow for DeepLabCut. Follow the instructions inside the notebook!

Note, the notebooks with labeled data: [reaching data](JUPYTER/Demo_labeledexample_MouseReaching.ipynb), or [open-field data](JUPYTER/Demo_labeledexample_Openfield.ipynb) can be run on a CPU, GPU, etc. The one with the open-field data even achieves good/okay results, when trained for half an hour on a GPU! (To note, this is NOT the full dataset that was used in Mathis et al, 2018)

## Demo 2: Set up DeepLabCut on [your own data](JUPYTER/Demo_yourowndata.ipynb)
- Now that you're a master of the demos, this Notebook walks you through how to build your own pipeline:
  - Create a new project
  - Label new data
  - Then, either use your CPU, or your GPU (the Notebook will guide you at this junction), to train, analyze and perform some basic analysis of your data.

For GPU-based training and analysis you will need to switch to either our [supplied Docker container](https://deeplabcut.github.io/DeepLabCut/docs/docker.html), or you need to [install your local GPU](https://deeplabcut.github.io/DeepLabCut/docs/recipes/installTips.html?highlight=gpu#how-to-confirm-that-your-gpu-is-being-used-by-deeplabcut) in an Anaconda Env, or use Google Colab, more below: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_YOURDATA_TrainNetwork_VideoAnalysis.ipynb)

## Demo 3: DeepLabCut training and analysis on Google Colaboratory (with Google's GPUs!):

We suggest making a "Fork" of this repo, git clone or download the folder into your google drive, then linking your google account to your GitHub (you'll see how to do this in the Notebook below). Then you can edit the Notebooks for your own data too (just put https://colab.research.google.com/ in front of the web address of your own repo).

- You can use Google [Colaboratory](https://colab.research.google.com) to demo running DeepLabCut on our data. Here is an example colab-ready Jupyter Notebook for the open field data, which you can launch by clicking the badge below: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_DEMO_mouse_openfield.ipynb)

- Using Colab on your data for the training and analysis of new videos, i.e. the parts that need a GPU!
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_YOURDATA_TrainNetwork_VideoAnalysis.ipynb)

1. Click Open in Colab to launch the notebook.
2. Make the notebook live by clicking 'Connect' in the Colab toolbar, and then click "Runtime > Change Runtime Type > and select Python3 and GPU as your hardware. Follow the instructions in the Notebook.
3. Be aware, they often don't let you run on their GPUs for very long (>6 hrs) without a Pro account, so make sure your ``save_inters`` variable is lower for this setting.

Here is a demo of us using the Colab Notebooks: https://www.youtube.com/watch?v=qJGs8nxx80A & https://www.youtube.com/watch?v=j13aXxysI2E


*Warning: Colab updates their CUDA/TensorFlow likely faster than we can keep up, so this may not work at all future points in time (and, as a reminder, this whole package is released with a [LICENSE](/LICENSE) that implies no Liability and no Warranty).*

## Using 3D DeepLabCut:

Ready to take your pose estimation to a new dimension? As of 2.0.7+ we support 3D within the package. Please check out the dedicated 3D_Demo_DeepLabCut.ipynb above for more details!

## Using the DLC Model Zoo:

We provide a COLAB notebook to use the growing number of networks that are trained on specific animals/scenarios. Read more here: http://www.mousemotorlab.org/dlc-modelzoo. This code will also create a new project folder so you can refine, add new bodyparts or label other objects, and re-train. Launch COLAB here: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_DLC_ModelZoo.ipynb)

## Using Python/iPython:

All of DeepLabCut can be run from an ipython console in the program **terminal**! Go [here](/docs/UseOverviewGuide.md) for detailed instructions!

We also have some video tutorials to demonstrate how we use Anaconda and Docker via the terminal:

 https://www.youtube.com/watch?v=7xwOhUcIGio &  https://www.youtube.com/watch?v=bgfnz1wtlpo


## * You can download DeepLabCut & associated files:

To have a copy of DeepLabCut on your own computer, we recommend using **Anaconda to install Python and Jupyter Notebooks, see the [Installation](/docs/installation.md) page**. Then, on your local machine using these notebooks to guide you, you can (1) demo our labeled data (or create your own), (2) create a project, extract frames to label, use the GUI to label, and create a training set for the neural network(s).

We suggest making a "Fork" of this repo and/or then place DeepLabCut files in a folder:
``git clone https://github.com/DeepLabCut/DeepLabCut``
so you can access it locally with **Anaconda.** You can also click the "download" button, rather than using ``git``. Then you can edit the Notebooks as you like!
