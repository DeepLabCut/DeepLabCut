(overview)=
# 🥳 Get started with DeepLabCut: our key recommendations 

Below we will first outline what you need to get started, the different ways you can use DeepLabCut, and then the full workflow. Note, we highly recommend you also read and follow our [Nature Protocols paper](https://www.nature.com/articles/s41596-019-0176-0), which is (still) fully relevant to standard DeepLabCut.

```{Hint}
💡📚 If you are new to Python and DeepLabCut, you might consider checking our [beginner guide](https://deeplabcut.github.io/DeepLabCut/docs/beginner-guides/beginners-guide.html) once you are ready to jump into using the DeepLabCut App!
```


## [How to install DeepLabCut](how-to-install)

We don't cover installation in depth on this page, so click on the link above if that is what you are looking for. See below for details on getting started with DeepLabCut!

## What we support:

We are primarily a package that enables deep learning-based pose estimation. We have a lot of models and options, but don't get overwhelmed -- the developer team has tried our best to "set the best defaults we possibly can"!

- Decide on your needs: there are **two main modes, standard DeepLabCut or multi-animal DeepLabCut**. We highly recommend carefully considering which one is best for your needs. For example, a white mouse + black mouse would call for standard, while two black mice would use multi-animal. **[Important Information on how to use DLC in different scenarios (single vs multi animal)](important-info-regd-usage)** Then pick a user guide:

  - (1) [How to use standard DeepLabCut](single-animal-userguide)
  - (2) [How to use multi-animal DeepLabCut](multi-animal-userguide)

- To note, as of DLC3+ the single and multi-animal code bases are more integrated and we support **top-down**,  **bottom-up**, and a new "hybrid" approach that is state-of-the-art, called **BUCTD** (bottom-up conditional top down), models.
  - If these terms are new to you, check out our [Primer on Motion Capture with Deep Learning!](https://www.sciencedirect.com/science/article/pii/S0896627320307170). In brief, both work for single or multiple animals and each method can be better or worse on your data.

<p align="center">
<img src= https://ars.els-cdn.com/content/image/1-s2.0-S0896627320307170-gr5_lrg.jpg?format=1000w width="50%">
 </p>

  - Here is more information on BUCTD:
<p align="center">
<img src= https://github.com/amathislab/BUCTD/raw/main/media/BUCTD_fig1.png?format=1000w width="50%">
 </p>
 
 **Additional Learning Resources:**

 - [TUTORIALS:](https://www.youtube.com/channel/UC2HEbWpC_1v6i9RnDMy-dfA?view_as=subscriber) video tutorials that demonstrate various aspects of using the code base.
 - [HOW-TO-GUIDES:](overview) step-by-step user guidelines for using DeepLabCut on your own datasets (see below)
 - [EXPLANATIONS:](https://github.com/DeepLabCut/DeepLabCut-Workshop-Materials) resources on understanding how DeepLabCut works
 - [REFERENCES:](https://github.com/DeepLabCut/DeepLabCut#references) read the science behind DeepLabCut
 - [BEGINNER GUIDE TO THE GUI](https://deeplabcut.github.io/DeepLabCut/docs/beginners-guide.html)

Getting Started: [a video tutorial on navigating the documentation!](https://www.youtube.com/watch?v=A9qZidI7tL8)


### What you need to get started:

 - **a set of videos that span the types of behaviors you want to track.** Having 10 videos that include different backgrounds, different individuals, and different postures is MUCH better than 1 or 2 videos of 1 or 2 different individuals (i.e. 10-20 frames from each of 10 videos is **much better** than 50-100 frames from 2 videos).

 - **minimally, a computer w/a CPU.** If you want to use DeepLabCut on your own computer for many experiments, then you should get an NVIDIA GPU. See technical specs [here](https://github.com/DeepLabCut/DeepLabCut/wiki/FAQ). You can also use cloud computing resources, including COLAB ([see how](https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/README.md)).


### What you DON'T need to get started:

 - no specific cameras/videos are required; color, monochrome, etc., is all fine. If you can see what you want to measure, then this will work for you (given enough labeled data).

 - no specific computer is required (but see recommendations above), our software works on Linux, Windows, and MacOS.


### Overview:
**DeepLabCut** is a software package for markerless pose estimation of animals performing various tasks. The software can manage multiple projects for various tasks. Each project is identified by the name of the project (e.g. TheBehavior), name of the experimenter (e.g. YourName), as well as the date at creation. This project folder holds a ``config.yaml`` (a text document) file containing various (project) parameters as well as links the data of the project.


<p align="center">
<img src=   https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572293604382-W6BWA63LZ9J8R7N0QEA5/ke17ZwdGBToddI8pDm48kIw6YkRUEyoge4858uAJfaMUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYwL8IeDg6_3B-BRuF4nNrNcQkVuAT7tdErd0wQFEGFSnH9wUPiI8bGoX-EQadkbLIJwhzjIpw393-uEwSKO7VZIL9gN_Sb5I_dLwvWryjeCJg/dlc_overview-01.png?format=1000w width="80%">
 </p>

 <p align="center">
 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1560124235138-A9VEZB45SQPD5Z0BDEXA/ke17ZwdGBToddI8pDm48kKsvCFNoOAts8bgs5LXY20UUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcZaDohTswVrVk6oKw3G03bTl18OXeDyNJsBjNlGiyPYGo9Ewyd5AI5wx6CleNeBtf/dlc_steps.jpg?format=1000w" width="80%">
</p>

### Overview of the workflow:
This page contains a list of the essential functions of DeepLabCut as well as demos. There are many optional parameters with each described function. For detailed function documentation, please refer to the main user guides or API documentation. For additional assistance, you can use the [help](UseOverviewGuide.md#help) function to better understand what each function does.

<p align="center">
  <img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5cca272524a69435c3251c40/1556752170424/flowfig.jpg?format=1000w" width=95%>
  <br>
  <em>
   <a href="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5cca272524a69435c3251c40/1556752170424/flowfig.jpg?format=1000w">View in full screen</a>
  </em>
</p>

You can have as many projects on your computer as you wish. You can have DeepLabCut installed in an [environment](../conda-environments/README.md) and always exit and return to this environment to run the code. You just need to point to the correct ``config.yaml`` file to [jump back in](/docs/UseOverviewGuide.md#tips-for-daily-use)! The documentation below will take you through the individual steps.

<p align="center">
<img src=  https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1559758477126-B9PU1EFA7L7L1I24Z2EH/ke17ZwdGBToddI8pDm48kH6mtUjqMdETiS6k4kEkCoR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UQf4d-kVja3vCG3Q_2S8RPAcZTZ9JxgjXkf3-Un9aT84H3bqxw7fF48mhrq5Ulr0Hg/howtouseDLC2d_3d-01.png?format=500w width="60%">
 </p>


(important-info-regd-usage)=

# Specific Advice for Using DeepLabCut:

## Important information on using DeepLabCut:

We recommend first using **DeepLabCut for a single animal scenario** to understand the workflow - even if it's just our demo data. Multi-animal tracking is more complex - i.e. it has several decisions the user needs to make. Then, when you are ready you can jump into multi-animals...

### Additional information for getting started with maDeepLabCut:

We highly recommend using it first in the Project Manager GUI ([Option 3](docs/functionDetails.md#deeplabcut-project-manager-gui)). This will allow you to get used to the additional steps by being walked through the process. Then, you can always use all the functions in your favorite IDE, notebooks, etc.

### *What scenario do you have?*

- **I have single animal videos:**
   - quick start: when you `create_new_project` (and leave the default flag to False in `multianimal=False`). This is the typical work path for many of you.

- **I have single animal videos, but I want to use the updated network capabilities introduced for multi-animal projects:**
   - quick start: when you `create_new_project` just set the flag `multianimal=True`. This enables you to use maDLC features even though you have only one animal. To note, this is rarely required for single animal projects, and not the recommended path. Some tips for when you might want to use this: this is good for say, a hand or a mouse if you feel the "skeleton" during training would increase performance. DON'T do this for things that could be identified an individual objects. i.e., don't do whisker 1, whisker 2, whisker 3 as 3 individuals. Each whisker always has a specific spatial location, and by calling them individuals you will do WORSE than in single animal mode.

[VIDEO TUTORIAL AVAILABLE!](https://youtu.be/JDsa8R5J0nQ)

- **I have multiple *identical-looking animals* in my videos:**
   - quick start: when you `create_new_project` set the flag `multianimal=True`. If you can't tell them apart, you can assign the "individual" ID to any animal in each frame. See this [labeling w/2.2 demo video](https://www.youtube.com/watch?v=_qbEqNKApsI)

[VIDEO TUTORIAL AVAILABLE!](https://youtu.be/Kp-stcTm77g)

- **I have multiple animals, *but I can tell them apart,* in my videos and want to use DLC2.2:**
   - quick start: when you `create_new_project` set the flag `multianimal=True`. And always label the "individual" ID name the same; i.e. if you have mouse1 and mouse2 but mouse2 always has a miniscope, in every frame label mouse2 consistently. See this [labeling w/2.2 demo video](https://www.youtube.com/watch?v=_qbEqNKApsI). Then, you MUST put the following in the config.yaml file: `identity: true`

[VIDEO TUTORIAL AVAILABLE!](https://youtu.be/Kp-stcTm77g) - ALSO, if you can tell them apart, label animals them consistently!

- **I have a pre-2.2 single animal project, but I want to use 2.2:**

Please read [this convert 2 maDLC guide](convert-maDLC)

# The options for using DeepLabCut:

Great - now that you get the overall workflow let's jump in! Here, you have several options.

[**Option 1**](using-demo-notebooks) DEMOs: for a quick introduction to DLC on our data.

[**Option 2**](using-project-manager-gui) Standalone GUI: is the perfect place for
beginners who want to start using DeepLabCut on your own data.

[**Option 3**](using-the-terminal) In the terminal: is best for more advanced users, as
with the terminal interface you get the most versatility and options.

(using-demo-notebooks)=
## Option 1: Demo Notebooks:
[VIDEO TUTORIAL AVAILABLE!](https://www.youtube.com/watch?v=DRT-Cq2vdWs)

We provide Jupyter and COLAB notebooks for using DeepLabCut on both a pre-labeled dataset, and on the end user's
own dataset. See all the demo's [here!](../examples/README.md) Please note that GUIs are not easily supported in Jupyter in MacOS, as you need a framework build of python. While it's possible to launch them with a few tweaks, we recommend using the Project Manager GUI or terminal, so please follow the instructions below.

(using-project-manager-gui)=
## Option 2: using the Project Manager GUI:
[VIDEO TUTORIAL!](https://www.youtube.com/watch?v=KcXogR-p5Ak)

[VIDEO TUTORIAL#2!](https://youtu.be/Kp-stcTm77g)

Start Python by typing ``ipython`` or ``python`` in the terminal (note: using pythonw for Mac users was depreciated in 2022).
If you are using DeepLabCut on the cloud, you cannot use the GUIs. If you use Windows, please always open the terminal with administrator privileges. Please read more in our Nature Protocols paper [here](https://www.nature.com/articles/s41596-019-0176-0). And, see our [troubleshooting wiki](https://github.com/DeepLabCut/DeepLabCut/wiki/Troubleshooting-Tips).

Simply open the terminal and type:
```python
python -m deeplabcut
```
That's it! Follow the GUI for details

(using-the-terminal)=
## Option 3: using the program terminal, Start iPython*:

[VIDEO TUTORIAL AVAILABLE!](https://www.youtube.com/watch?v=7xwOhUcIGio)

Please decide with mode you want to use DeepLabCut, and follow one of the following:

- (1) [How to use standard DeepLabCut](single-animal-userguide)
- (2) [How to use multi-animal DeepLabCut](multi-animal-userguide)
