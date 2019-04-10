## Documentation:

This page contains a list of the essential functions of DeepLabCut as well as demos. There are many optional parameters with each described function, which you can find [here](functionDetails.md). For additional assistance, you can use the [help](UseOverviewGuide.md#help) function to better understand what each function does.

## Overview of the work-flow:

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c3e47454fa51a420fa8ecdf/1547585367234/flowfig.png?format=750w" width="90%">
</p>

### Option 1: Demo Notebooks:
We also provide Jupyter notebooks for using DeepLabCut on both a pre-labeled dataset, and on the end user’s
own dataset. See all the demo's [here!](/examples)

### Option 2: using terminal, Start Python:

Open an ipython session and import the package by typing in the terminal:
Please note, if you are using MacOS, you must use ``pythonw`` vs. ``ipython`` (also, GUIs are not supported in Jupyter in MacOS so please follow the instructions below!)

``ipython``

``import deeplabcut``

TIP: for every function there is a associated help document that can be viewed by adding a **?** after the function name; i.e. ``deeplabcut.create_new_project?``. To exit this help screen, type q.


### Create a New Project:

``deeplabcut.create_new_project(`Name of the project',`Name of the experimenter', [`Full path of video 1',`Full path of video2',`Full path of video3'], working_directory=`Full path of the working directory',copy_videos=True/False)``

- Note, if you are a Ubuntu user the path should look like: ``['/home/username/yourFolder/video1.mp4']``; if you are a Windows user, it should look like: ``[r'C:\username\yourFolder\video1.mp4']``
- Note, you can also put ``config_path = `` in front of the above line to create the path to the config.yaml that is used in the next step, i.e. ``config_path=deeplabcut.create_project(...)``)
    - If you do not, we recommend setting a variable so this can be easily used! Once you run this step, the conig_path is printed for you!, so set something like: ``config_path = 'theoutputpath/yaml.yaml' `` just be mindful of the formating, see below.


(more details [here](functionDetails.md#a-create-a-new-project))

### Configure the Project:

- open the **config.yaml** file (in a text editor (like atom, gedit, vim etc.)), which can be found in the subfolder created when you set your project name, to change parameters and identify label names!

(PLEASE see more details [here](functionDetails.md#b-configure-the-project))

- set the config_path (LINUX): ``config_path = '/home/computername/DeepLabCut/yourprojectname/config.yaml'`` (for example)
- set the config_path (WINDOWS): ``config_path = r'C:\home\computername\DeepLabCut\yourprojectname\config.yaml'`` (for example)

- or you already set it as a variable in step "create_new_project" directly above.

**mini-demo:** create project and edit the yaml file

<p align="center">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/startdeeplabcut.gif" width="90%">
</p>

### Select Frames to Label:

``deeplabcut.extract_frames(config_path,`automatic/manual',`uniform/kmeans', crop=True/False)``

(more details [here](functionDetails.md#c-data-selection)) *update: as of 2.0.5 ``checkcropping=True`` is droppped; you now just the option to directly draw a rectangle over the image to crop before extraction (i.e. there no need to manually change in config.yaml then check).

### Label Frames:

``deeplabcut.label_frames(config_path)``

(more details [here](functionDetails.md#d-label-frames))

**mini-demo:** using the GUI to label

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c535f16fa0d60294ac39e4e/1548967721381/GUIdemo1.gif?format=750w" width="60%">
</p>

### Check Annotated Frames:

``deeplabcut.check_labels(config_path)``

(more details [here](functionDetails.md#e-check-annotated-frames))

### Create Training Dataset:

`` deeplabcut.create_training_dataset(config_path,num_shuffles=1)``

(more details [here](functionDetails.md#f-create-training-dataset))

### Train The Network:

``deeplabcut.train_network(config_path,shuffle=1)``

(more details [here](functionDetails.md#g-train-the-network))

### Evaluate the Trained Network:

``deeplabcut.evaluate_network(config_path,shuffle=[1], plotting=True)``

(more details [here](functionDetails.md#h-evaluate-the-trained-network))

### Video Analysis and Plotting Results:
- Plesae note that novel videos DO NOT need to be added to the config.yaml file. You can simply have a folder eslsewhere on your computer and pass the video folder (then it will analyze all videos of the specified type (i.e. ``videotype='.mp4'``), or pass the path to the exact video you wish to analyze:

``deeplabcut.analyze_videos(config_path,[`/fullpath/project/videos/reachingvideo1.avi'], shuffle=1, save_as_csv=True)``

``deeplabcut.create_labeled_video(config_path, [`/analysis/project/videos/reachingvideo1.avi',`/fullpath/project/videos/reachingvideo2.avi'])``

``deeplabcut.plot_trajectories(config_path,[`/fullpath/project/videos/reachingvideo1.avi'])``

You can also filter the points by:

``deeplabcut.filterpredictions(config_path,[`/fullpath/project/videos/reachingvideo1.avi'], shuffle=1)``
Note, this creates a file with the ending filtered.h5 that you can use for further analysis. This filtering step has many parameters, so please see the full docstring by typing: ``deeplabcut.filterpredictions?``

(more details [here](functionDetails.md#i-video-analysis-and-plotting-results))

# optional Refinement: Extract Outlier Frames:

``deeplabcut.extract_outlier_frames(config_path,[`videofile_path'])``

(more details [here](functionDetails.md#j-refinement-extract-outlier-frames))

### optional refinement of the labels with the GUI:
(refinement and augmentation of the training dataset)

``deeplabcut.refine_labels(config_path)``

**mini-demo:** using the refinement GUI, a user can load the file then zoom, pan, and edit and/or remove points:

<p align="center">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/refinelabels.gif" width="90%">
</p>

When done editing the labels, merge:

``deeplabcut.merge_datasets(config_path)``

(more details [here](functionDetails.md#k-refine-labels-augmentation-of-the-training-dataset))


### HELP:

In ipython/Jupyter notebook:

``deeplabcut.nameofthefunction?``

In Python:

``help(deeplabcut.nameofthefunction)``


Return to [readme](../README.md).
