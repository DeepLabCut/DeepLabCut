## Documentation:

This page contains a list of the essential functions of DeepLabCut as well as demos. There are many optional parameters with each described function, which you can find [here](functionDetails.md). For additional assistance, you can use the [help](UseOverviewGuide.md#help) function to better understand what each function does. 

## Overview of the work-flow:

<p align="center">
<img src="/docs/images/flowfig.png" width="90%">
</p>

### Option 1: Demo Notebooks: 
We also provide Jupyter notebooks for using DeepLabCut on both a pre-labeled dataset, and on the end user’s
own dataset. See all the demo's [here!](/examples)

### Option 2: using terminal, Start Python: 

Open an ipython session and import the package by typing in the terminal:

``ipython``

``import deeplabcut``


### Create a New Project:

``deeplabcut.create_new_project(`Name of the project',`Name of the experimenter', [`Full path of video 1',`Full path of video2',`Full path of video3'], working_directory=`Full path of the working directory',copy_videos=True/False)``

- (Note, you can also put ``config_path = `` in front of the above line to create the path to the config.yaml that is used in the next step, i.e. ``config_path=deeplabcut.create_project(...)``)

(more details [here](functionDetails.md#a-create-a-new-project))

### Configure the Project:

- open the **config.yaml** file (in a text editor (like atom, gedit, vim etc.)), which can be found in the subfolder created when you set your project name, to change parameters and identify label names!

(more details [here](functionDetails.md#b-configure-the-project))

- set the config_path: ``config_path = '/home/computername/DeepLabCut2.0/yourprojectname/config.yaml'`` (for example)

- or you already set it as a variable in step "Create_new_Project" directly above.

**mini-demo:** create project and edit the yaml file

<p align="center">
<img src="/docs/images/startdeeplabcut.gif" width="90%">
</p>

### Select Frames to Label: 

``deeplabcut.extract_frames(`config_path',`automatic/manual',`uniform/kmeans', crop=True/False, checkcropping=True)``

(more details [here](functionDetails.md#c-data-selection))

### Label Frames:

``deeplabcut.label_frames(`config_path')``

(more details [here](functionDetails.md#d-label-frames))

**mini-demo:** using the GUI to label

<p align="center">
<img src="/docs/images/guiexample.gif" width="90%">
</p>

### Check Annotated Frames:

``deeplabcut.check_labels(`config_path')``

(more details [here](functionDetails.md#e-check-annotated-frames))

### Create Training Dataset:

`` deeplabcut.create_training_dataset(`config_path',num_shuffles=1)``

(more details [here](functionDetails.md#f-create-training-dataset))

### Train The Network:

``deeplabcut.train_network(`config_path',shuffle=1)``

(more details [here](functionDetails.md#g-train-the-network))

### Evaluate the Trained Network:

``deeplabcut.evaluate_network(`config_path',shuffle=[1], plotting=True)``

(more details [here](functionDetails.md#h-evaluate-the-trained-network))

### Video Analysis and Plotting Results:

``deeplabcut.analyze_videos(`config_path',[`/analysis/project/videos/reachingvideo1.avi'], shuffle=1, save_as_csv=True)``

``deeplabcut.create_labeled_video(`config_path' [`/analysis/project/videos/reachingvideo1.avi',`/analysis/project/videos/reachingvideo2.avi'])``

``deeplabcut.plot_trajectories(`config_path',[`/analysis/project/videos/reachingvideo1.avi'])``

(more details [here](functionDetails.md#i-video-analysis-and-plotting-results))

# optional Refinement: Extract Outlier Frames:

``deeplabcut.extract_outlier_frames(`config_path',[`videofile_path'])``

(more details [here](functionDetails.md#j-refinement-extract-outlier-frames))

### optional refinement of the labels with the GUI: 
(refinement and augmentation of the training dataset)

``deeplabcut.refine_labels(`config_path')``

**mini-demo:** using the refinement GUI, a user can load the file then zoom, pan, and edit and/or remove points:

<p align="center">
<img src="/docs/images/refinelabels.gif" width="90%">
</p>

When done editing the labels, merge: 

``deeplabcut.merge_datasets(`config_path')``

(more details [here](functionDetails.md#k-refine-labels-augmentation-of-the-training-dataset))


### HELP:

In ipython/Jupyter notebook:

``deeplabcut.nameofthefunction?``

In Python:

``help(deeplabcut.nameofthefunction)``
