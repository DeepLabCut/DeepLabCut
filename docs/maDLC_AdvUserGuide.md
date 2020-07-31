# DeepLabCut for multi-animal projects <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1596197885984-YGB3SLDN795G1MHRL3QO/ke17ZwdGBToddI8pDm48kDeESvmL2CKgnnReSL1AEHUUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcEylu32zKYtOLotM52LGWiEW_ujpOfeZDdgBF7lQwn1cawo3Xn5c5CO-uPn_aeMmA/workflow22.png?format=2500w" width="550" title="DLC-live GUI" alt="DLC LIVE! GUI" align="right" vspace = "50">


The standard documents include all changes to the package to run DLC, but here is an extended version compiled for using DeepLabCut 2.2+ with multiple-animals. 

Note, we STRONGLY encourage you to use the [Project Manager GUI](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/PROJECT_GUI.md) when you first start using multi-animal mode. Each tab is customized for multi-animal when you create or 
load a multi-animal project. As long as you follow the recommendations within the GUI, you should be good to do! 

This document should serve as an **advanced user guide for maDLC,**
and it is here to support the scientific advances presented in the preprint (Mathis et al, 2020).

# Install and test: 

Install DLC as noted elsewhere, then run the test script found here (you will need to git clone first):
https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/testscript_multianimal.py
```python
python testscript_multianimal.py
```

# Get started in the terminal:

Start iPython, or if you are using MacOS, you must use ``pythonw`` vs. typing ``ipython`` or ``python``, but otherwise it's the same.
If you are using DeepLabCut on the cloud, you cannot use the GUIs and you need to first set `DLClight=True`. If you use Windows, please always open the terminal with administrator privileges. Please read more [here](https://github.com/DeepLabCut/Docker4DeepLabCut2.0), and in our Nature Protocols paper [here](https://www.nature.com/articles/s41596-019-0176-0). And, see our [troubleshooting wiki](https://github.com/AlexEMG/DeepLabCut/wiki/Troubleshooting-Tips).

Open an ``ipython`` session and import the package by typing in the terminal:
```python
ipython
import deeplabcut
```
**TIP:** for every function there is a associated help document that can be viewed by adding a **?** after the function name; i.e. ``deeplabcut.create_new_project?``. To exit this help screen, type ``:q``.


### Create a New Project:

```python
deeplabcut.create_new_project('ProjectName','YourName', ['/usr/FullPath/OfVideo1.avi', '/usr/FullPath/OfVideo2.avi', '/usr/FullPath/OfVideo1.avi'],
              copy_videos=True, multianimal=True)
```
Tip: if you want to place the project folder somewhere please pass : ``working_directory = 'FullPathOftheworkingDirectory'``

- Note, if you are a Ubuntu user the path should look like: ``['/home/username/yourFolder/video1.mp4']``; if you are a Windows user, it should look like: ``[r'C:\username\yourFolder\video1.mp4']``
- Note, you can also put ``config_path = `` in front of the above line to create the path to the config.yaml that is used in the next step, i.e. ``config_path=deeplabcut.create_project(...)``)
    - If you do not, we recommend setting a variable so this can be easily used! Once you run this step, the conig_path is printed for you once you run this line, so set a variable for ease of use, i.e. something like:
```python
config_path = '/thefulloutputpath/config.yaml'
```
 - just be mindful of the formatting for Windows vs. Linux, see below.


(more details [here](functionDetails.md#a-create-a-new-project))

### Configure the Project:

- open the **config.yaml** file (in a text editor (like atom, gedit, vim etc.)), which can be found in the subfolder created when you set your project name, to change parameters and identify label names! This is a crucial step.

**maDeepLabCut**: As of 2.2 you also **must** modify a few more parameters in the `config.yaml` file, **see [here](functionDetails.md#b-configure-the-project-).**

(PLEASE see more details [here](functionDetails.md#b-configure-the-project))

- set the config_path (LINUX):
```python
config_path = '/home/computername/DeepLabCut/yourprojectname/config.yaml'
```
- set the config_path (WINDOWS):
```python
config_path = r'C:\home\computername\DeepLabCut\yourprojectname\config.yaml'
```

- or you already set it as a variable in step "create_new_project" directly above.

**mini-demo:** create project and edit the yaml file

<p align="center">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/startdeeplabcut.gif" width="90%">
</p>


### Select Frames to Label:

```python
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', crop=True/False)
```

(more details [here](functionDetails.md#c-data-selection)) *update: as of 2.0.5 (spring 2019) ``checkcropping=True`` is dropped; you now just have the option to directly draw a rectangle over the image to crop before extraction (i.e. there no need to manually change in config.yaml then check).

### Label Frames:

```python
deeplabcut.label_frames(config_path)
```

**maDeepLabCut**: As of 2.2 there is a new multi-animal labeling GUI (as long as in your `config.yaml` says `multianimalproject: true` at the top, this will automatically launch).

(more details [here](functionDetails.md#d-label-frames))

**mini-demo:** using the GUI to label

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c535f16fa0d60294ac39e4e/1548967721381/GUIdemo1.gif?format=750w" width="60%">
</p>

### Check Annotated Frames:

```python
deeplabcut.check_labels(config_path)
```
**maDeepLabCut**: you can also look at both bodypart labeling (standard) and individual IDs by also passing `visualizeindividuals=True`

(more details [here](functionDetails.md#e-check-annotated-frames))

- Note, we also have a new, optional, functional to crop frames /labels for more efficient training. You can call this before you create a training dataset by:
```python
deeplabcut.cropimagesandlabels(path_config_file, userfeedback=False)
```
#### Reminder: Build your skeleton connections before you create a training set!

If you did not do this already be sure to define a skeleton in the `config.yaml` - See [more here for cruical details](functionDetails.md#b-configure-the-project-).

There is also a graphical way to define your skeleton:
```python
deeplabcut.SkeletonBuilder(config_path)
```
<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1589410182515-9SJO9MML6CNCXBAWQ6Z6/ke17ZwdGBToddI8pDm48kJ1oJoOIxBAgRD2ClXVCmKFZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpxBw7VlGKDQO2xTcc51Yv6DahHgScLwHgvMZoEtbzk_9vMJY_JknNFgVzVQ2g0FD_s/ezgif.com-video-to-gif+%2811%29.gif?format=750w" width="80%">
</p>

### Create Training Dataset:

```python
deeplabcut.create_training_dataset(config_path)
```
or to [compare different neural networks](/wiki/What-neural-network-should-I-use%3F) use:
```python
deeplabcut.create_training_model_comparison(config_path, num_shuffles=1, net_types=['resnet_50'], augmenter_types=['default', 'imgaug'] )
```
:movie_camera:[VIDEO TUTORIAL AVAILABLE!](https://www.youtube.com/watch?v=WXCVr6xAcCA)


**maDeepLabCut**:

For mutli-animal training we use batch processing. This means that we'd like the data to be similarly sized. You can of course have differing size of images you label (and we suggest cropping out useless pixels!). So, we have a new function that can pre-process your data to be compatible with batch training. Please run this function before you `create_multianmialtraining_dataset`:

```python
deeplabcut.cropimagesandlabels(path_config_file)
```
Then run:
```python
deeplabcut.create_multianimaltraining_dataset(path_config_file, allow_growth=True)
```
(more details [here](functionDetails.md#f-create-training-datasets))

### Train The Network:

```python
deeplabcut.train_network(config_path)
```

(more details [here](functionDetails.md#g-train-the-network))

### Evaluate the Trained Network:

Here, for traditional projects you will get a pixel distance metric and you should inspect the individual frames:
```python
deeplabcut.evaluate_network(config_path, plotting=True)
```
:movie_camera:[VIDEO TUTORIAL AVAILABLE!](https://www.youtube.com/watch?v=bgfnz1wtlpo)

**maDeepLabCut [CRITICAL POINT]:**

You need to **cross validate parameters** before inference. Here, you will run the new function:
```python
deeplabcut.evaluate_multianimal_crossvalidate(config_path, Shuffles=[1], edgewisecondition=True, leastbpts=1, init_points=20, n_iter=50)
```
We highly suggest that you read the docstring for this function to edit inputs appropriately.

([Here is more information](functionDetails.md#cross-validation-of-inference-parameters-a-madeeplabcut-critical-point))


**maDeepLabCut**: (or on normal projects!)

You can also plot the scoremaps, locref layers, and PAFs:
```python
deeplabcut.extract_save_all_maps(path_config_file, shuffle=shuffle, Indices=[0, 5])
```
- you can drop "Indices" to run this on all training/testing images (this is slow!)

(more details [here](functionDetails.md#h-evaluate-the-trained-network))

### Video Analysis:
- Please note that **novel videos DO NOT need to be added to the config.yaml file**. You can simply have a folder elsewhere on your computer and pass the video folder (then it will analyze all videos of the specified type (i.e. ``videotype='.mp4'``), or pass the path to the **folder** or exact video(s) you wish to analyze:

```python
deeplabcut.analyze_videos(config_path,['/fullpath/project/videos/'], videotype='.mp4', save_as_csv = True)
```
**maDeepLabCut**: there is a new step that allows you to plot *all* detections first. This allows you to check the pose-estimation quality before tracking of individuals! We recommend doing this step when you are running quality checks on new videos, etc. Once you have optimized pose-estimation and tracking, this is not required. `scorername` can be gotten from `scorername = deeplabcut.analyze_videos (...)` or just looking at the name of the DLC scorer in the folder name, h5 file, etc.

```python
deeplabcut.create_video_with_all_detections(path_config_file, ['videofile_path'], scorername)
```

### Assemble & Refine Tracklets in maDeepLabCut:

:movie_camera:[VIDEO TUTORIAL AVAILABLE!](https://youtu.be/bEuBKB7eqmk)
- Now that you have detections (which are saved as a pickle file, not h5, btw), we need to assemble and track the animals.

First, you need to convert detections to tracklets. This step has several tracker types (`track_method`), and we recommend testing which one works best on your data.

```python
deeplabcut.convert_detections2tracklets(path_config_file, ['videofile_path'], videotype='mp4',
                                                    shuffle=1, trainingsetindex=0, track_method='')
```
You should **cross-validate** the tracking parameters. ([Here is more information](functionDetails.md#cross-validation-of-inference-parameters-a-madeeplabcut-critical-point)). Namely, you can iteratively change the parameters, run `convert_detections2tracklets` then load them in the GUI (`refine_tracklets`). Note, that in the main Project Manager GUI there is a button for you to launch the inference file to seemlessly edit and rapidly test.

Secondly, you need to **refine the tracklets**. You can fix both "major" ID swaps, i.e. perhaps when animals cross, and you can micro-refine the individual body points. You will load the `...trackertype.pickle` file that was created above, and then you can launch a GUI to interactively refine the data. This also has several options, so please check out the docstring. Upon saving the refined tracks you get an `.h5` file (akin to what you might be used to from standard DLC. You can also load (1) filter this to take care of small jitters, and (2) load this `.h5` this to refine (again) in case you find another issue, etc!

```python
deeplabcut.refine_tracklets(path_config_file, pickle_or_h5_file, videofile_path, min_swap_len=2, min_tracklet_len=2, trail_len=50)
```
[Read more here!](functionDetails.md#madeeplabcut-critical-point---assemble--refine-tracklets)

Short demo:  
 <p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1588690928000-90ZMRIM8SN6QE20ZOMNX/ke17ZwdGBToddI8pDm48kJ1oJoOIxBAgRD2ClXVCmKFZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpxBw7VlGKDQO2xTcc51Yv6DahHgScLwHgvMZoEtbzk_9vMJY_JknNFgVzVQ2g0FD_s/refineDEMO.gif?format=750w" width="70%">
</p>

### Once you have analyzed video data (and refined your maDeepLabCut tracklets):

Firstly, Here are some tips for scaling up your video analysis, including looping over many folders for batch processing: https://github.com/AlexEMG/DeepLabCut/wiki/Batch-Processing-your-Analysis

You can also filter the predicted bodyparts by:
```python
deeplabcut.filterpredictions(config_path,['/fullpath/project/videos/reachingvideo1.avi'])
```
Note, this creates a file with the ending filtered.h5 that you can use for further analysis. This filtering step has many parameters, so please see the full docstring by typing: ``deeplabcut.filterpredictions?``

### Plotting Results:

- **NOTE :bulb::mega::** Before you create a video, you should set what threshold to use for plotting. This is set in the `config.yaml` file as `pcutoff` - if you have a well trained network, this should be high, i.e. set it to `0.8` or higher!


- You can also determine a good `pcutoff` value by looking at the likelihood plot created during `plot_trajectories`:

Plot the outputs:
```python
deeplabcut.plot_trajectories(config_path,['/fullpath/project/videos/reachingvideo1.avi'],filtered = True)
```

Create videos:
```python
deeplabcut.create_labeled_video(config_path, [`/analysis/project/videos/reachingvideo1.avi','/fullpath/project/videos/reachingvideo2.avi'],filtered = True)
```

(more details [here](functionDetails.md#i-video-analysis-and-plotting-results))

### [optional] Active Learning --> Network Refinement - extract outlier frames from a video:

```python
deeplabcut.extract_outlier_frames(config_path,['full/videofile_path'])
```

(more details [here](functionDetails.md#j-refinement-extract-outlier-frames))

### [optional] Refinement of the labels with our GUI:
(refinement and augmentation of the training dataset)

```python
deeplabcut.refine_labels(config_path)
```

**mini-demo:** using the refinement GUI, a user can load the file then zoom, pan, and edit and/or remove points:

<p align="center">
<img src="http://www.people.fas.harvard.edu/~amathis/dlc/refinelabels.gif" width="90%">
</p>

When done editing the labels, merge:
**PRO TIP:** if you added new data, even without refining, i.e. you added and labeled frames from new videos,  also use merge before creating a new training data set!

```python
deeplabcut.merge_datasets(config_path)
```
Now, create a new training set and re-train (same steps as above)! i.e.
```python
deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(config_path)
```

(more details [here](functionDetails.md#k-refine-labels-augmentation-of-the-training-dataset))


### HELP:

In ipython/Jupyter notebook:

```
deeplabcut.nameofthefunction?
```

In python or pythonw:

```
help(deeplabcut.nameofthefunction)
```

### Tips for "daily" use:

<p align="center">
<img src= https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5ccc5abe0d9297405a428522/1556896461304/howtouseDLC-01.png?format=1000w width="80%">
 </p>

You can always exit an conda environment and easily jump back into a project by simply:

Linux/MacOS formatting example:
```
source activate yourdeeplabcutEnvName
ipython or pythonw
import deeplabcut
config_path ='/home/yourprojectfolder/config.yaml'
```
Windows formatting example:
```
activate yourdeeplabcutEnvName
ipython
import deeplabcut
config_path = r'C:\home\yourprojectfolder\config.yaml'
```

Now, you can run any of the functions described in this documentation.
