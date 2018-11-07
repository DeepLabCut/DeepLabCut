### (A) Create a New Project


The function **create\_new\_project** creates a new project directory, required subdirectories, and a basic project configuration file. Each project is identified by the name of the project (e.g. Reaching), name of the experimenter (e.g. YourName), as well as the date at creation.

Thus, this function requires the user to input the enter the name of the project, the name of the experimenter, and the full path of the videos that are (initially) used to create the training dataset.  

Optional arguments specify the working directory, where the project directory will be created, and if the user wants to copy the videos (to the project directory). If the optional argument working\_directory is unspecified, the project directory is created in the current working directory, and if copy\_videos is unspecified symbolic links for the videos are created in the videos directory. Each symbolic link creates a reference to a video and thus eliminates the need to copy the entire video to the video directory (if the videos remain at that original location). 


          deeplabcut.create_project(`Name of the project',`Name of the experimenter', [`Full path of video 1',`Full path of video2',`Full path of video3'], working_directory=`Full path of the working directory',copy_videos=True/False) 
          
 (TIP: you can also place ``config_path`` in front of ``deeplabcut.create_project`` to create a vriable that holds the path to the config.yaml file, i.e. ``config_path=deeplabcut.create_project(...)``)

This set of arguments will create a project directory with the name **Name of the project+name of the experimenter+date of creation of the project** in the **Working directory** and creates the symbolic links to videos in the **videos** directory. The project directory will have subdirectories: **dlc-models**, **labeled-data**, **training-datasets**, and **videos**.  All the outputs generated during the course of a project will be stored in one of these subdirectories, thus allowing each project to be curated in separation from other projects. The purpose of the subdirectories is as follows:

**dlc-models:** This directory contains the subdirectories *test* and *train*, each of which holds the meta information with regard to the parameters of the feature detectors in configuration file. The configuration files are YAML files, a common human-readable data serialization language. These files can be opened and edited with standard text editors. The subdirectory *train* will store checkpoints (called snapshots in TensorFlow) during training of the model. These snapshots allow the user to reload the trained model without re-training it, or to pick-up training from a particular saved checkpoint, in case the training was interrupted.

**labeled-data:** This directory will store the frames used to create the training dataset. Frames from different videos are stored in separate subdirectories. Each frame has a filename related to the temporal index within the corresponding video, which allows the user to trace every frame back to its origin.

**training-datasets:**  This directory will contain the training dataset used to train the network and metadata, which contains information about how the training dataset was created.  

**videos:** Directory of video links or videos. When **copy\_videos** is set to ``False``, this directory contains symbolic links to the videos. If it is set to ``True`` then the videos will be copied to this directory. The default is ``False``. Additionally, if the user wants to add new videos to the project at any stage, the function **add\_new\_videos** can be used. This will update the list of videos in the project's configuration file. 

       deeplabcut.add_new_videos(`Full path of the project configuration file*',[`full path of video 4', `full path of video 5'],copy_videos=True/False)

*Please note, *Full path of the project configuration file* will be referenced as ``config_path`` throughout this protocol.

The project directory also contains the main configuration file called *config.yaml*. The *config.yaml* file contains many important parameters of the project. A complete list of parameters including their description can be found in Box1. 

The ``create a new project`` step writes the following parameters to the configuration file: *Task*, *scorer*, *date*, *project\_path* as well as a list of videos *video\_sets*. The first three parameters should **not** be changed. The list of videos can be changed by adding new videos or manually removing videos.

<p align="center">
<img src="/docs/images/Box1.png" width="90%">
</p>

### (B) Configure the Project

Next, open the **config.yaml** file, which was created during  **create\_new\_project**. You can edit this file in any text editor.  Familiarize yourself with the meaning of the parameters (Box 1). You can edit various parameters, in particular add the list of *bodyparts* (or points of interest) that you want to track. For the next data selection step *numframes2pick*, *start*, *stop*, *x1, x2, y1, y2* and *cropping* are of major importance.

 # (C) Data Selection
 
CRITICAL: A good training dataset should consist of a sufficient number of frames that capture the
full breadth of the behavior. This implies to select the frames from different (behavioral) sessions and different animals,
if those vary substantially (to train an invariant, robust feature detector). Thus, a good training dataset should reflect
the diversity of the behavior with respect to postures, luminance conditions, background conditions, animal identities,
etc. of the data that will be analyzed. For the behaviors we have tested so far, a data set of 100−200 frames gave good
results [11]. However, depending on the required accuracy and the nature of the scene statistics, more or less frames
might be necessary to create the training data set. Ultimately, in order to scale up the analysis to large collections
of videos with perhaps unexpected conditions, one can also refine the data set in an adaptive way (see refinement below).

The function extract_frames extracts the random frames from all the videos in the project configuration file in
order to create a training dataset. The extracted frames from all the videos are stored in a separate subdirectory
named after the video file’s name under the ‘labeled-data’. This function also has various parameters that might be
useful based on the user’s need.

          deeplabcut.extract_frames(‘config_path’,‘automatic/manual’,‘uniform/kmeans’, crop=True/False, checkcropping=True)

CRITICAL POINT: It is advisable to keep the frame size small, as large frames increase the training and
inference time. The cropping parameters for each video can be provided in the config.yaml file (and see below).
When running the function extract_frames, if the parameter crop=True and checkcropping=True, then it will crop
the frames to the size provided in the config.yaml file, and the user can first check the bounding box of the cropping.
Upon calling extract_frames a image will pop up with a red bounding box based on the crop parameters so that
the user can check those parameters. Once the user closes the pop-up window, they will be asked if the cropping is
correct. If yes, then the frames are extracted accordingly. If not, the cropping parameters can be iteratively adjusted
based on this graphical feedback before proceeding.

The provided function either selects frames from the videos in a randomly and temporally uniformly distributed
way (uniform), by clustering based on visual appearance (k-means), or by manual selection. Random
selection of frames works best for behaviors where the postures vary across the whole video. However, some behaviors
might be sparse, as in the case of reaching where the reach and pull are very fast and the mouse is not moving much
between trials. In such a case, the function that allows selecting frames based on k-means derived quantization would
be useful. If the user chooses to use k-means as a method to cluster the frames, then this function downsamples the
video and clusters the frames using k-means, where each frame is treated as a vector. Frames from different clusters
are then selected. This procedure makes sure that the frames look different. However, on large and long videos, this
code is slow due to computational complexity.

CRITICAL POINT: It is advisable to extract frames from a period of the video that contains interesting
behaviors, and not extract the frames across the whole video. This can be achieved by using the start and stop
parameters in the config.yaml file. Also, the user can change the number of frames to extract from each video using
the numframes2extract in the config.yaml file.

However, picking frames is highly dependent on the data and the behavior being studied. Therefore, it is hard to
provide all purpose code that extracts frames to create a good training dataset for every behavior and animal. If the user feels specific frames are lacking, they can extract hand selected frames of interest using the interactive GUI
provided along with the toolbox. This can be launched by using:

          >> deeplabcut.extract_frames(‘config_path’,‘manual’)
          
The user can use the *Load Video* button to load one of the videos in the project configuration file, use the scroll
bar to navigate across the video and *Grab a Frame* to extract the frame. The user can also look at the extracted
frames and e.g. delete frames (from the directory) that are too similar before re-loading the set and then manually
annotating them.

### (D) Label Frames

The toolbox provides a function **label_frames** which helps the user to easily label all the extracted frames using
an interactive graphical user interface (GUI). The user should have already named the body parts to label (points of
interest) in the project’s configuration file by providing a list. The following command invokes the labeling toolbox.

          >> deeplabcut.label_frames(‘config_path’)
          
The user needs to use the *Load Frames* button to select the directory which stores the extracted frames from one of
the videos. Subsequently, the user can use one of the radio buttons (top right) to select a body part to label. The
label will be plotted as a dot in a unique color (see Figure 4 for more details).

CRITICAL POINT: It is recommended to finalize the position of the selected label before changing the dot
size for the next labels.

The user is free to move around the body part and once satisfied with its position, can select another radio button
(in the top right) to switch to the respective body part. Once the user starts labeling a subsequent body part,
preceding labels of the body parts can no longer be moved. The user can skip a body part if it is not visible. Once all the visible body parts are labeled, then the user can use ‘Next Frame’ to load the following frame. The user needs
to save the labels after all the frames from one of the videos are labeled by clicking the save button at the bottom
right. Saving the labels will create a labeled dataset for each video in a hierarchical data file format (HDF) in the
subdirectory corresponding to the particular video in **labeled-data**.

CRITICAL POINT: It is advisable to consistently label similar spots (e.g. on a wrist that is very large, try
to label the same location). In general, invisible or occluded points should not be labeled by the user. They can
simply be skipped by not applying the label anywhere on the frame.

OPTIONAL: In an event of adding more labels to the existing labeled dataset, the user need to append the new
labels to the bodyparts in the config.yaml file. Thereafter, the user can call the function **label_frames** and check
the left lower tick box, *Add new labels to existing dataset?* before loading the frames. Saving the labels after all the
images are labelled will append the new labels to the existing labeled dataset.

###  (E) Check Annotated Frames

OPTIONAL: Checking if the labels were created and stored correctly is beneficial for training, since labeling
is one of the most critical parts for creating the training dataset. The DeepLabCut toolbox provides a function
‘check_labels’ to do so. It is used as follows:

          >> deeplabcut.check_labels(‘config_path’)
          
For each video directory in labeled-data this function creates a subdirectory with **labeled** as a suffix. Those directories contain the frames plotted with the annotated body parts. The user can double check if the body parts are labeled correctly. If they are not correct, the user can call the refinement GUI (see below, and check the tick box for ``adjust original labels`` to adjust the location of the labels).

### (F) Create Training Dataset

Combining the labeled datasets from all the videos and splitting them will create train and test datasets. The
training data will be used to train the network, while the test data set will be used for evaluating the network. The
function **create_training_dataset** performs those steps.

          >> deeplabcut.create_training_dataset(‘config_path’,num_shuffles=1)
          
The set of arguments in the function will shuffle the combined labeled dataset and split it to create train and test
sets. The subdirectory with suffix ``iteration#`` under the directory **training-datasets** stores the dataset and meta
information, where the ``#`` is the value of ``iteration`` variable stored in the project’s configuration file (this number
keeps track of how often the dataset was refined).

OPTIONAL: If the user wishes to benchmark the performance of the DeepLabCut, they can create multiple
training datasets by specifying an integer value to the num_shuffles.

Each iteration of the creation of a training dataset, will create a ``.mat`` file, which is used by the feature detectors
and a ``.pickle`` file which contains the meta information about the training dataset. This also creates two subdirectories
within **dlc-models** called ``test`` and ``train``, and these each have a configuration file called pose_cfg.yaml.
Specifically, the user can edit the **pose_cfg.yaml** within the train subdirectory before starting the training. These
configuration files contain meta information with regard to the parameters of the feature detectors. Key parameters
are listed in Box 2.

<p align="center">
<img src="/docs/images/Box2.png" width="90%">
</p>

###  (G) Train The Network

Timing: The time required to train the network mainly depends on the frame size of the dataset and the
computer hardware. On a NVIDIA GeForce GTX 1080 Ti GPU, it takes ≈ 6 hrs to train the network for at least
200,000 iterations. On the CPU, it will take several days to train for the same number of iterations on the same
training dataset.

The function ‘train_network’ helps the user in training the network. It is used as follows:

                    >> deeplabcut.train_network(‘config_path’,shuffle=1)
                    
The set of arguments in the function starts training the network for the dataset created for one specific shuffle.

Example parameters that one can call:

                    train_network(config_path,shuffle=1,trainingsetindex=0,gputouse=None,max_snapshots_to_keep=5,autotune=False,displayiters=None,saveiters=None)

By default, the pre-trained ResNet network is not provided in the DeepLabCut toolbox (as it has around 100MB).
However, if not previously downloaded from the TensorFlow model weights, it will be downloaded and stored in
a subdirectory *pre-trained* under the subdirectory *models* in *Pose_Estimation_Tensorflow*. At user specified
iterations during training checkpoints are stored in the subdirectory *train* under the respective iteration directory.
If the user wishes to restart the training at a specific checkpoint they can specify the full path of the checkpoint to
the variable ``init_weights`` in the **pose_cfg.yaml** file under the *train* subdirectory (see Box 2).

CRITICAL POINT: It is recommended to train for thousands of iterations until the loss plateaus (typically
 around **200,000**). The variables ``display_iters`` and ``save_iters`` in the **pose_cfg.yaml** file allows the user to alter how often the loss is displayed and how often the weights are stored.
  
**Important Parameters:** 

    config : string
        Full path of the config.yaml file as a string.

    shuffle: int, optional
        Integer value specifying the shuffle index to select for training. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    
    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
   
**Additional parameters:**
    
    max_snapshots_to_keep: int, or None. Sets how many snapshots are kept, i.e. states of the trained network. Every savinginteration many times 
    a snapshot is stored, however only the last max_snapshots_to_keep many are kept! If you change this to None, then all are kept. 
    See: https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835
    
    autotune: property of TensorFlow, somehow faster if 'false' (as Eldar found out, see https://github.com/tensorflow/tensorflow/issues/13317). Default: False
    
    displayiters: this variable is actually set in pose_config.yaml. However, you can overwrite it with this hack. Don't use this regularly, just if you are too lazy to dig out 
    the pose_config.yaml file for the corresponding project. If None, the value from there is used, otherwise it is overwritten! Default: None
    
    saveiters: this variable is actually set in pose_config.yaml. However, you can overwrite it with this hack. Don't use this regularly, just if you are too lazy to dig out 
    the pose_config.yaml file for the corresponding project. If None, the value from there is used, otherwise it is overwritten! Default: None

### (H) Evaluate the Trained Network

It is important to evaluate the performance of the trained network. This performance is measured by computing
the mean average Euclidean error (MAE; which is proportional to the average root mean square error) between the
manual labels and the ones predicted by DeepLabCut. The MAE is saved as a comma separated file and displayed
for all pairs and only likely pairs (>p-cutoff). This helps to exclude, for example, occluded body parts. One of the
strengths of DeepLabCut is that due to the probabilistic output of the scoremap, it can, if sufficiently trained, also
reliably report if a body part is visible in a given frame. (see discussions of finger tips in reaching and the Drosophila
legs during 3D behavior in [11]). The evaluation results are computed by typing:

          >> deeplabcut.evaluate_network(‘config_path’,shuffle=[1], plotting=True)

Setting ``plotting`` to true plots all the testing and training frames with the manual and predicted labels. The user
should visually check the labeled test (and training) images that are created in the ‘evaluation-results’ directory.
Ideally, DeepLabCut labeled unseen (test images) according to the user’s required accuracy, and the average train
and test errors are comparable (good generalization). What (numerically) comprises an acceptable MAE depends on
many factors (including the size of the tracked body parts, the labeling variability, etc.). Note that the test error can
also be larger than the training error due to human variability (in labeling, see Figure 2 in Mathis et al, Nature Neuroscience 2018).

The plots can be customized by editing the **config.yaml** file (i.e. the colormap, scale, marker size (dotsize), and
transparency of labels (alphavalue) can be modified). By default each body part is plotted in a different color
(governed by the colormap) and the plot labels indicate their source. Note that by default the human labels are
plotted as plus (‘+’), DeepLabCut’s predictions either as ‘.’ (for confident predictions with likelihood > p-cutoff) and
’x’ for (likelihood <= p-cutoff). 

The evaluation results for each shuffle of the training dataset are stored in a unique subdirectory in a newly created
directory ‘evaluation-results’ in the project directory. The user can visually inspect if the distance between the labeled
and the predicted body parts is acceptable. In the event of benchmarking with different shuffles of same training
dataset, the user can provide multiple shuffle indices to evaluate the corresponding network. If the generalization is
not sufficient, the user might want to:

• check if the labels were imported correctly, i.e. invisible points are not labeled and the points of interest are
labeled accurately 

• make sure that the loss has already converged 

• consider labeling additional images and make another iteration of the training data set 


### (I) Video Analysis and Plotting Results

The trained network can be used to analyze new videos. The user needs to first choose a checkpoint with the best
evaluation results for analyzing the videos. In this case, the user can enter the corresponding index of the checkpoint
to the variable snapshotindex in the config.yaml file. By default, the most recent checkpoint (i.e. last) is used for
analyzing the video. Then, a new video can be analyzed by typing:

          >> deeplabcut.analyze_videos(‘config_path’,[‘/analysis/project/videos/reachingvideo1.avi’],shuffle=1, save_as_csv=True)
          
The labels are stored in a [MultiIndex Pandas Array](http://pandas.pydata.org), which contains the name
of the network, body part name, (x, y) label position in pixels, and the likelihood for each frame per body part. These
arrays are stored in an efficient Hierarchical Data Format (HDF) in the same directory, where the video is stored.
However, if the flag ``save_as_csv`` is set to ``True``, the data can also be exported in comma-separated values format
(.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.; This flag is set to ``False``
by default.

Additionally, the toolbox provides a function to create labeled videos based on the extracted poses by plotting the
labels on top of the frame and creating a video. One can use it as follows to create multiple labeled videos:

          >> deeplabcut.create_labeled_video(‘config_path’,[‘/analysis/project/videos/reachingvideo1.avi’,‘/analysis/project/videos/reachingvideo2.avi’])
          
This function has various parameters, in particular the user can set the ``colormap``, the ``dotsize``, and ``alphavalue`` of the labels in **config.yaml** file.

The plotting components of this toolbox utilizes matplotlib therefore these plots can easily be customized by
the end user. We also provide a function to plot the trajectory of the extracted poses across the analyzed video, which
can be called by typing:

          >> deeplabcut.plot_trajectories(‘config_path’,[‘/analysis/project/videos/reachingvideo1.avi’])
          
### (J) Refinement: Extract Outlier Frames

While DeepLabCut typically generalizes well across datasets, one might want to optimize its performance in various,
perhaps unexpected, situations. For generalization to large data sets, images with insufficient labeling performance
can be extracted, manually corrected by adjusting the labels to increase the training set and iteratively improve the
feature detectors. Such an active learning framework can be used to achieve a predefined level of confidence for all
images with minimal labeling cost (discussed in Mathis et al 2018). Then, due to the large capacity of the neural network that underlies the feature detectors, one can continue training the network with these additional examples. One does not
necessarily need to correct all errors as common errors could be eliminated by relabeling a few examples and then
re-training. A priori, given that there is no ground truth data for analyzed videos, it is challenging to find putative
“outlier frames”. However, one can use heuristics such as the continuity of body part trajectories, to identify images
where the decoder might make large errors. We provide various frame-selection methods for this purpose. In particular
the user can:

• select frames if the likelihood of a particular or all body parts lies below *pbound* (note this could also be due to
occlusions rather then errors).

• select frames where a particular body part or all body parts jumped more than *\uf* pixels from the last frame.

• select frames if the predicted body part location deviates from a state-space model fit to the time series
of individual body parts. Specifically, this method fits an Auto Regressive Integrated Moving Average (ARIMA)
model to the time series for each body part. Thereby each body part detection with a likelihood smaller than
pbound is treated as missing data.  Putative outlier frames are then identified as time points, where the average body part estimates are at least *\uf* pixel away from the fits. The parameters of this method are *\uf*, *pbound*, the ARIMA parameters as well as the list of body parts to average over (can also be ``all``).

All this can be done for a specific video by typing:

          >> deeplabcut.extract_outlier_frames(‘config_path’,[‘videofile_path’])
          
In general, depending on the parameters, these methods might return much more frames than the user wants to
extract (``numframes2pick``). Thus, this list is then used to select outlier frames either by randomly sampling from this
list (``uniform``) or by performing ``k-means`` clustering on the corresponding frames. Furthermore, before this second selection happens, the user is informed about the amount of frames satisfying the criteria and asked if the selection should proceed. This step allows the user to perhaps change the parameters of the frame-selection heuristics first. The user can run the extract_outlier_frames iteratively, and (even) extract additional frames from the same video. Once enough outlier frames are extracted the refinement GUI can be used to adjust the labels based on user feedback (see below).

 # (K) Refine Labels: Augmentation of the Training Dataset
 
 Based on the performance of DeepLabCut, four scenarios are possible:
 
(A) Visible body part with accurate DeepLabCut prediction. These labels do not need any modifications.

(B) Visible body part but wrong DeepLabCut prediction. Move the label’s location to the actual position of the
body part.

(C) Invisible, occluded body part. Remove the predicted label by DeepLabCut with a right click. Every predicted
label is shown, even when DeepLabCut is uncertain. This is necessary, so that the user can potentially move
the predicted label. However, to help the user to remove all invisible body parts the low-likelihood predictions
are shown as open circles (rather than disks).

(D) Invalid images: In an unlikely event that there are any invalid images, the user should remove such an image
and their corresponding predictions, if any. Here, the GUI will prompt the user to remove an image identified
as invalid.

The labels for extracted putative outlier frames can be refined by opening the GUI:

          >> deeplabcut.refine_labels(‘config_path’)
          
This will launch a GUI where the user can refine the labels (Figure 6). Use the ‘Load Labels’ button to select one
of the subdirectories, where the extracted frames are stored. Every label will be identified by a unique color. For
better chances to identify the low-confidence labels, specify the threshold of the likelihood. This changes the body
parts with likelihood below this threshold to appear as circles and the ones above as solid disks while retaining the
same color scheme. Next, to adjust the position of the label, hover the mouse over the labels to identify the specific
body part, left click and drag it to a different location. To delete a specific label, right click on the label (once a label is deleted, it cannot be retrieved).

After correcting the labels for all the frames in each of the subdirectories, the users should merge the data set to
create a new dataset. In this step the iteration parameter in the config.yaml file is automatically updated.

          >> deeplabcut.merge_datasets(‘config_path’)

Once the dataset is merged, the user can test if the merging process was successful by plotting all the labels (Step E).
Next, with this expanded training set the user can now create a novel training set and train the network as described
in Steps F and G. The training dataset will be stored in the same place as before but under a different ``iteration #``
subdirectory, where the ``#`` is the new value of ``iteration`` variable stored in the project’s configuration file (this is
automatically done).

If after training the network generalizes well to the data, proceed to analyze new videos. Otherwise, consider labeling more data.

### Jupyter Notebooks for Demonstration of the DeepLabCut Work-flow

We also provide two Jupyter notebooks for using DeepLabCut on both a pre-labeled dataset, and on the end user’s
own dataset. Firstly, we prepared an interactive Jupyter notebook called run_yourowndata.ipynb that can serve as a
template for the user to develop a project. Furthermore, we provide a notebook for an already started project with
labeled data. The example project, named as Reaching-Mackenzie-2018-08-30 consists of a project configuration file
with default parameters and 20 images, which are cropped around the region of interest as an example dataset. These
images are extracted from a video, which was recorded in a study of skilled motor control in mice. Some example
labels for these images are also provided. See more details [here](/examples).
