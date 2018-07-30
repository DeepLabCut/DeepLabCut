# Test the DeepLabCut Toolbox:

- After installing the required packages, and/or the Docker Container, follow the following steps to check your installation and run the demo. 

- There is a supplied demo video, a mouse reaching from [Mathis et al., 2017](http://www.cell.com/neuron/fulltext/S0896-6273(17)30157-5). We have put on a few test labels just to provide a pre-labeled dataset. Also note that this demo data contains so few labeled frames that one should not train the network (other than for brief testing) on the corresponding data set and expect it to work properly - it is only for demo purposes. As we have already labeled the frames, you will NOT run code from sections **(0)**, **(1)**, or **(2)** below, but we will explain what these steps do. 

# Using the Toolbox code - Labeling and Training Instructions:

 - The following steps document using the code with either Python scripts or in Jupyter Notebooks:

**(0) Configuration of your project:**

DEMO users: familiarize yourself with the folder structure, and the **myconfig.py** file first. You can edit this document outside of Python (such as in getit, in Ubuntu). The **"myconfig.py"** file and set the global variables for your dataset. (Demo users, don't edit this if you want to test on the supplied video)

**(1) Selecting data to label:** 

DEMO users: don't run this code, as it would create new images for you to label. This code, however, will be used to select random frames from your own videos in the near future. In the folder "Generating_a_Training_Set", the provided code allows you to select a subset of frames in a video(s) for labeling. Make sure videos you want to use for the training set are in a sub-folder under **"Generating_a_Training_Set"** or change the video path accordingly in **"myconfig.py"**. (DEMO users, this is already generated). 

Generally speaking, one should **create a training set that reflects the diversity of the behavior with respect to postures, animal identities, etc.** of the data that will be analyzed. The *provided code randomly selects frames from the videos in a temporally uniformly distributed way*. This is fine when the postures vary accordingly. However, the behavior might be sparse (as in the case of reaching, where the reach and pull is very fast and the mouse is not moving much between trials). However, one can extract various example videos of different pulls, then this code will sample the behavior well. One should take this into account when selecting frames to label (i.e. because you can label so little data, be sure your selected frames capture the full breadth of the behavior. You may want to additionally hand select extra frames of interest). *If this extraction code is suboptimal given your behavior, consider extracting training frames by different means.* We will also provide additional code in future realeases. 

If you want to check out the code: 

 - **IDE users (such as Spyder):** open "Step1_SelectRandomFrames_fromVideos.py" and crop videos if behavior of interest only happens in subset of frame (see Step1_SelectRandomFrames_fromVideos.py for detailed instructions; edit in Spyder or your favorite integrated development environment (IDE) an run the script). 
            
   - **Juypter Users:** use the Step1_.._demo.ipynb file* - In general, the supplied Jupyter Notebook is helpful to optimize the video cropping step.
   
   The output from this step will be a set of **.png** files for you to then label. 
   
   NOTE: If you use another tool to select frames, your images will need to follow the format **"img001.png"**, where the length of the numbers are consistant, i.e.  001, 002, ... 215, but not 1, 2, ... 215 (for example).
            
**(2) Label the frames:**

DMEO users: you will not run this step, but this how we currently recommend you label your own data frames in the future. 

   - As of the current release, you need to use **.png** files that start with **img** to create the training set (This is not a limitation of the toolbox, it is just hard coded, which you can change in the base code if you wish). 
    
   - You should label a sufficient number of frames with the anatomical locations of your choice. For the behaviors we have tested so far, 100-200 frames gave good results (see preprint). Depending on your required accuracy and the nature of the scene statistics more training data might be necessary. Try to consistently label similar spots (e.g. on a wrist that is very large, try to label the same location).
     
   - Labeling can be done in any program, but we recommend using [Fiji](https://fiji.sc/). In Fiji one can simply open the images, create a (virtual) stack* (in brief, in fiji: File > Import > Image Sequence > (check "virtual stack")), then use the "Multi-point Tool" to label frames. You scroll through the frames and click on as many points as you wish in the same order on each frame. Then simply measure and save the resulting .csv file (Analyze>Measure (or simple Ctrl+M)). 

   - You can either store one .csv file per folder that contains all body parts in a cyclical way (same, repeating order). If a particular body part is not visible in a frame, then click close to (0,0) to later exclude those labels (see description in `myconfig.py` for details). In this case, set `multibodypartsfile=True` and put the name of the corresponding csv file under `multibodypartsfilename` in `myconfig.py`. Furthermore, make sure that the sequence of body parts has exactly the same order as the cyclically labeled body parts. Alternatively, you can put one csv file per body part and store them with the names defined in `bodypart` list of `myconfig.py`. In this case set `multibodypartsfile=False`.
     
   *To open virtual stack see: https://imagej.nih.gov/ij/plugins/virtual-opener.html  The virtual stack is helpful when the images have different sizes. This way they are not rescaled and the label information does not need to be rescaled.

<p align="center">
<img src="images/img0000_labels.jpg" width="40%">
</p>

**(3) Formatting the data I:**

DEMO users: you will run the following code!

  - **IDE users:** the code "Step2_ConvertingLabels2DataFrame.py" creates a data structure in [pandas](https://pandas.pydata.org/) (stored as .h5 and .csv) combining the various labels together with the (local) file path of the images. This data structure also keeps track of who labeled the data and allows to combine data from multiple labelers. 

Keep in mind that ".csv" files for each bodypart or multiple bodyparts listed in the myconfig.py file should exist in the folder alongside the individual images.

   - **Juypter Users:** use the Step2_.._demo.ipynb file
   
**(4) Checking the formatted data:**

 DEMO users: you will run the following code!
 
 After this step, you may **check** if the data was loaded correctly and all the labels are properly placed (Use "Step3_CheckLabels.py").
 
   - **Juypter Users:** use the Step3_.._demo.ipynb file

**(5) Formatting the data II:** 

DEMO users: you will run the following code!

Next split the labeled data into test and train sets for benchmarking ("Step4_GenerateTrainingFileFromLabelledData.py"). This step will create a ".mat" file, which is used by DeeperCut as well as a ".yaml" file containing meta information with regard to the parameters of the DeeperCut. Before this step consider changing the parameters in 'pose_cfg.yaml'.  This file also contains short descriptions of what these parameters mean. Generally speaking pos_dist_thresh and global_scale will be of most importance. Then run the code. This file will create a folder with the training data as well as a folder for training the corresponding model in DeeperCut. 

   - **IDE users:** run the code; if you wish to do so in the terminal: `python3 Step4_GenerateTrainingFileFromLabelledData.py`

   - **Juypter Users:** use the Step4_.._demo.ipynb file

   - The output will be two folders for train and test data (with their respective yaml files)

 **(6) Training the deep neural network:**
 
 DEMO users: you will run the following code!
 
The folder pose-tensorflow contains an earlier, minimal yet sufficient variant of [DeeperCut](https://github.com/eldar/pose-tensorflow). Before training a model for the first time you need to download the weights for the [ResNet pretrained on ImageNet from tensorflow.org](https://github.com/tensorflow/models/tree/master/official/resnet) (~200MB). To do that: 
    
     $ cd pose-tensorflow/models/pretrained
     $ ./download.sh
    
Next copy the two folders generated in step **(5) Formatting the data II** into the **models** folder of pose-tensorflow (i.e. pose-tensorflow/models/). We have already done this for the demo, which you will find there. To transfer (in a terminal):


	$ cp -r YOURexperimentNameTheDate-trainset95shuffle1 ../pose-tensorflow/models/
	$ cp -r UnaugmentedDataSet_YOURexperimentNameTheDate ../pose-tensorflow/models/

Then (in a terminal) navigate to the subfolder "train" of the machine file, i.e. in our case

	 $ cd pose-tensorflow/models/reachingJan30-trainset95shuffle1/train
	 
and then start training (good luck!)
    
     $ TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py 

If your machine has multiple GPUs, you can select which GPU you want to run 
on by setting the environment variable, eg. CUDA_VISIBLE_DEVICES=0. If you are running the code on a CPU you can ommit that command (also in step 7).

Tips: You can also stop during a training (Ctrl-C), and restart from a snapshot (aka checkpoint):
Just change the init_weights term, i.e. instead of "init_weights: ../../pretrained/resnet_v1_50.ckpt"  put "init_weights: ./snapshot-insertthe#ofstepshere" (i.e. 10,000). We recommend training for **thousands** of iterations until the loss plateaus (typically >200,000 - see Fig 2 and Supp Fig 2 in our [paper](https://arxiv.org/pdf/1804.03142.pdf)). How often the loss is displayed and how often the weights are stored can be changed in the pose_cfg.yaml (variables: `display_iters: 1,000`, and `save_iters: 50,000` respectively). Note, DEMO users, you must at least train to 50,000 to save any snapshots as it is set; you can change `save_iters: 50,000` to a lower value, if you wish!

**(7) Evaluate your network:**

DEMO users: you will run the following code!

In the folder "Evaluation-tools", you will find code to evaluate the performance of the trained network on the whole data set (train and test images).

     $ CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py #to evaluate your model [needs TensorFlow]
     $ python3 Step2_AnalysisofResults.py  #to compute test & train errors for your trained model

You can then check the labeled test (and training images), which will be created by this code as well as the root mean square error distance (RMSE) between your labels and the ones by DeepLabCut. Ideally DeepLabCut labeled unseen (test images) according to your required accuracy and the average train and test (RMS-)errors are comparable (good generalization). What (numerically) comprises an acceptable RMSE depends on many things (including the size of the tracked body parts, your labeling variablity etc.). You can visually inspect if the distance between the labeled bodyparts is acceptable (see Results folder). Note that the test error can also be larger than the training error due to human variablity (in labeling, see Fig 2 in our [paper](https://arxiv.org/pdf/1804.03142.pdf)). 

If you set various parameters for the plotting of predicted and human annotator labels in ```myconfig.py``` (i.e. colormap, scale, marker size msize, transparency of labels alphavalue). The plots can also be customized by editing the ```MakeLabeledImage``` function inside the [analysis script](https://github.com/AlexEMG/DeepLabCut/blob/master/Evaluation-Tools/Step2_AnalysisofResults.py). Each body part is plotted in a different color and the plot labels indicate their source. Note that by default the human labels are plotted as plus ('+'), DeepLabCut's predictions either as '.' (for confident predictions with likelihood > pcutoff) and 'x' for (likelihood <= pcutoff). The RMSE output is also broken down into RMSE for all pairs or only likely pairs (>pcutoff). This helps for excluding occluded body parts etc. One of the strenghts of DeepLabCut is that due to the probabilistic output of the scoremap, it can, if sufficiently trained, also reliably report if a body part is visible in a given frame (see discussions of [finger tips in reaching or legs in Drosophila behavior in our paper](https://arxiv.org/pdf/1804.03142.pdf))

If the generalization is not good, you might want to a) make sure that the loss was already converged, b) consider labeling additional images, c) check if the labels were imported correctly  ... 

 **(8) Run the trained network on videos and analyze results**
 
After successfully training the network and finding a low generalization error on test images, you can extract body parts/poses from other videos. See [Analysis tools](analysis-tools.md) for details.

Return to [readme](../README.md).
