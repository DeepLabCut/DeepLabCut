# coding: utf-8

#####################################################################################
# This configuration file sets various parameters for training and evaluating DeepLabCut
#####################################################################################

# myconfig.py:

########################################
# Step 1: Selecting Frames from videos
########################################

Task = 'reaching'

# Filename and path to behavioral video:
videopath = '.'

# Put name of video: / or 'all' to extract frames from all videos in folder.
filename='reachingvideo1.avi'
#filename = 'all' 
videotype='.avi'

checkcropping=False #set "True" if you want to check cropping 
cropping = True #set "True" if you want to crop images with values below

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the cropped region.

x1 = 0
x2 = 640
y1 = 277
y2 = 624

# Portion of the video to sample from in step 1 (In relative terms of video length, i.e. [0,1] is the full video)
start = 0.1
stop =1

# Number of frames to pick
numframes2pick = 22

# Method to pick frames: uniform or kmeans
selectionalgorithm='uniform' 

# the algorithm 'uniform' temporally uniformly sampes frames in interval (start,stop). Visual information within video is irrelevant for this method. This code is very fast and sufficient (to extract distinct frames) when behavioral videos naturally covers many states.

# Alternatively consider using: 'kmeans'

# This code downsamples the video. The video is extracted as a numpy array, which is then 
# clustered by kmeans whereby each frames are treated as a vector. Frames from different clusters are then selected for labeling. This 
# procedure makes sure that the frames "look different", i.e. different postures etc. 
# On large videos this code is slow. Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior that you want DeepLabCut to resolve/analyze (i.e. a reach). 

########################################
# Step 2: Converting frames to pandas array 
########################################

bodyparts = ["hand", "Finger1", "Finger2","Joystick"]  # Exact sequence of labels as were put by
# annotator in *.csv file (for multibodypartsfile). For single files order is irrelevant
Scorers = ['Mackenzie']  # who is labeling?

# Set this true if the data was sequentially labeled and if there is one file per folder (you can set the name of this file below, i.e. multibodypartsfilename)
# Otherwise there should be individual files per bodypart, i.e. in our demo case hand.csv, Finger1.csv etc.
# If true then those files will be generated from Results.txt
multibodypartsfile=False 
multibodypartsfilename="Results.csv"

# When importing the images and the labels in the csv/xls files should be in the same order!
# During labeling in Fiji one can thus (for occluded body parts) click in the origin of the image 
#(i.e. top left corner (close to 0,0)), these "false" labels will then be removed. To do so set the following variable:
#set this to 0 if no labels should be removed!

invisibleboundary=10 # If labels are closer to origin than this number they are set to NaN (not a number). Please adjust to your situation. Units in pixel.
 
imagetype=".png" # image type of extracted frames (do not change if you used our step1). 
# If you started from already extracted frames in a different format then change the format here (for step2 to 4).

########################################
# Step 3: Check labels / makes plots
########################################

colormap = 'cool' #set color map, i.e. viridis, cool, hsv
scale = 1  # scale for plotting
msize=10   # size of labels
alphavalue =.6 #transparency of labels

########################################
# Step 4: Generate Training Files 
########################################

date = 'Jan30'
scorer = 'Mackenzie'

# Userparameters for training set. Other parameters can be set in pose_cfg.yaml
Shuffles = [1]  # Index for shuffles, i.e. range(5) for 5 shuffles, or [1] for 1. 
TrainingFraction = [0.95]  # Fraction of labeled images used for training (can be a list)

# Which resnet to use, 101 for deeper! 
# then change net_type and init_weights in Generating_a_Training_Set/pose_cfg.yaml 
resnet = 50

#Afterwards train your network!

########################################################################################################################
# For Evaluation your network
########################################################################################################################

#identifier of evaluation network: 
shuffleindex = 0 # indexes above Shuffles. [i.e. if 0 then the 1st shuffle in Shuffles, etc.]
snapshotindex = -1 #"all"
# To evaluate the last snapshot (i.e. the network that was trained longest) : -1 
# To evaluate all models (training stages) set this to: "all"  (as string!)


pcutoff=.1 # likelihood. RMSE will be reported for all pairs and pairs with larger likelihood than pcutoff (see paper). 
# This cutoff will also be used in plots.
plotting=True #If true will plot train & test images including DeepLabCut labels next to human labels. 
# Note that this will be plotted for all snapshots as indicated by snapshotindex
