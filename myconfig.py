# coding: utf-8

############################
# This configuration file sets various parameters for generation of training
# set file & evalutation of results
############################

# myconfig.py:

########################################
# Step 1:
Task = 'reaching'
########################################

# Filename and path to behavioral video:
vidpath = '.'
filename = 'reachingvideo1.avi'

cropping = True

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.

x1 = 0
x2 = 640
y1 = 277
y2 = 624


# Portion of the video to sample from in step 1. Set to 1 by default.
portion = 1

########################################
# Step 2:
########################################

bodyparts = ["hand", "Finger1", "Finger2",
             "Joystick"]  # Exact sequence of labels as were put by

# annotator in *.csv file
Scorers = ['Mackenzie']  # who is labeling?

# When importing the images and the labels in the csv/xls files should be in the same order!
# During labeling in Fiji one can thus (for occluded body parts) click in the origin of the image, these
# "false" labels will then be removed. To do so set the following variable:
invisibleboundary=10 # If labels are closer to origin than this number they are set to NaN (not a number)

########################################
# Step 3:
########################################

date = 'Jan30'
scorer = 'Mackenzie'

# Userparameters for training set. Other parameters can be set in pose_cfg.yaml
Shuffles = [1]  # Ids for shuffles, i.e. range(5) for 5 shuffles
TrainingFraction = [0.95]  # Fraction of labeled images used for training

# Which resnet to use
# (these are parameters reflected in the pose_cfg.yaml file)
resnet = 50

# trainingsiterations='1030000'

# For Evaluation/ Analyzing videos
# To evaluate model that was trained most set this to: "-1"
# To evaluate all models (training stages) set this to: "all"

snapshotindex = -1
shuffleindex = 0
