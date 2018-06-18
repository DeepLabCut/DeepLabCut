# coding: utf-8

#########################################################################################
# This configuration file sets various parameters for running a trained model on videos!
# First, make of course sure that the network performed well on train/test data set. 
#
# The pose output is save as hdf file with a name containing the video name as well as the network name
# You can also save the output as csv (see below) or in many other formats see https://github.com/AlexEMG/DeepLabCut/issues/17
#
#########################################################################################

# Filename and path to behavioral video (for labeling)
videofolder = '../videos/'
cropping = False
videotype='.avi' #type of videos to analyze 

#Note: under the hood there is moviepy, which can handle many types of videos:
#See: https://zulko.github.io/moviepy/_modules/moviepy/video/io/VideoFileClip.html

# If you have stacks of tiffs (instead of videos) you can use "AnalyzeABunchofPictures.py"

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.

x1 = 0
x2 = 640
y1 = 277
y2 = 624

#########################################################################################
# Analysis Network parameters (so that the right networks is loaded for analysis)
#########################################################################################
scorer = 'Mackenzie'
Task = 'reaching'
date = 'Jan30'
trainingsFraction = 0.95  # Fraction of labeled images used for training
resnet = 50
snapshotindex = -1
shuffle = 1

storedata_as_csv=False #if true then the time series of poses will (also) be saved as csv. 
# Note the data is always saved in hdf - format which is an efficient format that easily allows to load the full pandas multiarray at a later stage

#########################################################################################
## For plotting (MakingLabeledVideo.py)
#########################################################################################
trainingsiterations = 3500  # type the number listed in the h5 file containing the pose estimation data. The video will be generated
#based on the labels for this network state.
pcutoff = 0.1  # likelihood cutoff for body part in image

# delete individual (labeled) frames after making video? (note there could be many...)
deleteindividualframes = False
alphavalue=.1 # "strength/transparency level of makers" in individual frames (Vary from 0 to 1. / not working in "MakingLabeledVideo_fast.py")
dotsize = 5
colormap='cool' #hsv'
