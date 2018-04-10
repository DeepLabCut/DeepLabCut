# coding: utf-8

############################
# This configuration file sets various parameters for running a trained model,
# that performed well on train/test set on videos
############################

# Filename and path to behavioral video (for labeling)
videofolder = '../videos/'
cropping = False

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.

x1 = 0
x2 = 640
y1 = 277
y2 = 624

# Analysis Network parameters:

scorer = 'Mackenzie'
Task = 'reaching'
date = 'Jan1'
trainingsFraction = 0.95  # Fraction of labeled images used for training
resnet = 50
snapshotindex = -1
shuffle = 1

# For plotting:
trainingsiterations = 750000  # type the number listed in .pickle file
pcutoff = 0.1  # likelihood cutoff for body part in image
# delete individual (labeled) frames after making video?
deleteindividualframes = False
