"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

Generates training images with labels to check if annotation was done
correctly/correctly loaded.
"""
####################################################
# Loading dependencies
####################################################
import os.path
import sys
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])

import matplotlib
matplotlib.use('Agg')
from myconfig import Task, filename, bodyparts, Scorers
from myconfig import scorer as cfg_scorer
import numpy as np
import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt

###################################################
# Code if each bodypart has its own label file!
###################################################

Labels = ['.', '+', '*']  # order of labels for different scorers

#############################################
# Make sure you update the train.yaml file!
#############################################

num_joints = len(bodyparts)
all_joints = map(lambda j: [j], range(num_joints))
all_joints_names = bodyparts


# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

Colorscheme = get_cmap(len(bodyparts))

print(num_joints)
print(all_joints)
print(all_joints_names)


basefolder = './' + 'data-' + Task
numbodyparts = len(bodyparts)

# Data frame to hold data of all data sets for different scorers, bodyparts and images
DataCombined = None

os.chdir(basefolder)

DataCombined = pd.read_hdf(
    'CollectedData_' + cfg_scorer + '.h5', 'df_with_missing')

# Make list of different video data sets:
folders = [
    videodatasets for videodatasets in os.listdir(os.curdir)
    if os.path.isdir(videodatasets) and
    filename.split('.')[0] in videodatasets and 'labeled' not in videodatasets
]

print(folders)
# videos=np.sort([fn for fn in os.listdir(os.curdir) if ("avi" in fn)])
scale = 1  # for plotting
msize=25   #size of labels

for folder in folders:
    tmpfolder = folder + 'labeled'
    try:
        os.mkdir(tmpfolder)
    except:
        pass
    os.chdir(folder)
    # sort image file names according to how they were stacked (when labeled in Fiji)
    files = [
        fn for fn in os.listdir(os.curdir)
        if ("img" in fn and ".png" in fn and "_labelled" not in fn)
    ]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    comparisonbodyparts = list(set(DataCombined.columns.get_level_values(1)))

    for index, imagename in enumerate(files):
        image = io.imread(imagename)
        plt.axis('off')

        if np.ndim(image)==2:
            h, w = np.shape(image)
        else:
            h, w, nc = np.shape(image)

        plt.figure(
            frameon=False, figsize=(w * 1. / 100 * scale, h * 1. / 100 * scale))
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # This is important when using data combined / which runs consecutively!
        imindex = np.where(
            np.array(DataCombined.index.values) == folder + '/' + imagename)[0]

        plt.imshow(image, 'bone')
        for cc, scorer in enumerate(Scorers):
            if index==0:
                print("Creating images with labels by ", scorer)		
            for c, bp in enumerate(comparisonbodyparts):
                plt.plot(
                    DataCombined[scorer][bp]['x'].values[imindex],
                    DataCombined[scorer][bp]['y'].values[imindex],
                    Labels[cc],
                    color=Colorscheme(c),
                    alpha=.5,
                    ms=msize)

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()
        plt.savefig('../' + tmpfolder + '/' + imagename)
        plt.close("all")

    os.chdir("../")
