"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script generates a data structure in pandas, that contains the (relative)
physical address of the image as well as the labels. These data are extracted
from the "labeling.csv" files that can be generated in a different file e.g.
ImageJ / Fiji

Load data from individial folders with a Task and combine in one
panda dataframe.

Keys of panda frame:
    - scorer
    - bodypart
    - x,y

The index is given by the file name / image name.

"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])
from myconfig import Task, bodyparts, Scorers, invisibleboundary

###################################################
# Code if each bodypart has its own label file!
###################################################

###################################################
basefolder = 'data-' + Task + '/'

# Data frame to hold data of all data sets for different scorers,
# bodyparts and images
DataCombined = None

for scorer in Scorers:
    os.chdir(basefolder)
    # Make list of different video data sets / each one has its own folder
    folders = [
        videodatasets for videodatasets in os.listdir(os.curdir)
        if os.path.isdir(videodatasets)
    ]
    try:
        DataSingleUser = pd.read_hdf('CollectedData_' + scorer + '.h5',
                                     'df_with_missing')
        numdistinctfolders = list(
            set([s.split('/')[0] for s in DataSingleUser.index
                 ]))  # NOTE: SLICING to eliminate multiindices!
        # print("found",len(folders),len(numdistinctfolders))
        if len(folders) > len(numdistinctfolders):
            DataSingleUsers = None
            print("Not all data converted!")
        else:
            print(scorer, "'s data already collected!")
            print(DataSingleUser.head())
    except:
        DataSingleUser = None

    if DataSingleUser is None:
        for folder in folders:
            # print("Loading folder ", folder)
            os.chdir(folder)
            # sort image file names according to how they were stacked
            # files=np.sort([fn for fn in os.listdir(os.curdir)
            # if ("img" in fn and ".png" in fn and "_labelled" not in fn)])
            files = [
                fn for fn in os.listdir(os.curdir)
                if ("img" in fn and ".png" in fn and "_labelled" not in fn)
            ]
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            imageaddress = [folder + '/' + f for f in files]
            Data_onefolder = pd.DataFrame({'Image name': imageaddress})

            frame, Frame = None, None
            for bodypart in bodyparts:
                datafile = bodypart
                try:
                    dframe = pd.read_csv(datafile + ".xls", sep='\t')
                except:
                    os.rename(datafile + ".csv", datafile + ".xls")
                    dframe = pd.read_csv(datafile + ".xls", sep='\t')

                if dframe.shape[0] != len(imageaddress):
                    # Filling up with nans
                    # dframe.set_index('Slice')
                    new_index = pd.Index(
                        np.arange(len(files)) + 1, name='Slice')
                    dframe = dframe.set_index('Slice').reindex(new_index)
                    dframe = dframe.reset_index()
                # print(dframe.index)
                # print(frame.head())
                # print(bodypart)
                index = pd.MultiIndex.from_product(
                    [[scorer], [bodypart], ['x', 'y']],
                    names=['scorer', 'bodyparts', 'coords'])

                Xrescaled = dframe.X.values.astype(float)
                Yrescaled = dframe.Y.values.astype(float)
                
                # get rid of values that are invisible >> thus user scored in left corner!
                invisiblemarkersmask = (Xrescaled < invisibleboundary) * (Yrescaled < invisibleboundary)
                Xrescaled[invisiblemarkersmask] = np.nan
                Yrescaled[invisiblemarkersmask] = np.nan
                
                if Frame is None:
                    # frame=pd.DataFrame(np.vstack([dframe.X,dframe.Y]).T, columns=index,index=imageaddress)
                    frame = pd.DataFrame(
                        np.vstack([Xrescaled, Yrescaled]).T,
                        columns=index,
                        index=imageaddress)
                    # print(frame.head())
                    Frame = frame
                else:
                    frame = pd.DataFrame(
                        np.vstack([Xrescaled, Yrescaled]).T,
                        columns=index,
                        index=imageaddress)
                    Frame = pd.concat(
                        [Frame, frame],
                        axis=1)  # along bodyparts & scorer dimension

            # print("Done with folder ", folder)
            if DataSingleUser is None:
                DataSingleUser = Frame
            else:
                DataSingleUser = pd.concat(
                    [DataSingleUser, Frame], axis=0)  # along filenames!

            os.chdir('../')

        # Save data by this scorer
        DataSingleUser.to_csv("CollectedData_" + scorer +
                              ".csv")  # breaks multiindices HDF5 tables better!
        DataSingleUser.to_hdf(
            'CollectedData_' + scorer + '.h5',
            'df_with_missing',
            format='table',
            mode='w')

    os.chdir('../')

    print("Merging scorer's data.")
