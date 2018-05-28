"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script generates the training data information for DeepCut (which requires a mat file)
based on the pandas dataframes that hold label information. The user can set the fraction of
the traing set size (from all labeled image in the hd5 file) and can create multiple shuffles.
"""

####################################################
# Loading dependencies
####################################################

import numpy as np
import scipy.io as sio
from skimage import io
import os
import yaml
import pickle
import shutil
import sys
import pandas as pd
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])
from myconfig import Task, bodyparts, date, scorer, Shuffles, TrainingFraction


def SplitTrials(trialindex, trainFraction=0.8):
    ''' Split a trial index into train and test sets'''
    trainsize = int(len(trialindex) * trainFraction)
    shuffle = np.random.permutation(trialindex)
    testIndexes = shuffle[trainsize:]
    trainIndexes = shuffle[:trainsize]

    return (trainIndexes, testIndexes)


def boxitintoacell(joints):
    ''' Auxiliary function for creating matfile.'''
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype='int64')
    return outer


def MakeTrain_pose_yaml(itemstochange, saveasfile, filename='pose_cfg.yaml'):
    raw = open(filename).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)


####################################################
# Definitions (Folders, data source and labels)
####################################################

# Loading scorer's data:
folder = 'data-' + Task + '/'
Data = pd.read_hdf(folder + 'CollectedData_' + scorer + '.h5',
                   'df_with_missing')[scorer]
# Make that folder and put in the collecteddata (see below)
bf = "UnaugmentedDataSet_" + Task + date + "/"

# This relative path is required due way DeeperCut is structured
basefolder = "../../" + bf
# copy images and folder structure in the folder containing
# training data comparison
shutil.copytree(folder, bf + folder)

for shuffle in Shuffles:
    for trainFraction in TrainingFraction:
        trainIndexes, testIndexes = SplitTrials(
            range(len(Data.index)), trainFraction)
        filename_matfile = Task + "_" + scorer + str(int(
            100 * trainFraction)) + "shuffle" + str(shuffle)
        # Filename for pickle file:
        fn = bf + "Documentation_" + folder[:-1] + "_" + str(
            int(trainFraction * 100)) + "shuffle" + str(shuffle)

        ####################################################
        # Generating data structure with labeled information & frame metadata (for deep cut)
        ####################################################

        # Make matlab train file!
        data = []
        for jj in trainIndexes:
            H = {}
            # load image to get dimensions:
            filename = Data.index[jj]
            im = io.imread(folder + filename)
            H['image'] = basefolder + folder + filename

            try:
                H['size'] = np.array(
                    [np.shape(im)[2],
                     np.shape(im)[0],
                     np.shape(im)[1]])
            except:
                # print "Grayscale!"
                H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

            indexjoints=0
            joints=np.zeros((len(bodyparts),3))*np.nan
            for bpindex,bodypart in enumerate(bodyparts):
                if Data[bodypart]['x'][jj]<np.shape(im)[1] and Data[bodypart]['y'][jj]<np.shape(im)[0]: #are labels in image?
                    	joints[indexjoints,0]=int(bpindex)    
                    	joints[indexjoints,1]=Data[bodypart]['x'][jj]
                    	joints[indexjoints,2]=Data[bodypart]['y'][jj]       
                    	indexjoints+=1

            joints = joints[np.where(
                np.prod(np.isfinite(joints),
                        1))[0], :]  # drop NaN, i.e. lines for missing body parts

            assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0])
                    )  # y coordinate within!
            assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1])
                    )  # x coordinate within!

            H['joints'] = np.array(joints, dtype=int)
            if np.size(joints)>0: #exclude images without labels
                    data.append(H)
            

        with open(fn + '.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump([data, trainIndexes, testIndexes, trainFraction], f,
                        pickle.HIGHEST_PROTOCOL)

        ################################################################################
        # Convert to idosyncratic training file for deeper cut (*.mat)
        ################################################################################

        DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
        MatlabData = np.array(
            [(np.array([data[item]['image']], dtype='U'),
              np.array([data[item]['size']]),
              boxitintoacell(data[item]['joints']))
             for item in range(len(data))],
            dtype=DTYPE)
        sio.savemat(bf + filename_matfile + '.mat', {'dataset': MatlabData})

        ################################################################################
        # Creating file structure for training &
        # Test files as well as pose_yaml files (containing training and testing information)
        #################################################################################

        experimentname = Task + date + '-trainset' + str(
            int(trainFraction * 100)) + 'shuffle' + str(shuffle)

        try:
            os.mkdir(experimentname)
            os.mkdir(experimentname + '/train')
            os.mkdir(experimentname + '/test')
        except:
            print("Apparently ", experimentname, "already exists!")

        items2change = {
            "dataset": basefolder + filename_matfile + '.mat',
            "num_joints": len(bodyparts),
            "all_joints": [[i] for i in range(len(bodyparts))],
            "all_joints_names": bodyparts
        }

        trainingdata = MakeTrain_pose_yaml(
            items2change,
            experimentname + '/train/' + 'pose_cfg.yaml',
            filename='pose_cfg.yaml')
        keys2save = [
            "dataset", "num_joints", "all_joints", "all_joints_names",
            "net_type", 'init_weights', 'global_scale', 'location_refinement',
            'locref_stdev'
        ]
        MakeTest_pose_yaml(trainingdata, keys2save,
                           experimentname + '/test/' + 'pose_cfg.yaml')
