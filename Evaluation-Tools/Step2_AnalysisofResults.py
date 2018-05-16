"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates the scorer's labels and computes the accuracy.
"""

####################################################
# Dependencies
####################################################

import sys
import os
import pickle
sys.path.append(os.getcwd().split('Evaluation-Tools')[0])
from myconfig import Task, date, scorer, Shuffles, TrainingFraction, snapshotindex
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

####################################################
# Auxiliary functions
####################################################


def pairwisedistances(DataCombined, scorer1, scorer2, bodyparts=None):
    if bodyparts is None:
        Pointwisesquareddistance = (
            DataCombined[scorer1] - DataCombined[scorer2])**2
        MSE = np.sqrt(
            Pointwisesquareddistance.xs('x', level=1, axis=1) +
            Pointwisesquareddistance.xs('y', level=1, axis=1))
        return MSE
    else:  # calculationg MSE only for specific bodyparts / given by list
        Pointwisesquareddistance = (DataCombined[scorer1][bodyparts] -
                                    DataCombined[scorer2][bodyparts])**2
        MSE = np.sqrt(
            Pointwisesquareddistance.xs('x', level=1, axis=1) +
            Pointwisesquareddistance.xs('y', level=1, axis=1))
        return MSE


fs = 15  # fontsize for plots
####################################################
# Loading dependencies
####################################################

# loading meta data / i.e. training & test files
basefolder = '../pose-tensorflow/models/'
datafolder = basefolder + "UnaugmentedDataSet_" + Task + date + '/'
Data = pd.read_hdf(
    datafolder + 'data-' + Task + '/CollectedData_' + scorer + '.h5',
    'df_with_missing')

####################################################
# Models vs. benchmark for varying training state
####################################################

# only specific parts can also be compared (not all!):
comparisonbodyparts = list(np.unique(Data.columns.get_level_values(1)))

for trainFraction in TrainingFraction:
    for shuffle in Shuffles:

        fns = [
            file for file in os.listdir('Results')
            if "forTask:" + str(Task) in file and "shuffle" + str(shuffle) in
            file and "_" + str(int(trainFraction * 100)) in file
        ]

        metadatafile = datafolder + "Documentation_" + "data-" + Task + "_" + str(
            int(trainFraction * 100)) + "shuffle" + str(shuffle) + ".pickle"
        with open(metadatafile, 'rb') as f:
            [
                trainingdata_details, trainIndexes, testIndexes,
                testFraction_data
            ] = pickle.load(f)

        #extract training iterations:
        TrainingIterations=[(int(fns[j].split("forTask")[0].split('_')[-1]),j) for j in range(len(fns))]
        TrainingIterations.sort(key=lambda tup: tup[0]) #sort according to increasing # training steps!
        print("Found the following training snapshots: ",TrainingIterations)
        print("You can choose among those for analyis of train/test performance.")

        if snapshotindex == -1:
            snapindices = [TrainingIterations[-1]]
        elif snapshotindex == "all":
            snapindices = TrainingIterations
        elif snapshotindex<len(TrainingIterations):
            snapindices=[TrainingIterations[snapshotindex]]
        else:
            print("Invalid choice, only -1 (last), all (as string), or index corresponding to one of the listed training snapshots can be analyzed.")
            print("Others might not have been evaluated!")
            snapindices=[]

        for trainingiterations,index in snapindices:
        		DataMachine = pd.read_hdf("Results/" + fns[index], 'df_with_missing')
        		DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
        		scorer_machine = DataMachine.columns.get_level_values(0)[0]
        
        		MSE = pairwisedistances(DataCombined, scorer, scorer_machine,
        		                        comparisonbodyparts)
        
        		testerror = np.nanmean(MSE.iloc[testIndexes].values.flatten())
        		trainerror = np.nanmean(MSE.iloc[trainIndexes].values.flatten())
        
        		print("Results for",trainingiterations, "training iterations:", int(100 * trainFraction), shuffle, "train error:",
        		      trainerror, "test error:", testerror)
