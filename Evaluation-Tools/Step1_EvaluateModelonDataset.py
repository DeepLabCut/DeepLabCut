"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set (images)
and stores the results in a pandas dataframe.

You need tensorflow for evaluation. Run by:
CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py
"""

####################################################
# Dependencies
####################################################

import sys
import os
subfolder = os.getcwd().split('Evaluation-Tools')[0]
sys.path.append(subfolder)

# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow")
sys.path.append(subfolder + "Generating_a_Training_Set")

from myconfig import Task, date, scorer, Shuffles, TrainingFraction, \
    snapshotindex, shuffleindex

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

# Dependencies for anaysis
import pickle
import skimage
import numpy as np
import pandas as pd
from skimage import io
import skimage.color
import auxiliaryfunctions
from tqdm import tqdm

####################################################
# Loading data, and defining model folder
####################################################

# Construct correct folder from config file references. In the case that multiple
# Shuffles were trained, this generates the datafile for the last by default (see the myconfig file):

basefolder = os.path.join('..','pose-tensorflow','models')
folder = os.path.join('UnaugmentedDataSet_' + Task + date)

datafile = ('Documentation_data-' + Task + '_' +
            str(int(TrainingFraction[shuffleindex] * 100)) + 'shuffle' +
            str(int(Shuffles[shuffleindex])) + '.pickle')

print(datafile)
with open(os.path.join(basefolder, folder ,datafile), 'rb') as f:
    data, trainIndices, testIndices, __ignore__ = pickle.load(f)

# loading meta data / i.e. training & test files
Data = pd.read_hdf(os.path.join(basefolder, folder, 'data-'+Task ,'CollectedData_' + scorer + '.h5'),
    'df_with_missing')

# "/data-"+Task+'-labelledby'+scorer+'/'
for shuffle in Shuffles:
    for trainFraction in TrainingFraction:
        experimentname = Task + date + '-trainset' + str(
            int(trainFraction * 100)) + 'shuffle' + str(shuffle)
        # loading config file:
        cfg = load_config(os.path.join(basefolder , experimentname , 'test' ,"pose_cfg.yaml"))
        modelfolder = os.path.join(basefolder, experimentname)
        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check which snap shots are available and sort them by # iterations
        Snapshots = np.array([
            fn.split('.')[0]
            for fn in os.listdir(os.path.join(basefolder, experimentname , 'train'))
            if "index" in fn
        ])
        increasing_indices = np.argsort(
            [int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        ##################################################
        # Compute predictions over images
        ##################################################

        if snapshotindex == -1:
            snapindices = [-1]
        elif snapshotindex == "all":
            snapindices = range(len(Snapshots))
        elif snapshotindex<len(Snapshots):
            snapindices=[snapshotindex]
        else:
            print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

        for snapindex in snapindices:
            cfg['init_weights'] = os.path.join(modelfolder,'train',Snapshots[snapindex])
            trainingsiterations = (
                cfg['init_weights'].split('/')[-1]).split('-')[-1]
            scorer = 'DeepCut' + "_" + str(cfg["net_type"]) + "_" + str(
                int(trainFraction *
                    100)) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations) + "forTask_" + Task

            print("Running ", scorer, " with # of trainingiterations:", trainingsiterations)

            try:
                Data = pd.read_hdf(os.path.join("Results",scorer + '.h5'),'df_with_missing')
                print("This net has already been evaluated!")
            except:
                # Specifying state of model (snapshot / training state)
                cfg['init_weights'] = os.path.join(modelfolder,'train',Snapshots[snapindex])
                sess, inputs, outputs = predict.setup_pose_prediction(cfg)
                
                Numimages = len(Data.index)
                PredicteData = np.zeros((Numimages,3 * len(cfg['all_joints_names'])))
                Testset = np.zeros(Numimages)
                print("Analyzing data...")
                for imageindex, imagename in tqdm(enumerate(Data.index)):
                    image = io.imread(os.path.join(basefolder,folder,'data-' + Task , imagename),mode='RGB')
                    image = skimage.color.gray2rgb(image)
                    image_batch = data_to_input(image)
                    # Compute prediction with the CNN
                    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
                    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    
                    # Extract maximum scoring location from the heatmap, assume 1 person
                    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
                    PredicteData[imageindex, :] = pose.flatten(
                    )  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!
    
                index = pd.MultiIndex.from_product(
                    [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                    names=['scorer', 'bodyparts', 'coords'])
                
                # Saving results:
                auxiliaryfunctions.attempttomakefolder("Results")
                DataMachine = pd.DataFrame(
                    PredicteData, columns=index, index=Data.index.values)
                DataMachine.to_hdf(os.path.join("Results",scorer + '.h5'),'df_with_missing',format='table',mode='w')
                print("Done and results stored for snapshot: ", Snapshots[snapindex])