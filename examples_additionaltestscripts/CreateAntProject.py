#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:25:28 2019

@author: alex
"""

import numpy as np

from pathlib import Path
import os, sys

os.environ['DLClight']='True'
import deeplabcut
import shutil


p1='/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/silversideschooling-Valentina-2019-07-14'
p2='/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/escaping-souvik-2019-09-25'

for projectpath in [p1,p2]:
    config=os.path.join(projectpath,'config.yaml')
    cfg=deeplabcut.auxiliaryfunctions.read_config(config)
    videopath=os.path.join(projectpath,'videos')

    if 'escaping' in projectpath:
        PAF_graph=[[0,1], [1,4], [2,3], [3,4], [4,12], [12,11], [11,10], [10,9], [4,25], [25,24], [24,23], [23,22], [4,5], [5,15], [15,14], [14,13], [5,28], [28,27], [27,26], [5,6], [6,18], [18,17], [17,16], [6,31], [31,30], [30,29], [6,21], [21,20], [20,19], [6,34], [34,33], [33,32], [6,7], [7,8]]
        PAF_graph.extend([[4,9],[4,22]]) #head to tip of antenna
        PAF_graph.extend([[15,13],[28,26],[31,29],[18,16],[21,19],[34,32],[9,11],[24,22]])
        PAF_graph.extend([[4,6],[6,8],[6,14],[6,27],[6,17],[6,30],[6,33],[6,20]]) #direct leg
    else:
        PAF_graph=[[0,1],[0,2],[1,2],[2,3],[3,1]] #central bodyparts
        PAF_graph.extend([[4,5],[2,5],[1,6],[1,4],[4,6],[5,6],[6,2],[3,6]])
        PAF_graph.extend([[7,8],[2,9],[1,9],[1,7],[7,9],[8,9],[8,2],])

    deeplabcut.utils.auxfun_multianimal.setgraphfromarray(config,PAF_graph)

    #deeplabcut.cropimagesandlabels(config,userfeedback=False)

    # Then for cropped images:
    #deeplabcut.check_labels(config)


    # Then for cropped images (remove full image from index!)
    # Get the data:
    trainingsetfolder = deeplabcut.auxiliaryfunctions.GetTrainingSetFolder(cfg)
    deeplabcut.auxiliaryfunctions.attempttomakefolder(Path(os.path.join(projectpath,str(trainingsetfolder))),recursive=True)
    Data = deeplabcut.generate_training_dataset.trainingsetmanipulation.merge_annotateddatasets(cfg,projectpath,Path(os.path.join(projectpath,trainingsetfolder)),windows2linux=False)

    fraction=cfg['TrainingFraction'][0]

    indices=[(fn.split('/')[-2],fn.split('/')[-1]) for fn in Data.index]
    #this ugly code below splits the data so that crops from the same image are either in test or train!
    for shuffle in range(3):
        frac=0
        while frac<.03:
            trainindex=[]
            testindex=[]

            for index,(folder,imname) in enumerate(indices):
                if 'cropped' in folder:
                    stem,cropindex=imname.split('c')
                    c=int(cropindex.split('.png')[0])
                    if c==0:
                        if np.random.rand()>fraction:
                            test=True
                        else:
                            test=False
                    if test:
                        testindex.append(index)
                    else:
                        trainindex.append(index)
                else:
                    if np.random.rand()>fraction:
                        testindex.append(index)
                    else:
                        trainindex.append(index)

            #print("Train:")
            #for t in trainindex:
            #    print(Data.index[t])
            #print("Test:")
            #for t in testindex:
            #    print(Data.index[t])
            print("Fraction for split", shuffle)
            print(len(testindex)*1./(len(trainindex)+len(testindex)))
            frac=len(testindex)*1./(len(trainindex)+len(testindex))

        print(testindex)
        #trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.train_network_path(config,shuffle=shuffle,trainingsetindex=trainingsetindex)
        deeplabcut.create_training_dataset(config,Shuffles=[shuffle],trainIndexes=trainindex,testIndexes=testindex)
