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


p1='/home/alex/Dropbox/InterestingCode/social_datasets/croppedNov18/CrackingParenting-Mostafizur-2019-08-08'

for projectpath in [p1]:
    config=os.path.join(projectpath,'config.yaml')
    cfg=deeplabcut.auxiliaryfunctions.read_config(config)
    videopath=os.path.join(projectpath,'videos')

    PAF_graph=[[0,1],[1,2],[2,3],[3,4],[0,4],[0,2],[2,4]] #PUPS!
    #due to introduction of 3 interim points!
    PAF_graph.extend([[p[0]+3,p[1]+3] for p in [[2,3],[2,4],[2,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[11,13],[9,11],[9,13]]])

    #additional PAFs!!!
    PAF_graph.extend([[p[0]+3,p[1]+3] for p in [[2,7],[3,7],[4,7],[7,9]]])
    # TO DO: fix for uniquebodyparts!!
    #deeplabcut.utils.auxfun_multianimal.setgraphfromarray(config,PAF_graph)

    #deeplabcut.cropimagesandlabels(config,userfeedback=False,numcrops=10, size=(400, 400))

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
