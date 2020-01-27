"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os, pickle
import pandas as pd
import numpy as np

def extractindividualsandbodyparts(cfg):
    individuals=cfg['individuals']
    if len(cfg['uniquebodyparts'])>0:
        individuals.append('single')
    return individuals,cfg['uniquebodyparts'],cfg['multianimalbodyparts']

def getpafgraph(cfg,printnames=True):
    ''' auxiliary function that turns skeleton (list of connected bodypart pairs) INTO
    list of corresponding indices '''
    individuals,uniquebodyparts,multianimalbodyparts=extractindividualsandbodyparts(cfg)
    # Attention this order has to be consistent (for training set creation, training, inference etc.)
    bodypartnames=multianimalbodyparts+uniquebodyparts
    lookupdict={bodypartnames[j]:j for j in range(len(bodypartnames))}
    partaffinityfield_graph=[]
    for link in cfg['skeleton']:
        if link[0] in bodypartnames and link[1] in bodypartnames:
            partaffinityfield_graph.append([int(lookupdict[link[0]]),int(lookupdict[link[1]])])
        else:
            print("Attention, parts do not exist!", link)

    if printnames:
        graph2names(cfg,partaffinityfield_graph)

    return partaffinityfield_graph

def graph2names(cfg,partaffinityfield_graph):
    individuals,uniquebodyparts,multianimalbodyparts=extractindividualsandbodyparts(cfg)
    bodypartnames=uniquebodyparts+multianimalbodyparts
    for pair in partaffinityfield_graph:
        print(pair,bodypartnames[pair[0]],bodypartnames[pair[1]])

def SaveFullMultiAnimalData(data, metadata, dataname,suffix='_full'):
    ''' Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py '''
    with open(dataname.split('.h5')[0]+suffix+'.pickle', 'wb') as f:
            # Pickle the 'labeled-data' dictionary using the highest protocol available.
            pickle.dump(data, f,pickle.HIGHEST_PROTOCOL)
    #if suffix=='_full': #save metadata!
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'wb') as f:
            # Pickle the 'labeled-data' dictionary using the highest protocol available.
            pickle.dump(metadata, f,pickle.HIGHEST_PROTOCOL)

def LoadFullMultiAnimalData(dataname):
    ''' Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py '''
    with open(dataname.split('.h5')[0]+'_full.pickle', 'rb') as handle:
        data=pickle.load(handle)
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'rb') as handle:
        metadata=pickle.load(handle)
    return data, metadata

#TODO: FIX THIS FUNCTION!
def SaveMultiAnimalData(PredicteData, metadata, dataname, pdindex, imagenames,save_as_csv):
    ''' Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py '''
    #TODO: update!
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
    if save_as_csv:
        print("Saving csv poses!")
        DataMachine.to_csv(dataname.split('.h5')[0]+'.csv')
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


from deeplabcut.utils.auxiliaryfunctions import read_config
from pathlib import Path
def returnlabelingdata(config):
    ''' Returns a specific labeleing data set -- the user will be asked which one. '''
    cfg = read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]
    for folder in folders:
            print("Do you want to get the data for folder:", folder, "?")
            askuser = input("yes/no")
            if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha': # multilanguage support :)
                fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.h5')
                Data=pd.read_hdf(fn)
                return Data


def convertmultianimaltosingleanimaldata(config,userfeedback=True,target=None):
    ''' Convert multi animal to single animal code and vice versa. Note that by providing target='single'/'multi' this will be target! '''
    cfg = read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]

    prefixes, uniquebodyparts, multianimalbodyparts = extractindividualsandbodyparts(cfg)
    for folder in folders:
        if userfeedback==True:
            print("Do you want to convert the annotation file in folder:", folder, "?")
            askuser = input("yes/no")
        else:
            askuser="yes"

        if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha': # multilanguage support :)
            fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'])
            Data=pd.read_hdf(fn+ '.h5','df_with_missing')
            imindex=Data.index

            if 'individuals' in Data.columns.names and (target==None or target=='single'):
                print("This is a multianimal data set, converting to single...",folder)
                for prfxindex,prefix in enumerate(prefixes):
                    if prefix=='single':
                        for j,bpt in enumerate(uniquebodyparts):
                            index = pd.MultiIndex.from_product([[cfg['scorer']], [bpt], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                            frame = pd.DataFrame(Data[cfg['scorer']][prefix][bpt].values, columns = index, index = imindex)
                            if j==0:
                                dataFrame=frame
                            else:
                                dataFrame = pd.concat([dataFrame, frame],axis=1)
                    else:
                        for j,bpt in enumerate(multianimalbodyparts):
                            index = pd.MultiIndex.from_product([[cfg['scorer']], [prefix+bpt], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                            frame = pd.DataFrame(Data[cfg['scorer']][prefix][bpt].values, columns = index, index = imindex)
                            if j==0:
                                dataFrame=frame
                            else:
                                dataFrame = pd.concat([dataFrame, frame],axis=1)
                    if prfxindex==0:
                        DataFrame=dataFrame
                    else:
                        DataFrame=pd.concat([DataFrame,dataFrame],axis=1)

                Data.to_hdf(fn + 'multianimal.h5','df_with_missing',format='table', mode='w')
                Data.to_csv(fn + "multianimal.csv")

                DataFrame.to_hdf(fn + '.h5','df_with_missing',format='table', mode='w')
                DataFrame.to_csv(fn + ".csv")
            elif target==None or target=='multi':
                print("This is a single animal data set, converting to multi...",folder)
                for prfxindex,prefix in enumerate(prefixes):
                    if prefix=='single':
                        if cfg['uniquebodyparts']!=[None]:
                            for j,bpt in enumerate(uniquebodyparts):
                                index = pd.MultiIndex.from_arrays(np.array([2*[cfg['scorer']], 2*[prefix], 2*[bpt], ['x', 'y']]),names=['scorer', 'individuals', 'bodyparts', 'coords'])
                                if bpt in Data[cfg['scorer']].keys():
                                    frame = pd.DataFrame(Data[cfg['scorer']][bpt].values, columns = index, index = imindex)
                                else: #fill with nans...
                                    frame = pd.DataFrame(np.ones((len(imindex),2))*np.nan, columns = index, index = imindex)

                                if j==0:
                                    dataFrame=frame
                                else:
                                    dataFrame = pd.concat([dataFrame, frame],axis=1)
                        else:
                            dataFrame=None
                    else:
                        for j,bpt in enumerate(multianimalbodyparts):
                            index = pd.MultiIndex.from_arrays(np.array([2*[cfg['scorer']], 2*[prefix], 2*[bpt], ['x', 'y']]),names=['scorer', 'individuals', 'bodyparts', 'coords'])
                            if prefix+'_'+bpt in Data[cfg['scorer']].keys():
                                frame = pd.DataFrame(Data[cfg['scorer']][prefix+'_'+bpt].values, columns = index, index = imindex)
                            else:
                                frame = pd.DataFrame(np.ones((len(imindex),2))*np.nan, columns = index, index = imindex)

                            if j==0:
                                dataFrame=frame
                            else:
                                dataFrame = pd.concat([dataFrame, frame],axis=1)
                    if prfxindex==0:
                        DataFrame=dataFrame
                    else:
                        DataFrame=pd.concat([DataFrame,dataFrame],axis=1)

                Data.to_hdf(fn + 'singleanimal.h5','df_with_missing',format='table', mode='w')
                Data.to_csv(fn + "singleanimal.csv")

                DataFrame.to_hdf(fn + '.h5','df_with_missing',format='table', mode='w')
                DataFrame.to_csv(fn + ".csv")
