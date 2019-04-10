"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""

import os, pickle, yaml
import pandas as pd
from pathlib import Path
import numpy as np

from deeplabcut.utils import auxiliaryfunctions

def convertcsv2h5(config,userfeedback=True,scorer=None):
    """
    Convert (image) annotation files in folder labeled-data from csv to h5.
    This function allows the user to manually edit the csv (e.g. to correct the scorer name and then convert it into hdf format).
    WARNING: conversion might corrupt the data.
    
    config : string
        Full path of the config.yaml file as a string.
    
    userfeedback: bool, optional
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.
        
    scorer: string, optional
        If a string is given, then the scorer/annotator in all csv and hdf files that are changed, will be overwritten with this name. 
        
    Examples
    --------
    Convert csv annotation files for reaching-task project into hdf. 
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml')
    
    --------
    Convert csv annotation files for reaching-task project into hdf while changing the scorer/annotator in all annotation files to Albert!
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml',scorer='Albert')
    --------
    """
    cfg = auxiliaryfunctions.read_config(config)
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    folders = [Path(config).parent / 'labeled-data' /Path(i) for i in video_names]
    if scorer==None:
        scorer=cfg['scorer']

    for folder in folders:
        try:
            if userfeedback==True:
                print("Do you want to convert the csv file in folder:", folder, "?")
                askuser = input("yes/no")
            else:
                askuser="yes"
            
            if askuser=='y' or askuser=='yes' or askuser=='Ja' or askuser=='ha': # multilanguage support :)
                fn=os.path.join(str(folder),'CollectedData_' + cfg['scorer'] + '.csv')
                data=pd.read_csv(fn)
                
                #nlines,numcolumns=data.shape
                
                orderofbpincsv=list(data.values[0,1:-1:2])
                imageindex=list(data.values[2:,0])
                
                #assert(len(orderofbpincsv)==len(cfg['bodyparts']))
                print(orderofbpincsv)
                print(cfg['bodyparts'])
                
                #TODO: test len of images vs. len of imagenames for another sanity check
                
                index = pd.MultiIndex.from_product([[scorer], orderofbpincsv, ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(np.array(data.values[2:,1:],dtype=float), columns = index, index = imageindex)

                frame.to_hdf(os.path.join(str(folder),'CollectedData_'+ cfg['scorer']+".h5"), key='df_with_missing', mode='w')
                frame.to_csv(fn)
        except FileNotFoundError:
            print("Attention:", folder, "does not appear to have labeled data!")

def analyze_videos_converth5_to_csv(videopath,videotype='.avi'):
    """
    By default the output poses (when running analyze_videos) are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
    in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
    in the same directory, where the video is stored. If the flag save_as_csv is set to True, the data is also exported as comma-separated value file. However,
    if the flag was *not* set, then this function allows the conversion of all h5 files to csv files (without having to analyze the videos again)!
    
    This functions converts hdf (h5) files to the comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.
    
    Parameters
    ----------
    
    videopath : string
        A strings containing the full paths to videos for analysis or a path to the directory where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\nOnly videos with this extension are analyzed. The default is ``.avi``

    Examples
    --------

    Converts all pose-output files belonging to mp4 videos in the folder '/media/alex/experimentaldata/cheetahvideos' to csv files. 
    deeplabcut.analyze_videos_converth5_to_csv('/media/alex/experimentaldata/cheetahvideos','.mp4')  
 
    """
    start_path=os.getcwd()
    os.chdir(videopath)
    Videos=[fn for fn in os.listdir(os.curdir) if (videotype in fn) and ('_labeled.mp4' not in fn)] #exclude labeled-videos!
    
    Allh5files=[fn for fn in os.listdir(os.curdir) if (".h5" in fn) and ("resnet" in fn)]
    
    for video in Videos:
         vname = Path(video).stem
         #Is there a scorer for this?
         PutativeOutputFiles=[fn for fn in Allh5files if vname in fn]
         for pfn in PutativeOutputFiles:
             scorer=pfn.split(vname)[1].split('.h5')[0]
             if "DeepCut" in scorer:
                 DC = pd.read_hdf(pfn, 'df_with_missing')
                 print("Found output file for scorer:", scorer)
                 print("Converting to csv...")
                 DC.to_csv(pfn.split('.h5')[0]+'.csv')
    
    os.chdir(str(start_path))
    print("All pose files were converted.")

def pathmagic(string):
    parts=string.split('\\')
    if len(parts)==1:
        return string
    elif len(parts)==3: #this is the expected windows case, it will split into labeled-data, video, imgNR.png
        return os.path.join(*parts) #unpack arguments from list with splat operator
    else:
        return string

def convertpaths_to_unixstyle(Data,fn,cfg):
    ''' auxiliary function that converts paths in annotation files:
        labeled-data\\video\\imgXXX.png to labeled-data/video/imgXXX.png '''
    Data.to_csv(fn + "windows" + ".csv")
    Data.to_hdf(fn + "windows" + '.h5','df_with_missing',format='table', mode='w')

    imindex=[pathmagic(s) for s in Data.index]
    for j,bpt in enumerate(cfg['bodyparts']):
        index = pd.MultiIndex.from_product([[cfg['scorer']], [bpt], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(Data[cfg['scorer']][bpt].values, columns = index, index = imindex)
        if j==0:
            dataFrame=frame
        else:
            dataFrame = pd.concat([dataFrame, frame],axis=1)
    
    dataFrame.to_csv(fn + ".csv")
    dataFrame.to_hdf(fn + '.h5','df_with_missing',format='table', mode='w')
    return dataFrame

def merge_windowsannotationdataONlinuxsystem(cfg):
    ''' If a project was created on Windows (and labeled there,) but ran on unix then the data folders
    corresponding in the keys in cfg['video_sets'] are not found. This function gets them directly by 
    looping over all folders in labeled-data '''
    
    AnnotationData=None
    data_path = Path(cfg['project_path'],'labeled-data')
    annotationfolders=[fn for fn in os.listdir(data_path) if "_labeled" not in fn]
    print("The following folders were found:", annotationfolders)
    for folder in annotationfolders:
        try:
            data = pd.read_hdf(os.path.join(data_path , folder, 'CollectedData_'+cfg['scorer']+'.h5'),'df_with_missing')
            if AnnotationData is None:
                AnnotationData=data
            else:
                AnnotationData=pd.concat([AnnotationData, data])

        except FileNotFoundError:
            print(str(os.path.join(data_path , folder, 'CollectedData_'+cfg['scorer']+'.h5')), " not found (perhaps not annotated)")

    return AnnotationData

