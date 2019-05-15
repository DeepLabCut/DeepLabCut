"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu


"""

####################################################
# Dependencies
####################################################
import os.path
from pathlib import Path
import argparse
from deeplabcut.utils import auxiliaryfunctions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def Histogram(vector,color,bins):
    dvector=np.diff(vector)
    dvector=dvector[np.isfinite(dvector)]
    plt.hist(dvector,color=color,histtype='step',bins=bins)
#    fig.colorbar(im, ax=ax)

def PlottingResults(video,tmpfolder,Dataframe,scorer,cfg,showfigures,suffix='.png'):
    ''' Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods.'''
    plt.figure(figsize=(8, 6))
    bodyparts2plot = cfg['bodyparts']
    pcutoff = cfg['pcutoff']
    colors = get_cmap(len(bodyparts2plot),name = cfg['colormap'])
    alphavalue = cfg['alphavalue']

    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Dataframe[scorer][bp]['x'].values[Index],Dataframe[scorer][bp]['y'].values[Index],'.',color=colors(bpindex),alpha=alphavalue)

    plt.gca().invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cfg['colormap']), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.savefig(os.path.join(tmpfolder,"trajectory"+suffix))
    plt.figure(figsize=(30, 10))
    Time=np.arange(np.size(Dataframe[scorer][bodyparts2plot[0]]['x'].values))

    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Time[Index],Dataframe[scorer][bp]['x'].values[Index],'--',color=colors(bpindex),alpha=alphavalue)
        plt.plot(Time[Index],Dataframe[scorer][bp]['y'].values[Index],'-',color=colors(bpindex),alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cfg['colormap']), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('X and y-position in pixels')
    plt.savefig(os.path.join(tmpfolder,"plot"+suffix))

    plt.figure(figsize=(30, 10))
    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Time,Dataframe[scorer][bp]['likelihood'].values,'-',color=colors(bpindex),alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cfg['colormap']), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('likelihood')

    plt.savefig(os.path.join(tmpfolder,"plot-likelihood"+suffix))

    plt.figure()
    bins=np.linspace(0,np.amax(Dataframe.max()),100)

    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values < pcutoff
        X=Dataframe[scorer][bp]['x'].values
        X[Index]=np.nan
        Histogram(X,colors(bpindex),bins)
        Y=Dataframe[scorer][bp]['x'].values
        Y[Index]=np.nan
        Histogram(Y,colors(bpindex),bins)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cfg['colormap']), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.ylabel('Count')
    plt.xlabel('DeltaX and DeltaY')
    plt.savefig(os.path.join(tmpfolder,"hist"+suffix))

    if showfigures!=True:
        plt.close("all")
    else:
        plt.show()

def RunTrajectoryAnalysis(video,basefolder,DLCscorer,videofolder,cfg,showfigures,filtered=False):
    vname = str(Path(video).stem)
    auxiliaryfunctions.attempttomakefolder(os.path.join(basefolder,'plot-poses'))
    tmpfolder = os.path.join(basefolder,'plot-poses', vname)
    auxiliaryfunctions.attempttomakefolder(tmpfolder)

    print("Loading ", video, "and data.")
    try:
        if filtered==False:
            dataname = str(Path(video).stem) + DLCscorer + '.h5'
            Dataframe = pd.read_hdf(os.path.join(videofolder,dataname))
            suffix='.png'
        else: #for filtered output
            try:
                Dataframe = pd.read_hdf(os.path.join(videofolder,str(Path(video).stem) + DLCscorer + 'filtered.h5'))
            except FileNotFoundError:
                print("No filtered predictions found, using frame-by-frame output instead.")
                Dataframe = pd.read_hdf(os.path.join(videofolder,str(Path(video).stem) + DLCscorer + '.h5'))
            suffix='_filtered.png'
            
        PlottingResults(video,tmpfolder,Dataframe,DLCscorer,cfg,showfigures,suffix)
        
    except FileNotFoundError:
        datanames=[fn for fn in os.listdir(videofolder) if (vname in fn) and (".h5" in fn) and "resnet" in fn]
        if len(datanames)==0:
            print("The video was not analyzed with this scorer:", DLCscorer)
            print("No other scorers were found, please run AnalysisVideos.py first.")
            
        elif len(datanames)>0:
            print("The video was not analyzed with this scorer:", DLCscorer)
            print("Other scorers were found, however:", datanames)
            print("Creating plots for:", datanames[0]," instead.")
            
            Dataframe = pd.read_hdf(os.path.join(videofolder,datanames[0]))
            PlottingResults(video,tmpfolder,Dataframe,DLCscorer,cfg,showfigures,suffix)


##################################################
# Looping analysis over video
##################################################

def plot_trajectories(config,videos,videotype='.avi',shuffle=1,trainingsetindex=0,filtered=False,showfigures=False, destfolder=None):
    """
    Plots the trajectories of various bodyparts across the video.
    
    Parameters
    ----------
     config : string
    Full path of the config.yaml file as a string.
    
    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.
    
    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``
    
    shuffle: list, optional
    List of integers specifying the shuffle indices of the training dataset. The default is [1]
    
    trainingsetindex: int, optional
    Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    
    filtered: bool, default false
    Boolean variable indicating if filtered output should be plotted rather than frame-by-frame predictions. Filtered version can be calculated with deeplabcut.filterpredictions
    
    showfigures: bool, default false
    If true then plots are also displayed.
    
    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video). 
    
    Example
    --------
    for labeling the frames
    >>> deeplabcut.plot_trajectories('home/alex/analysis/project/reaching-task/config.yaml',['/home/alex/analysis/project/videos/reachingvideo1.avi'])
    --------
    
    """
    
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction) #automatically loads corresponding model (even training iteration based on snapshot index)
    
    Videos=auxiliaryfunctions.Getlistofvideos(videos,videotype)
    for video in Videos:
        print(video)
        if destfolder is None:
            videofolder = str(Path(video).parents[0])
        else:
            videofolder=destfolder
            
        videotype = str(Path(video).suffix)
        print("Starting % ", videofolder, videos)
        basefolder=videofolder
        auxiliaryfunctions.attempttomakefolder(basefolder)
        
        RunTrajectoryAnalysis(video,basefolder,DLCscorer,videofolder,cfg,showfigures,filtered)
        print('Plots created! Please check the directory "plot-poses" within the video directory')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('video')
    cli_args = parser.parse_args()
