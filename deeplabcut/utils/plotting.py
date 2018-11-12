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

def PlottingResults(video,tmpfolder,Dataframe,scorer,cfg,showfigures):
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
    plt.savefig(os.path.join(tmpfolder,"trajectory.png"))
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
    plt.savefig(os.path.join(tmpfolder,"plot.png"))

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

    plt.savefig(os.path.join(tmpfolder,"plot-likelihood.png"))

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
    plt.savefig(os.path.join(tmpfolder,"hist.png"))

    if showfigures!=True:
        plt.close("all")
    else:
        plt.show()

def RunTrajectoryAnalysis(video,basefolder,scorer,videofolder,cfg,showfigures):
#    vname = video.split('.')[0]
    vname = str(Path(video).stem)
    auxiliaryfunctions.attempttomakefolder(os.path.join(basefolder,'plot-poses'))
    tmpfolder = os.path.join(basefolder,'plot-poses', vname)
    auxiliaryfunctions.attempttomakefolder(tmpfolder)

    print("Loading ", video, "and data.")
    dataname = str(Path(video).stem) + scorer + '.h5'
    try:
        Dataframe = pd.read_hdf(os.path.join(videofolder,dataname))
        PlottingResults(video,tmpfolder,Dataframe,scorer,cfg,showfigures)
    except FileNotFoundError:
        datanames=[fn for fn in os.listdir(videofolder) if (vname in fn) and (".h5" in fn) and "resnet" in fn]
        if len(datanames)==0:
            print("The video was not analyzed with this scorer:", scorer)
            print("No other scorers were found, please run AnalysisVideos.py first.")
        elif len(datanames)>0:
            print("The video was not analyzed with this scorer:", scorer)
            print("Other scorers were found, however:", datanames)
            print("Creating labeled video for:", datanames[0]," instead.")
            Dataframe = pd.read_hdf(os.path.join(videofolder,datanames[0]))
            PlottingResults(video,tmpfolder,Dataframe,scorer,cfg,showfigures)
    except FileNotFoundError:
        print("Data was not analyzed (run AnalysisVideos.py first).")


##################################################
# Looping analysis over video
##################################################

def plot_trajectories(config,videos,shuffle=1,trainingsetindex=0,videotype='.avi',showfigures=False):
    """
    Plots the trajectories of various bodyparts across the video.
    
    Parameters
    ----------
     config : string
    Full path of the config.yaml file as a string.
    
    videos : list
    A list of strings containing the full paths of the videos to analyze.
    
    shuffle: list, optional
    List of integers specifying the shuffle indices of the training dataset. The default is [1]
    
     trainingsetindex: int, optional
    Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    
    videotype: string, optional
    Checks for the extension of the video in case the input is a directory.\nOnly videos with this extension are analysed. The default is ``.avi``
    
    showfigures: bool, default false
    If true then plots are also displayed.
    
    
    Example
    --------
    for labeling the frames
    >>> deeplabcut.plot_trajectories('home/alex/analysis/project/reaching-task/config.yaml',['/home/alex/analysis/project/videos/reachingvideo1.avi'])
    --------
    
    """
    
    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction) #automatically loads corresponding model (even training iteration based on snapshot index)
    
    #checks if input is a directory
    if [os.path.isdir(i) for i in videos] == [True]:#os.path.isdir(video)==True:
        """
        Analyze all the videos in the directory
        """
        print("Analyzing all the videos in the directory")
        videofolder= videos[0]
        os.chdir(videofolder)
        Videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn) and ("labeled" not in fn)])
    else:
        Videos = videos
    
    for video in Videos:
        print(video)
        videofolder= str(Path(video).parents[0]) #where your folder with videos is.
        videotype = str(Path(video).suffix)
        print("Starting % ", videofolder, videos)
        basefolder=videofolder
        auxiliaryfunctions.attempttomakefolder(basefolder)
        RunTrajectoryAnalysis(video,basefolder,DLCscorer,videofolder,cfg,showfigures)
        print('Plots created! Please check the directory "plot-poses" within the video directory')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('video')
    cli_args = parser.parse_args()
