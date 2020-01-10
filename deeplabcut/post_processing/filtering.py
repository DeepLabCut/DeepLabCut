"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import numpy as np
import os
from pathlib import Path
import pandas as pd

from deeplabcut.utils import auxiliaryfunctions, visualization
from deeplabcut.utils import frameselectiontools
from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from scipy import signal


def filterpredictions(config,video,videotype='avi',shuffle=1,trainingsetindex=0,
            filtertype='median',windowlength=5,
            p_bound=.001,ARdegree=3,MAdegree=1,alpha=.01,save_as_csv=True,destfolder=None):
    """

    Fits frame-by-frame pose predictions with ARIMA model (filtertype='arima') or median filter (default).

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    video : string
        Full path of the video to extract the frame from. Make sure that this video is already analyzed.

    shuffle : int, optional
        The shufle index of training dataset. The extracted frames will be stored in the labeled-dataset for
        the corresponding shuffle of training dataset. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    filtertype: string
        Select which filter, 'arima' or 'median' filter.

    windowlength: int
        For filtertype='median' filters the input array using a local window-size given by windowlength. The array will automatically be zero-padded.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html The windowlenght should be an odd number.

    p_bound: float between 0 and 1, optional
        For filtertype 'arima' this parameter defines the likelihood below,
        below which a body part will be consided as missing data for filtering purposes.

    ARdegree: int, optional
        For filtertype 'arima' Autoregressive degree of Sarimax model degree.
        see https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    MAdegree: int
        For filtertype 'arima' Moving Avarage degree of Sarimax model degree.
        See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    alpha: float
        Significance level for detecting outliers based on confidence interval of fitted SARIMAX model.

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    Example
    --------
    Arima model:
    deeplabcut.filterpredictions('C:\\myproject\\reaching-task\\config.yaml',['C:\\myproject\\trailtracking-task\\test.mp4'],shuffle=3,filterype='arima',ARdegree=5,MAdegree=2)

    Use median filter over 10bins:
    deeplabcut.filterpredictions('C:\\myproject\\reaching-task\\config.yaml',['C:\\myproject\\trailtracking-task\\test.mp4'],shuffle=3,windowlength=10)

    One can then use the filtered rather than the frame-by-frame predictions by calling:

    deeplabcut.plot_trajectories('C:\\myproject\\reaching-task\\config.yaml',['C:\\myproject\\trailtracking-task\\test.mp4'],shuffle=3,filtered=True)

    deeplabcut.create_labeled_video('C:\\myproject\\reaching-task\\config.yaml',['C:\\myproject\\trailtracking-task\\test.mp4'],shuffle=3,filtered=True)
    --------

    Returns filtered pandas array with the same structure as normal output of network.
    """
    cfg = auxiliaryfunctions.read_config(config)
    DLCscorer,DLCscorerlegacy=auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction = cfg['TrainingFraction'][trainingsetindex])
    Videos=auxiliaryfunctions.Getlistofvideos(video,videotype)

    if len(Videos)>0:
        for video in Videos:
            if destfolder is None:
                destfolder = str(Path(video).parents[0])

            print("Filtering with %s model %s"%(filtertype,video))
            videofolder = destfolder
            vname=Path(video).stem
            notanalyzed,outdataname,sourcedataname,scorer=auxiliaryfunctions.CheckifPostProcessing(destfolder,vname,DLCscorer,DLCscorerlegacy,suffix='filtered')
            if notanalyzed:
                    Dataframe = pd.read_hdf(sourcedataname,'df_with_missing')
                    for bpindex,bp in tqdm(enumerate(cfg['bodyparts'])):
                        pdindex = pd.MultiIndex.from_product([[scorer], [bp], ['x', 'y','likelihood']],names=['scorer', 'bodyparts', 'coords'])
                        x,y,p=Dataframe[scorer][bp]['x'].values,Dataframe[scorer][bp]['y'].values,Dataframe[scorer][bp]['likelihood'].values

                        if filtertype=='arima':
                            meanx,CIx=FitSARIMAXModel(x,p,p_bound,alpha,ARdegree,MAdegree,False)
                            meany,CIy=FitSARIMAXModel(y,p,p_bound,alpha,ARdegree,MAdegree,False)

                            meanx[0]=x[0]
                            meany[0]=y[0]
                        else:
                            meanx=signal.medfilt(x,kernel_size=windowlength)
                            meany=signal.medfilt(y,kernel_size=windowlength)

                        if bpindex==0:
                            data = pd.DataFrame(np.hstack([np.expand_dims(meanx,axis=1),np.expand_dims(meany,axis=1),np.expand_dims(p,axis=1)]), columns=pdindex)
                        else:
                            item=pd.DataFrame(np.hstack([np.expand_dims(meanx,axis=1),np.expand_dims(meany,axis=1),np.expand_dims(p,axis=1)]), columns=pdindex)
                            data=pd.concat([data.T, item.T]).T

                    data.to_hdf(outdataname, 'df_with_missing', format='table', mode='w')
                    if save_as_csv:
                        print("Saving filtered csv poses!")
                        data.to_csv(outdataname.split('.h5')[0]+'.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('videos')
    cli_args = parser.parse_args()
