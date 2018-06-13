"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates the scorer's labels and computes the accuracy, and if plotting is set to 1, will also plot the train and test images
with the human labels (+), DeepLabCut's confident labels (.), and DeepLabCut's labels with less then pcutoff likelihood as (x). 

See Fig 7A in our preprint https://arxiv.org/abs/1804.03142v1 for illustration of pcutoff. 
"""

####################################################
# Dependencies
####################################################

import os.path
import sys
subfolder = os.getcwd().split('Evaluation-Tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
#sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(os.path.join(subfolder, "Generating_a_Training_Set"))

import auxiliaryfunctions
import pickle
from myconfig import Task, date, scorer, Shuffles, TrainingFraction, snapshotindex, pcutoff, plotting

if plotting==True:
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage import io

import numpy as np
import pandas as pd

####################################################
# Auxiliary functions
####################################################
# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def MakeLabeledImage(DataCombined,imagenr,imagefilename,Scorers,bodyparts,colors,labels=['+','.','x'],scaling=1,alphavalue=.5,dotsize=15):
    '''Creating a labeled image with the original human labels, as well as the DeepLabCut's!'''
    plt.axis('off')
    im=io.imread(os.path.join(imagefilename,DataCombined.index[imagenr]))
    if np.ndim(im)>2:
        h,w,numcolors=np.shape(im)
    else:
        h,w=np.shape(im)
    plt.figure(frameon=False,figsize=(w*1./100*scaling,h*1./100*scaling))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)    
    plt.imshow(im,'gray')
    for scorerindex,loopscorer in enumerate(Scorers):
       for bpindex,bp in enumerate(bodyparts):
           if np.isfinite(DataCombined[loopscorer][bp]['y'][imagenr]+DataCombined[loopscorer][bp]['x'][imagenr]):
                y,x=int(DataCombined[loopscorer][bp]['y'][imagenr]), int(DataCombined[loopscorer][bp]['x'][imagenr])
                if 'DeepCut' in loopscorer:
                    p=DataCombined[loopscorer][bp]['likelihood'][imagenr]
                    if p>pcutoff:
                        plt.plot(x,y,labels[1],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                    else:
                        plt.plot(x,y,labels[2],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                else: #by exclusion this is the human labeler (I hope nobody has DeepCut in his name...)
                        plt.plot(x,y,labels[0],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()    
    return 0

def pairwisedistances(DataCombined,scorer1,scorer2,pcutoff=-1,bodyparts=None):
    mask=DataMachine[scorer2].xs('likelihood',level=1,axis=1)>=pcutoff
    if bodyparts==None:
            Pointwisesquareddistance=(DataCombined[scorer1]-DataCombined[scorer2])**2
            MSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1))
            return MSE,MSE[mask]
    else:
            Pointwisesquareddistance=(DataCombined[scorer1][bodyparts]-DataCombined[scorer2][bodyparts])**2
            MSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1))
            return MSE,MSE[mask]

fs = 15  # fontsize for plots
####################################################
# Loading dependencies
####################################################

# loading meta data / i.e. training & test files
basefolder = os.path.join('..','pose-tensorflow','models')
folder = os.path.join('UnaugmentedDataSet_' + Task + date)
datafolder = os.path.join(basefolder,folder)

Data = pd.read_hdf(os.path.join(datafolder , 'data-' + Task , 'CollectedData_' + scorer + '.h5'),'df_with_missing')

####################################################
# Models vs. benchmark for varying training state
####################################################

# only specific parts can also be compared (not all!) (in that case change which bodyparts by providing a list below)
comparisonbodyparts = list(np.unique(Data.columns.get_level_values(1)))
if plotting==True:
    colors = get_cmap(len(comparisonbodyparts))
    
for trainFraction in TrainingFraction:
    for shuffle in Shuffles:

        fns = [
            file for file in os.listdir('Results')
            if "forTask_" + str(Task) in file and "shuffle" + str(shuffle) in
            file and "_" + str(int(trainFraction * 100)) in file
        ]

        metadatafile =os.path.join(datafolder , "Documentation_" + "data-" + Task + "_" + str(
            int(trainFraction * 100)) + "shuffle" + str(shuffle) + ".pickle")
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
          DataMachine = pd.read_hdf(os.path.join("Results",fns[index]), 'df_with_missing')
          DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
          scorer_machine = DataMachine.columns.get_level_values(0)[0]
          MSE,MSEpcutoff = pairwisedistances(DataCombined, scorer, scorer_machine,pcutoff,comparisonbodyparts)
          testerror = np.nanmean(MSE.iloc[testIndexes].values.flatten())
          trainerror = np.nanmean(MSE.iloc[trainIndexes].values.flatten())
          testerrorpcutoff = np.nanmean(MSEpcutoff.iloc[testIndexes].values.flatten())
          trainerrorpcutoff = np.nanmean(MSEpcutoff.iloc[trainIndexes].values.flatten())
          print("Results for",trainingiterations, "training iterations:", int(100 * trainFraction), shuffle, "train error:",np.round(trainerror,2), "pixels. Test error:", np.round(testerror,2)," pixels.")
          print("With pcutoff of", pcutoff," train error:",np.round(trainerrorpcutoff,2), "pixels. Test error:", np.round(testerrorpcutoff,2), "pixels")
          
          if plotting==True:
             foldername=os.path.join('LabeledImages_'+scorer_machine)
             auxiliaryfunctions.attempttomakefolder(foldername)
             NumFrames=np.size(DataCombined.index)
             for ind in np.arange(NumFrames):
                 fn=DataCombined.index[ind]
                 
                 fig=plt.figure()
                 ax=fig.add_subplot(1,1,1)
                 MakeLabeledImage(DataCombined,ind,os.path.join(datafolder,'data-'+Task),[scorer,scorer_machine],comparisonbodyparts,colors)
                 if ind in trainIndexes:
                     plt.savefig(os.path.join(foldername,'TrainingImg'+str(ind)+'_'+fn.split('/')[0]+'_'+fn.split('/')[1]))
                 else:
                     plt.savefig(os.path.join(foldername,'TestImg'+str(ind)+'_'+fn.split('/')[0]+'_'+fn.split('/')[1]))
                 plt.close("all")
