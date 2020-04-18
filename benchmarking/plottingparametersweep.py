import deeplabcut, pickle, os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage import io
import multiprocessing as mp
import deeplabcut


#plot parameters
width=.5
cmap='jet'
lw=2.5
ms=10

params = {
   'axes.labelsize': 22,
   'legend.fontsize': 10,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18,
   'text.usetex': False,
   'figure.figsize': [8,8],
   'font.size': 22,
   'axes.linewidth': 2,
   'xtick.major.size': 5,
   'xtick.major.width': 2,
   'ytick.major.size': 5,
   'ytick.major.width': 2
   }

plt.rcParams.update(params) #this allows global definition of parameters ...
def figurchen():
    fig = plt.figure()
    ax = fig.add_subplot(111,position=[0.14,.14, .84,.84]) #[left, bottom, width, height]
    #ax.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)  #allows turning off spines... i.e. top and right is good :)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

def SaveResults(datafilename, data):
        with open(datafilename, 'wb') as f:
            pickle.dump(data, f,pickle.HIGHEST_PROTOCOL)


def LoadResults(datafile):
    with open(datafile, 'rb') as f:
        return pickle.load(f)

def fraction(a,b,default=0.):
    z=np.ones(np.shape(a))*default
    z[b!=0.]=a[b!=0.]*1./b[b!=0.]
    return z

Shuffles=[0]
basepath='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16'
animal='bird'
modelprefix=''


configfile=os.path.join(basepath,'config.yaml')
cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
individuals, uniquebodyparts,multianimalbodyparts = deeplabcut.utils.auxfun_multianimal.extractindividualsandbodyparts(cfg)


#for modelsuffix in Models:

deeplabcut.utils.attempttomakefolder(os.path.join(animal,'SweepPlots'),recursive=True)

#modelprefix='DLC-benchmarking'+modelsuffix
#returns pickle file paths for all snapshots of that type!
imagefolder=animal+'/' #+modelsuffix

summaryresultsfn=os.path.join(animal,'SweepPlots/summary.pickle')
if os.path.isfile(summaryresultsfn):
    print("loading precomputed results...")
    Data,Dataperbpt,Parameters,columnresults,columnsparams,columnresultsperbpt=LoadResults(summaryresultsfn)
else:
    rstfn=[fn for fn in os.listdir(str(os.path.join(basepath,modelprefix,'quantificationassembly'))) if '.pickle' in fn and 'GTmask' not in fn]
    numvariants=len(rstfn)
    Data=np.zeros((numvariants,10))
    Parameters=np.zeros((numvariants,8))
    columnresults =['misses','hits','labeledbutnotbyhuman','samples','numlabeledGTbpts','numanimalsGT','numanimalsalgo','rmse','closestrmse','fractionclosest']
    columnresultsperbpt =['misses','hits','labeledbutnotbyhuman','numlabeledGTbpts','rmse','closestrmse','fractionclosest']
    columnsparams = ['minimalnumberofconnections', 'averagescore','distmax','detectionthresholdsquare','distmin','addlikelihoods','pafthreshold','method']

    #StatspBPtudate=np.zeros((3,7600,5,15))
    for variant,resultsfn in enumerate(rstfn):
        #resultsfilename=os.path.join(animal,modelsuffix+'_inferred.pickle')
        resultsfn=os.path.join(basepath,modelprefix,'quantificationassembly',resultsfn)
        RMSE,RMSEclosestbpt,Stats,StatsperBpt,inferencecfg,PAF=LoadResults(resultsfn)
        numshuffles=np.shape(RMSE)[0]
        if variant==0:
            GTmask=LoadResults(os.path.join(basepath,'quantificationassembly','GTmask.pickle'))
            GTmask=np.array(GTmask,dtype=bool)
            GTmask=np.repeat(np.expand_dims(GTmask,axis=0),numshuffles,axis=0)

            numbodyparts=np.shape(StatsperBpt[:,:,1,:])[-1]
            Dataperbpt=np.zeros((numvariants,7,numbodyparts))
            print(np.shape(Stats),np.shape(StatsperBpt))

        #fraction of misses
        Data[variant,0]=np.nanmean(Stats[:,:,0]*1./Stats[:,:,4],axis=(0,1)) #average performance / do test/train!
        #fraction of hits
        Data[variant,1]=np.nanmean(Stats[:,:,1]*1./Stats[:,:,4],axis=(0,1)) #average performance / do test/train!
        #fraction of additional bpts!
        Data[variant,2]=np.nanmean(Stats[:,:,2]*1./Stats[:,:,4],axis=(0,1)) #average performance / do test/train!
        #is it different for train/test?

        Data[variant,3]=np.nanmean(Stats[:,:,3]) #num samples (union of algo + human)
        Data[variant,4]=np.nanmean(Stats[:,:,4]) #num bodyparts labeled (by human)

        Data[variant,5]=np.nanmean(Stats[:,:,5]) #num animals by human
        Data[variant,6]=np.nanmean(Stats[:,:,6]) #num animals by algorithm

        Data[variant,7]=np.nanmean(RMSE)
        Data[variant,8]=np.nanmean(RMSEclosestbpt) #closest to GT bpt of that individual


        #fraction of bpts closer to GT than for assembled individuals!
        Data[variant,9]=np.nanmean((RMSE<=RMSEclosestbpt)[GTmask])

        #misses
        Dataperbpt[variant,0,:]=np.nanmean(fraction(StatsperBpt[:,:,0,:],StatsperBpt[:,:,4,:],default=np.nan),axis=(0,1))
        #hits
        Dataperbpt[variant,1,:]=np.nanmean(fraction(StatsperBpt[:,:,1,:],StatsperBpt[:,:,4,:],default=np.nan),axis=(0,1))
        #additional bpts
        Dataperbpt[variant,2,:]=np.nanmean(fraction(StatsperBpt[:,:,2,:],StatsperBpt[:,:,4,:],default=np.nan),axis=(0,1))
        Dataperbpt[variant,3,:]=np.nanmean(StatsperBpt[:,:,4,:],axis=(0,1)) #average scored per bpt!

        Dataperbpt[variant,4,:]=np.nanmean(RMSE,axis=(0,1,3)) #average scored per bpt!

        Dataperbpt[variant,5,:]=np.nanmean(RMSEclosestbpt,axis=(0,1,3)) #average scored per bpt!

        better=(RMSE<=RMSEclosestbpt)*GTmask
        better[better==0]=np.nan #remove for mean!
        Dataperbpt[variant,6,:]=np.mean(better,axis=(0,1,3)) #fraction within animal closer than closest *


        #num animals!
        Parameters[variant,0]=inferencecfg.minimalnumberofconnections
        Parameters[variant,1]=inferencecfg.averagescore
        Parameters[variant,2]=inferencecfg.distnormalization
        Parameters[variant,3]=inferencecfg.detectionthresholdsquare
        Parameters[variant,4]=inferencecfg.distnormalizationLOWER
        Parameters[variant,5]=inferencecfg.addlikelihoods
        Parameters[variant,6]=inferencecfg.pafthreshold
        Parameters[variant,7]=(inferencecfg.method=='m1') #boolean!

    SaveResults(summaryresultsfn,[Data,Dataperbpt,Parameters,columnresults,columnsparams,columnresultsperbpt])

#now analysis:
#columnresults =['misses','hits','labeledbutnotbyhuman','samples','numlabeledGTbpts','numanimalsGT','numanimalsalgo','rmse']

#columnsparams = ['minimalnumberofconnections', 'averagescore','distmax','detectionthresholdsquare','distmin','addlikelihoods','pafthreshold','method']

for j in range(8):
    figurchen()
    values=set(Parameters[:,j])
    for v in values:
        mask=np.array(Parameters[:,j]==v,dtype=bool)
        plt.plot(Data[mask,1],Data[mask,2],'.',label=v) #my method
        #plt.title(columnsparams[j])
    plt.legend()
    plt.xlabel("Hit rate")
    plt.ylabel("Additional detections")
    plt.savefig(os.path.join(animal,'SweepPlots/comparison2_'+str(columnsparams[j])+'.png'))

for j in range(8):
    figurchen()
    values=set(Parameters[:,j])
    for v in values:
        mask=np.array(Parameters[:,j]==v,dtype=bool)
        plt.plot(Data[mask,1],Data[mask,9],'.',label=v) #my method
        #plt.title(columnsparams[j])
    plt.legend()
    plt.xlabel("Hit rate")
    plt.ylabel("Assembled better than closest")
    plt.savefig(os.path.join(animal,'SweepPlots/comparison3_'+str(columnsparams[j])+'.png'))

for j in range(8):
    figurchen()
    values=set(Parameters[:,j])
    for v in values:
        mask=np.array(Parameters[:,j]==v,dtype=bool)
        plt.plot(Data[mask,5],Data[mask,6],'.',label=v) #my method
        #plt.title(columnsparams[j])
    plt.legend()
    plt.xlabel("Num animals by algorithm")
    plt.ylabel("Num animals")
    plt.savefig(os.path.join(animal,'SweepPlots/comparison4_'+str(columnsparams[j])+'.png'))

for j in range(8):
    figurchen()
    values=set(Parameters[:,j])
    for v in values:
        mask=np.array(Parameters[:,j]==v,dtype=bool)
        plt.plot(Data[mask,0],Data[mask,2],'.',label=v) #my method
        #plt.title(columnsparams[j])
    plt.legend()
    plt.ylabel("Additional detections")
    plt.xlabel("Misses")
    plt.savefig(os.path.join(animal,'SweepPlots/comparison5'+str(columnsparams[j])+'.png'))

print(columnsparams)
for j in range(8):
    figurchen()
    values=set(Parameters[:,j])
    for v in values:
        mask=np.array(Parameters[:,j]==v,dtype=bool)
        plt.plot(Data[mask,1],Data[mask,6],'.',label=v) #my method
        #plt.title(columnsparams[j])
    plt.legend()
    plt.xlabel("Correctly detected bodyparts")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(animal,'SweepPlots/comparisonRMSE'+str(columnsparams[j])+'.png'))

#columnresultsperbpt =['misses','hits','labeledbutnotbyhuman','numlabeledGTbpts','rmse']
topk=5
print(np.sort(Data[:,6]))
topvalues=np.argsort(Data[:,6])[:topk]

figurchen()
for l in topvalues:
    plt.plot(Dataperbpt[l,1,:],'.',ms=ms) #my method
    print("top rmse",l,Parameters[l,:])
plt.xlabel("Bodypart index")
plt.ylabel("Detection bodyparts")
plt.savefig(os.path.join(animal,'SweepPlots/hitsperbpt_sortedbyRMSE.png'))
plt.savefig(os.path.join(animal,'SweepPlots/hitsperbpt_sortedbyRMSE.pdf'))

## TOP OVERALL HIT RATES:
topvalues=np.argsort(-Data[:,1])[:topk]
figurchen()
for l in topvalues:
    plt.plot(Dataperbpt[l,1,:],'.',ms=ms) #my method
    print("top hits",l,Parameters[l,:])
plt.xlabel("Bodypart index")
plt.ylabel("Detection bodyparts")
plt.savefig(os.path.join(animal,'SweepPlots/hitsperbpt_sortedbytophitrate.png'))
plt.savefig(os.path.join(animal,'SweepPlots/hitsperbpt_sortedbytophitrate.pdf'))

figurchen()
for l in topvalues:
    plt.plot(Dataperbpt[l,0,:],'.',ms=ms) #my method
    print("top hits",l,Parameters[l,:])
plt.xlabel("Bodypart index")
plt.ylabel("Missed bodyparts")
plt.savefig(os.path.join(animal,'SweepPlots/missesperbpt_sortedbytophitrate.png'))
plt.savefig(os.path.join(animal,'SweepPlots/missesperbpt_sortedbytophitrate.pdf'))

figurchen()
l=topvalues[0]

plt.plot(Dataperbpt[l,3,:],'.',ms=ms,label='animals labeled') #my method
plt.plot(Dataperbpt[l,2,:],'.',ms=ms,label='additional labels') #my method

plt.xlabel("Bodypart index")
plt.ylabel("Missed bodyparts")
plt.savefig(os.path.join(animal,'SweepPlots/numlabeledGTbpts_sortedbytophitrate.png'))
plt.savefig(os.path.join(animal,'SweepPlots/mnumlabeledGTbpts_sortedbytophitrate.pdf'))

plt.close("all")
