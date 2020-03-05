import deeplabcut, pickle, os, sys
from deeplabcut.utils import auxfun_multianimal
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm
import cv2
from skimage import io
import multiprocessing as mp
import quantificationutils

def SaveResults(datafilename, data):
        with open(datafilename, 'wb') as f:
            pickle.dump(data, f,pickle.HIGHEST_PROTOCOL)

def LoadResults(datafile):
    with open(datafile, 'rb') as f:
        return pickle.load(f)

def distance(v,w,single=True):
    if (len(v)*len(w))>0:
        return np.sqrt(np.sum((v-w)**2))
    else:
        return 0 #TBA

def getfinitevalues(array,indices=None):
    if indices is None:
        return array[np.isfinite(array)]
    else:
        a=array[indices]
        return a[np.isfinite(a)]

def getparams(variant):
    v=0
    for minimalnumberofconnections in [2,3,4]:
        for averagescore in [.1,.25,.5]:
            for distmax in [200,350]:
                for detectionthresholdsquare in [0,0.025,0.1]:
                    for distmin in [0,20]:
                        for addlikelihoods in [0,1]:
                            for pafthreshold in [0,.1,.2]: #[.1,.2]: # MARMOSET: [0,.1,.2]:
                                for method in ['m1','m2']:
                                    if v==variant:
                                        print("retrieving...", variant)
                                        ##########parameters for interence
                                        inferencecfg=edict()
                                        inferencecfg.minimalnumberofconnections=minimalnumberofconnections
                                        inferencecfg.averagescore=averagescore
                                        inferencecfg.distnormalization=distmax
                                        inferencecfg.distnormalizationLOWER=distmin
                                        inferencecfg.detectionthresholdsquare=detectionthresholdsquare
                                        inferencecfg.addlikelihoods=addlikelihoods
                                        inferencecfg.pafthreshold=pafthreshold
                                        inferencecfg.method=method
                                        inferencecfg.withid=False
                                        inferencecfg.slack=10
                                        inferencecfg.variant=variant

                                        return inferencecfg
                                    v+=1
    return v

Shuffles=[0]
basepath='/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/examples-multianimal/MontBlanc-Daniel-2019-12-16'
animal='bird'
modelprefix=''

Shuffles=[1,2]
basepath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'
animal='mouse'
modelprefix=''

#basepath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/Marmoset-Mackenzie-2019-05-29'
#animal='marmoset'
#modelprefix='DLC-benchmarkinggoodbaseline3'

configfile=os.path.join(basepath,'config.yaml')
cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
individuals, uniquebodyparts,multianimalbodyparts = auxfun_multianimal.extractindividualsandbodyparts(cfg)

#modelprefix=''
SearchVariant='Vanilla'
PAF=None
printoutput=False
def runinstance1(variant): #,Models,Shuffles,configfile,animal):
    print("Starting",variant)
    deeplabcut.auxiliaryfunctions.attempttomakefolder(os.path.join(basepath,'quantificationassembly'))

    resultsfn=os.path.join(basepath,'quantificationassembly',SearchVariant+'_parametersearch'+str(variant)+'_inferred.pickle')

    if os.path.isfile(resultsfn):
        print("Data already exists...")
    else:
        inferencecfg=getparams(variant)
        #if 'idchannel' in modelsuffix:
        #    #also extracting identity
        #    inferencecfg.withid=True

        if variant==0: #also calculates GTmask
            results=quantificationutils.quantifiyassembly(cfg,Shuffles,configfile,modelprefix,inferencecfg,printoutput=printoutput,PAF=PAF,calculatemask=True)
            GTmask=results[-1]
            SaveResults(os.path.join(basepath,'quantificationassembly','GTmask.pickle'), GTmask)
            SaveResults(resultsfn, results[:-1])
        else:
            results=quantificationutils.quantifiyassembly(cfg,Shuffles,configfile,modelprefix,inferencecfg)
            SaveResults(resultsfn, results)

        print("Saving...",resultsfn)
    return 0

#runinstance1(0)


#run in parallel!
numvalues=getparams(np.nan) #dumb trick!
print("Starting, ", animal, "with ", numvalues, "variants!!!")
pool = mp.Pool(mp.cpu_count())
results = pool.map(runinstance1, range(numvalues))
#results = pool.map(runinstance2, range(numvalues))

pool.close()
#runinstance(variant,Models,Shuffles,configfile,modelprefix,inferencecfg,animal)
