import numpy as np
import os

os.environ['DLClight'] = 'True'
import pickle
import deeplabcut
import pandas as pd
from easydict import EasyDict as edict
from itertools import product
from tqdm import tqdm

import deeplabcut, pickle, os, sys
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
import numpy as np
import pandas as pd

#sys.path.append(os.path.join('/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/deeplabcut/pose_estimation_tensorflow/lib'))
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils


projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'
configfile=os.path.join(projectpath,'config.yaml')

cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
Shuffles=[2]
modelprefix=''
PAF=None

####
'''
path_inference_config = Path(modelfolder) / 'test' / 'inference_cfg.yaml'
if inferencecfg is None: #then load or initialize
    inferencecfg=auxfun_multianimal.read_inferencecfg(path_inference_config,cfg)
else: #TODO: check if all variables present
    inferencecfg=edict(inferencecfg)
'''

def metric(df1,df2,pcutoff=0):
    #mask=df2.xs('likelihood',level=1,axis=1)>=pcutoff
    
    dist=(df1-df2)**2
    RMSE=np.sqrt(dist.xs('x',level=1,axis=1)+dist.xs('y',level=1,axis=1)) #bpt-element wise RMSE

    return np.nanmean(RMSE)

from scipy.optimize import linear_sum_assignment


inferencecfg=edict()
inferencecfg.minimalnumberofconnections=3
inferencecfg.averagescore=.2
inferencecfg.distnormalization=400
inferencecfg.distnormalizationLOWER=0
inferencecfg.detectionthresholdsquare=.1
inferencecfg.addlikelihoods=0.
inferencecfg.pafthreshold=.1
inferencecfg.method='m1'

inferencecfg.withid=False
inferencecfg.slack=10
inferencecfg.variant=0 #


#def quantifiyassembly(cfg, Shuffles, configfile, modelprefix,inferencecfg,PAF=None,printoutput=False,calculatemask=False):
TrainIndices=[]
TestIndices=[]
individuals, uniquebodyparts,multianimalbodyparts = auxfun_multianimal.extractindividualsandbodyparts(cfg)

trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
Data = pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')


for shuffle, shuffleval in enumerate(Shuffles):
    #DLCdev version:
    #fns=deeplabcut.return_evaluate_network_data(configfile,shuffle=shuffleval,trainingsetindex=0,modelprefix=modelprefix,returnjustfns=True)
    #sDLC version:
    fns=deeplabcut.return_evaluate_network_data(configfile,shuffle=shuffleval,trainingsetindex=0,modelprefix=modelprefix)
    #print(fns)
    dataname=fns[-1]

    ########### Loading full detection file!
    data, metadata=deeplabcut.utils.auxfun_multianimal.LoadFullMultiAnimalData(dataname)

    #print(dataname,shuffleval)
    trainIndices=metadata['data']['trainIndices']
    testIndices=metadata['data']['testIndices']
    DLCscorer=metadata['data']['Scorer']
    dlc_cfg=metadata['data']['DLC-model-config file']

    TrainIndices.append(trainIndices)
    TestIndices.append(testIndices)

    nms_radius=data['metadata']['nms radius']
    minconfidence=data['metadata']['minimal confidence']
    partaffinityfield_graph=data['metadata']['PAFgraph']
    all_joints=data['metadata']['all_joints']
    all_jointnames=data['metadata']['all_joints_names']

    numjoints=len(all_joints)

    if PAF is None:
        PAF=np.arange(len(partaffinityfield_graph))

    partaffinityfield_graph=[partaffinityfield_graph[l] for l in PAF]

    BPTS=range(numjoints) 
    iBPTS=BPTS 

    numjoints=len(BPTS)
    numlimbs=len(partaffinityfield_graph)
    imnames=np.sort([fn for fn in data.keys() if fn !='metadata'])
    numimages=len(imnames)

    numuniquebodyparts=len(uniquebodyparts)
    numanimals=len(individuals)-1*(len(uniquebodyparts)>0) #number of individuals in ground truth (excluding single!)
    nummultianimalparts=len(cfg['multianimalbodyparts'])

    
imgid=33
#for imgid in tqdm(range(numimages)):
if imgid ==33:
        imname=imnames[imgid]
        detectedcoordinates = data[imname]['prediction']['coordinates'][0]
        detectedlikelihood = data[imname]['prediction']['confidence']
        groundtruthidentity, groundtruthcoordinates, GT=data[imname]['groundtruth']

        all_detections = inferenceutils.convertdetectiondict2listoflist(data[imname],BPTS,withid=inferencecfg.withid,evaluation=True)

        connection_all, missing_connections = inferenceutils.extractstrongconnections(inferencecfg,data[imname],
                                    all_detections, iBPTS, partaffinityfield_graph, PAF,evaluation=True)
        
        subsets, candidate = inferenceutils.linkjoints2individuals(inferencecfg, all_detections, connection_all, missing_connections,
                                                    partaffinityfield_graph, iBPTS, numjoints=numjoints,log=False)


bodyparts=[all_jointnames[i] for i in BPTS]
detectedindividuals=[str(j) for j in range(len(subsets))]
columnindex = pd.MultiIndex.from_product([detectedindividuals, bodyparts, ['x', 'y','likelihood']],names=['individuals','bodyparts', 'coords'])

elements=3 * numjoints
detections = np.zeros((1, elements*len(detectedindividuals))) * np.nan
#recode the data into a pandas array
for sj, subset in enumerate(subsets):
    if subset is not None:
            for i in range(numjoints):  # number of joints
                ind = int(subset[i])  # bpt index in global coordinates
                if -1 == ind:  # not assigned
                    continue
                else:  # xyl=np.ones(3)*np.nan
                    detections[0, elements*sj + 3 * i: elements*sj + 3 * i + 3] = candidate[ind, :3]

if sj>0:
    DF=pd.DataFrame(detections, columns=columnindex, index=[imname])


mat=np.zeros((len(individuals),len(detectedindividuals)))*np.nan
# distance for each detection and ground truth pose
for i, individual in enumerate(individuals):
    for j, ind in enumerate(detectedindividuals):
        rmse = metric(GT[cfg['scorer']][individual], DF[ind])

        mat[i, j] = rmse #rmse 

row_indices, col_indices = linear_sum_assignment(mat)

columnindex = pd.MultiIndex.from_product([individuals, ['rmse', 'f1','likelihood']],names=['individuals','metrics'])

#DF=pd.DataFrame(detections, columns=columnindex, index=[imname])

#calculate metrics for the matches:
for i, individual in enumerate(individuals): 
    match=detectedindividuals[col_indices[i]]
    
    df,dg=GT[cfg['scorer']][individual], DF[str(col_indices[i])]
    rmse = mat[i, col_indices[i]]

    #closest distance! rmse!