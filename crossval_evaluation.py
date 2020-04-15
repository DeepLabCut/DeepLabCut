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
from deeplabcut.utils import auxfun_multianimal
import numpy as np
import pandas as pd

#sys.path.append(os.path.join('/media/alex/dropboxdisk/Dropbox/Collaborations/Cancer/DLCdev/deeplabcut/pose_estimation_tensorflow/lib'))
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils


projectpath='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/croppedNov18/MultiMouse-Daniel-2019-12-16'
configfile=os.path.join(projectpath,'config.yaml')

dlccfg=deeplabcut.auxiliaryfunctions.read_config(configfile)
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


#def quantifiyassembly(dlccfg, Shuffles, configfile, modelprefix,inferencecfg,PAF=None,printoutput=False,calculatemask=False):
TrainIndices=[]
TestIndices=[]
individuals, uniquebodyparts,multianimalbodyparts = auxfun_multianimal.extractindividualsandbodyparts(dlccfg)
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
    nummultianimalparts=len(dlccfg['multianimalbodyparts'])

trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
Data = pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')
    
imgid=0
#for imgid in tqdm(range(numimages)):
if imgid ==0:
        imname=imnames[imgid]
        detectedcoordinates = data[imname]['prediction']['coordinates'][0]
        detectedlikelihood = data[imname]['prediction']['confidence']
        groundtruthidentity, groundtruthcoordinates, GT=data[imname]['groundtruth']

        #animals = inferenceutils.assemble_individuals(inferencecfg, data[imname], numjoints, BPTS, iBPTS,
        #                                                          PAF, partaffinityfield_graph, linkingpartaffinityfield_graph,
        #                                                          lowerbound,upperbound,printintermediate)

        all_detections = inferenceutils.convertdetectiondict2listoflist(data[imname],BPTS,withid=inferencecfg.withid,evaluation=True)

        # filter connections according to inferencecfg parameters
        #connection_all, missing_connections = extractstrongconnections(inference_cfg, data,
        #                                         all_detections, iBPTS, 
        #                                         paf_graph, PAF, lowerbound, upperbound)

        connection_all, missing_connections = inferenceutils.extractstrongconnections(inferencecfg,data[imname],
                                    all_detections, iBPTS, partaffinityfield_graph, PAF,evaluation=True)
        
        subsets, candidate = inferenceutils.linkjoints2individuals(inferencecfg, all_detections, connection_all, missing_connections,
                                                    partaffinityfield_graph, iBPTS, numjoints=numjoints,log=False)


[all_jointnames[i] for i in BPTS]

distances=np.zeros((len(subsets),))

# distance for each detection and ground truth object
for individual in individuals:
    GT[cfg['scorer']]
if subset is not None:
    for n in len(subset):  # range(len(subset)): #number of individuals
        individual = np.zeros(3 * numjoints) * np.nan
        for i in range(numjoints):  # number of joints
            ind = int(subset[n][i])  # bpt index in global coordinates
            if -1 == ind:  # not assigned
                continue
            else:  # xyl=np.ones(3)*np.nan
                individual[3 * i:3 * i + 3] = candidate[ind, :3]

            numassembledindividuals=len(subset[:,-2])

            #for each GT individual find the best assembled individual!
            ComparisonCoordinates=np.zeros((numindividuals,numjoints,2,2))*np.nan
            for gtindindex,ids in enumerate(individuals):
                if numassembledindividuals>0:
                    costs=np.zeros(numassembledindividuals)*np.nan #minimal cost (euclidian distance per individual)
                    distancesperbpt=np.zeros((numjoints,numassembledindividuals))*np.nan
                    for individualindex,n in enumerate(subset[:,-2]): # loop over number of individuals
                        for index,content in enumerate(subset[individualindex][:-2]): #loop over links!
                            if content!=-1:
                                coordinates_individual=np.argwhere(groundtruthidentity[BPTS[index]]==ids) #all coordinates for bpt index, and indiv. ids
                                #coordinates_individual=np.argwhere(groundtruthidentity[index]==ids) #all coordinates for bpt index, and indiv. ids
                                if len(coordinates_individual)>0:
                                    gt=groundtruthcoordinates[BPTS[index]][coordinates_individual[0][0]]
                                    inferred=candidate[subset[individualindex][index].astype(np.int)][:2]
                                    distance=np.linalg.norm(gt-inferred)
                                    if np.isnan(costs[individualindex]):
                                        costs[individualindex]=distance
                                    else:
                                        costs[individualindex]+=distance

                                    distancesperbpt[index,individualindex]=distance
                    if np.any(np.isfinite(distancesperbpt)):
                        RMSEclosestbpt[shuffle,imgid,:,gtindindex]=np.nanmin(distancesperbpt,axis=1) #closest bodyparts to GT individual for all detected animals!
                if numassembledindividuals>0:
                    if np.any(np.isfinite(costs)):
                        assigned_assembledanimal=np.nanargmin(costs)
                        for index,content in enumerate(subset[assigned_assembledanimal][:-2]): #loop over links!
                            if content!=-1:
                                ComparisonCoordinates[gtindindex,index,0,:]=candidate[subset[assigned_assembledanimal][index].astype(np.int)][:2]

                            coordinates_individual=np.argwhere(groundtruthidentity[BPTS[index]]==ids)
                            if len(coordinates_individual)>0:
                                ComparisonCoordinates[gtindindex,index,1,:]=groundtruthcoordinates[BPTS[index]][coordinates_individual[0][0]]


            RMSE[shuffle,imgid,:,:]=np.sqrt(np.sum((ComparisonCoordinates[:,:,1,:]-ComparisonCoordinates[:,:,0,:])**2,axis=2)).T



'''
    if shuffleval==Shuffles[0]: #initialize variables
        numshuffles=len(Shuffles)
        ## COMPARE WITH BEST MATCH + score!
        numindividuals=len(individuals)
        RMSE=np.zeros((numshuffles,numimages,numjoints,numindividuals))*np.nan
        RMSEclosestbpt=np.zeros((numshuffles,numimages,numjoints,numindividuals))*np.nan
        Stats=np.zeros((numshuffles,numimages,7))*np.nan
        StatsperBpt=np.zeros((numshuffles,numimages,5,numjoints))*np.nan

    if calculatemask:
        GTmask=np.zeros((numimages,numjoints,numindividuals))*False
        for imgid in range(numimages):
            imname=imnames[imgid]
            groundtruthidentity, groundtruthcoordinates,GT=data[imname]['groundtruth']
            for gtindindex,ids in enumerate(individuals):
                for index,bptind in enumerate(BPTS):
                    if len(np.argwhere(groundtruthidentity[bptind]==ids))>0: #Note when bodypart exists!
                        GTmask[imgid,index,gtindindex]=True

    #for imgid in tqdm(range(numimages)):
    for imgid in range(numimages):
        imname=imnames[imgid]
        detectedcoordinates=data[imname]['prediction']['coordinates'][0]
        detectedlikelihood=data[imname]['prediction']['confidence']
        groundtruthidentity, groundtruthcoordinates,GT=data[imname]['groundtruth']

        all_detections=inferenceutils.convertdetectiondict2listoflist(data[imname],imname,BPTS,withid=inferencecfg.withid,evaluation=True)

        connection_all, special_k=inferenceutils.matchconnections(inferencecfg,data[imname],
                                    all_detections, iBPTS, partaffinityfield_graph, PAF,evaluation=True)
        try:
            subset, candidate=inferenceutils.linkjoints2individuals(inferencecfg, all_detections, connection_all, special_k,
                                                    partaffinityfield_graph, iBPTS, numjoints=numjoints,log=False)
        except:
            #print("failed for ", inferencecfg.variant,imgid)
            subset=None

        if subset is not None:
            numassembledindividuals=len(subset[:,-2])

            #for each GT individual find the best assembled individual!
            ComparisonCoordinates=np.zeros((numindividuals,numjoints,2,2))*np.nan
            for gtindindex,ids in enumerate(individuals):
                if numassembledindividuals>0:
                    costs=np.zeros(numassembledindividuals)*np.nan #minimal cost (euclidian distance per individual)
                    distancesperbpt=np.zeros((numjoints,numassembledindividuals))*np.nan
                    for individualindex,n in enumerate(subset[:,-2]): # loop over number of individuals
                        for index,content in enumerate(subset[individualindex][:-2]): #loop over links!
                            if content!=-1:
                                coordinates_individual=np.argwhere(groundtruthidentity[BPTS[index]]==ids) #all coordinates for bpt index, and indiv. ids
                                #coordinates_individual=np.argwhere(groundtruthidentity[index]==ids) #all coordinates for bpt index, and indiv. ids
                                if len(coordinates_individual)>0:
                                    gt=groundtruthcoordinates[BPTS[index]][coordinates_individual[0][0]]
                                    inferred=candidate[subset[individualindex][index].astype(np.int)][:2]
                                    distance=np.linalg.norm(gt-inferred)
                                    if np.isnan(costs[individualindex]):
                                        costs[individualindex]=distance
                                    else:
                                        costs[individualindex]+=distance

                                    distancesperbpt[index,individualindex]=distance
                    if np.any(np.isfinite(distancesperbpt)):
                        RMSEclosestbpt[shuffle,imgid,:,gtindindex]=np.nanmin(distancesperbpt,axis=1) #closest bodyparts to GT individual for all detected animals!
                if numassembledindividuals>0:
                    if np.any(np.isfinite(costs)):
                        assigned_assembledanimal=np.nanargmin(costs)
                        for index,content in enumerate(subset[assigned_assembledanimal][:-2]): #loop over links!
                            if content!=-1:
                                ComparisonCoordinates[gtindindex,index,0,:]=candidate[subset[assigned_assembledanimal][index].astype(np.int)][:2]

                            coordinates_individual=np.argwhere(groundtruthidentity[BPTS[index]]==ids)
                            if len(coordinates_individual)>0:
                                ComparisonCoordinates[gtindindex,index,1,:]=groundtruthcoordinates[BPTS[index]][coordinates_individual[0][0]]


            RMSE[shuffle,imgid,:,:]=np.sqrt(np.sum((ComparisonCoordinates[:,:,1,:]-ComparisonCoordinates[:,:,0,:])**2,axis=2)).T

            ##misses (annotated bpts that were not detected)
            StatsperBpt[shuffle,imgid,0,:]=np.sum(np.isfinite(ComparisonCoordinates[:,:,1,:])*np.isnan(ComparisonCoordinates[:,:,0,:]),axis=(0,2))/2
            Stats[shuffle,imgid,0]=np.sum(StatsperBpt[shuffle,imgid,0,:])

            ## hits (annotated bpts that were detected)
            StatsperBpt[shuffle,imgid,1,:]=np.sum(np.isfinite(ComparisonCoordinates[:,:,1,:])*np.isfinite(ComparisonCoordinates[:,:,0,:]),axis=(0,2))/2
            Stats[shuffle,imgid,1]=np.sum(StatsperBpt[shuffle,imgid,1,:]) #don't count x & y twice!

            ## additional detections >> algo detected bpt, but human did not label...
            StatsperBpt[shuffle,imgid,2,:]=np.sum(np.isnan(ComparisonCoordinates[:,:,1,:])*np.isfinite(ComparisonCoordinates[:,:,0,:]),axis=(0,2))/2
            Stats[shuffle,imgid,2]=np.sum(StatsperBpt[shuffle,imgid,2,:])

            #samples
            StatsperBpt[shuffle,imgid,3,:]=np.sum((np.isfinite(ComparisonCoordinates[:,:,1,:])+np.isfinite(ComparisonCoordinates[:,:,0,:]))>0,axis=(0,2))/2
            Stats[shuffle,imgid,3]=np.sum(StatsperBpt[shuffle,imgid,3,:])

            #numbpts by human!
            StatsperBpt[shuffle,imgid,4,:]=np.sum(np.isfinite(ComparisonCoordinates[:,:,1,:]),axis=(0,2))/2
            Stats[shuffle,imgid,4]=np.sum(StatsperBpt[shuffle,imgid,4,:])

            #num animals!
            Stats[shuffle,imgid,5]=np.sum(np.any(np.isfinite(ComparisonCoordinates[:,:,1,:]),axis=(1,2)))

            #num animals by algorithm!
            Stats[shuffle,imgid,6]=np.sum(np.any(np.isfinite(ComparisonCoordinates[:,:,0,:]),axis=(1,2)))
            #print(subset,imgid)
            #print(RMSE[shuffle,imgid])
            #print(Stats[shuffle,imgid])
        if printoutput:
            print("misses",Stats[shuffle,imgid,0])
            print("hits",Stats[shuffle,imgid,1])
            print("maybe FA",Stats[shuffle,imgid,2])
            print('Samples: ',Stats[shuffle,imgid,3])
            print("error",np.nanmean(RMSE[shuffle,imgid,:,:]))
if calculatemask:
    return [RMSE,RMSEclosestbpt,Stats,StatsperBpt,inferencecfg,PAF,GTmask]
else:
    return [RMSE,RMSEclosestbpt,Stats,StatsperBpt,inferencecfg,PAF]


'''